from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from ..base import BaseEstimator

if TYPE_CHECKING:
    import torch
    from datasets.schema import TrajectoryDataset


class SequentialNeuralFilter(BaseEstimator):
    """
    Shared scaffolding for the GPU-train / CPU-infer neural filters
    (Neural-ODE, PINN, Transformer, Mamba). It factors out everything the four
    estimators have in common with `KalmanNetEstimator` -- per-epoch
    train/val loop, best-checkpoint-in-memory by val loss, gradient clipping,
    optional LR scheduler, optional early stopping, NaN/Inf-loss skip, seeding,
    verbose printing, the `history_` dict, and the save() recipe -- so each
    concrete filter only supplies its network and its two forward passes.

    The hardware-split deployment contract (Issue 0) is enforced here by
    structure: subclasses implement
      _build_network()                                  -> nn.Module
      _forward_train(network, obs, states, ts, device)  -> x_hat [B, T, nx]  (GPU, batched, parallel)
      _loss(network, obs, states, ts, device)           -> scalar loss        (GPU; default: MSE of _forward_train)
      _estimate_sequential_cpu(network, obs, ts)        -> x_hat [N, T, nx]   (CPU, strictly sequential)
      _save_hyperparams()                               -> dict               (extra keys for save())

    `fit()` runs batched on the GPU when available; `estimate()` runs strictly
    sequentially on the CPU. Neither path looks at `dataset.states` inside
    `estimate()`.
    """

    estimator_id: str = "sequential_neural"

    def __init__(
        self,
        filter_model,
        learning_rate: float = 1e-3,
        num_epochs: int = 1,
        batch_size: int = 32,
        device: Optional[str] = None,
        random_seed: int = 0,
        grad_clip_norm: float = 0.5,
        weight_decay: float = 0.0,
        scheduler: str = "plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
        min_lr: float = 1e-6,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self._model = filter_model
        self._nx = filter_model.Q.shape[0]
        self._ny = filter_model.R.shape[0]
        self._lr = learning_rate
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._device_name = device
        self._random_seed = random_seed
        self._grad_clip_norm = grad_clip_norm
        self._weight_decay = weight_decay
        self._scheduler = scheduler
        self._scheduler_factor = scheduler_factor
        self._scheduler_patience = scheduler_patience
        self._min_lr = min_lr
        self._early_stopping_patience = early_stopping_patience
        self._early_stopping_min_delta = early_stopping_min_delta
        self._verbose = verbose
        self._network = None
        self._best_val_loss = float("inf")
        self._best_state_dict = None
        self.history_ = {"train_loss": [], "val_loss": [], "lr": [], "epoch": []}
        self.best_epoch_ = None
        self.stopped_epoch_ = None

    # --- interface a subclass must satisfy -------------------------------

    @property
    def estimator_name(self) -> str:
        return self.estimator_id

    @property
    def estimator_type(self) -> str:
        return "neural"

    @property
    def best_val_loss(self) -> float:
        return self._best_val_loss

    def _build_network(self):
        raise NotImplementedError

    def _forward_train(self, network, observations, states, timestamps, device):
        """Batched, parallel forward over the whole [B, T, *] sequence on GPU,
        returning x_hat [B, T, nx]. Used by the default `_loss`."""
        raise NotImplementedError

    def _loss(self, network, observations, states, timestamps, device):
        """Default training loss: MSE of the batched forward vs ground truth.
        Override (e.g. PINN) to add physics residual terms."""
        import torch.nn.functional as F
        pred = self._forward_train(network, observations, states, timestamps, device)
        return F.mse_loss(pred, states)

    def _estimate_sequential_cpu(self, network, observations, timestamps):
        """Strictly sequential CPU inference: one trajectory / one timestep at a
        time with the NumPy filter_model.f/h. Returns x_hat [N, T, nx] (np)."""
        raise NotImplementedError

    def _save_hyperparams(self) -> dict:
        return {}

    # --- shared machinery (mirrors kalmannet.py) -------------------------

    def _require_torch_dynamics(self) -> None:
        if self._model.torch is None:
            raise ValueError(
                f"{self.estimator_name}.fit() needs FilterModel.torch (batched "
                "torch dynamics) for vectorized GPU training; this model provides "
                "none. Add a TorchDynamics to the level (see _torch_dynamics.py)."
            )

    def _training_device(self):
        import torch
        if self._device_name is not None:
            return torch.device(self._device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_scheduler(self, optimizer, num_epochs=None):
        import torch
        if num_epochs is None:
            num_epochs = self._num_epochs
        name = (self._scheduler or "none").lower()
        if name in ("none", "off"):
            return None
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self._scheduler_factor,
                patience=self._scheduler_patience, min_lr=self._min_lr,
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(num_epochs, 1), eta_min=self._min_lr
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(self._scheduler_patience, 1),
                gamma=self._scheduler_factor,
            )
        raise ValueError(f"Unknown scheduler '{self._scheduler}'")

    def fit(self, train_dataset: "TrajectoryDataset", val_dataset: "TrajectoryDataset") -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self._random_seed)
        device = self._training_device()
        network = self._build_network().to(device)
        self._network = network

        train_obs = torch.as_tensor(np.asarray(train_dataset.observations), dtype=torch.float32)
        train_states = torch.as_tensor(np.asarray(train_dataset.states), dtype=torch.float32)
        val_obs = torch.as_tensor(np.asarray(val_dataset.observations), dtype=torch.float32)
        val_states = torch.as_tensor(np.asarray(val_dataset.states), dtype=torch.float32)
        train_ts = torch.as_tensor(np.asarray(train_dataset.timestamps), dtype=torch.float32)
        val_ts = torch.as_tensor(np.asarray(val_dataset.timestamps), dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_obs, train_states), batch_size=self._batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_obs, val_states), batch_size=self._batch_size, shuffle=False,
        )

        optimizer = torch.optim.Adam(
            network.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        scheduler = self._build_scheduler(optimizer, num_epochs=self._num_epochs)

        self._best_val_loss = float("inf")
        self._best_state_dict = None
        self.history_ = {"train_loss": [], "val_loss": [], "lr": [], "epoch": []}
        self.best_epoch_ = None
        self.stopped_epoch_ = None
        epochs_no_improve = 0

        for epoch in range(self._num_epochs):
            current_lr = optimizer.param_groups[0]["lr"]

            network.train()
            train_loss_total, train_batches = 0.0, 0
            for obs_b, states_b in train_loader:
                obs_b = obs_b.to(device)
                states_b = states_b.to(device)
                loss = self._loss(network, obs_b, states_b, train_ts, device)
                # Skip update if loss is NaN/Inf to prevent weight corruption.
                if not math.isfinite(loss.item()):
                    optimizer.zero_grad()
                    continue
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), self._grad_clip_norm)
                optimizer.step()
                train_loss_total += loss.item()
                train_batches += 1
            train_loss = train_loss_total / max(train_batches, 1)

            network.eval()
            val_loss_total, val_batches = 0.0, 0
            with torch.no_grad():
                for obs_b, states_b in val_loader:
                    obs_b = obs_b.to(device)
                    states_b = states_b.to(device)
                    val_loss_total += self._loss(network, obs_b, states_b, val_ts, device).item()
                    val_batches += 1
            val_loss = val_loss_total / max(val_batches, 1)

            self.history_["epoch"].append(epoch + 1)
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["lr"].append(current_lr)
            if self._verbose:
                print(
                    f"[{self.estimator_name}] epoch {epoch + 1}/{self._num_epochs} "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.2e}"
                )

            improved = False
            if math.isfinite(val_loss):
                if val_loss < self._best_val_loss - self._early_stopping_min_delta:
                    self._best_val_loss = val_loss
                    self._best_state_dict = {
                        k: v.cpu().clone() for k, v in network.state_dict().items()
                    }
                    self.best_epoch_ = epoch + 1
                    improved = True
                if scheduler is not None:
                    if self._scheduler == "plateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

            epochs_no_improve = 0 if improved else epochs_no_improve + 1
            if (
                self._early_stopping_patience is not None
                and epochs_no_improve >= self._early_stopping_patience
            ):
                self.stopped_epoch_ = epoch + 1
                if self._verbose:
                    print(
                        f"[{self.estimator_name}] early stopping at epoch {epoch + 1} "
                        f"(best val_loss={self._best_val_loss:.6f} @ epoch {self.best_epoch_})"
                    )
                break

        if self._best_state_dict is not None:
            network.load_state_dict(self._best_state_dict)
        self._network = network.to("cpu")

    def estimate(self, dataset: "TrajectoryDataset") -> np.ndarray:
        import torch
        if self._network is None:
            raise RuntimeError(f"{self.estimator_name} must be fit() before estimate().")
        # Enforce CPU-only inference (microprocessor deployment simulation).
        network = self._network.to("cpu")
        network.eval()
        observations = torch.as_tensor(
            np.asarray(dataset.observations), dtype=torch.float32, device=torch.device("cpu")
        )
        timestamps = np.asarray(dataset.timestamps)
        with torch.inference_mode():
            estimates = self._estimate_sequential_cpu(network, observations, timestamps)
        return np.asarray(estimates)

    def save(self, path: Path) -> None:
        import torch
        if self._network is None:
            raise RuntimeError("No trained network to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self._network.state_dict(),
            "nx": self._nx,
            "ny": self._ny,
            "estimator_name": self.estimator_name,
        }
        payload.update(self._save_hyperparams())
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "SequentialNeuralFilter":
        raise NotImplementedError(
            f"{cls.__name__}.load requires a FilterModel. "
            "Reconstruct the estimator from a BenchmarkLevel.get_filter_model() "
            "with the saved hyperparameters, then torch.load(path) and "
            "load_state_dict() on its network."
        )


def dt_array(timestamps: np.ndarray) -> np.ndarray:
    """Inter-sample intervals dt[t] = timestamps[t] - timestamps[t-1], with
    dt[0] set to dt[1] (no propagation happens before the first sample, but a
    finite seed value keeps any t=0 substep well-defined). Shared by the
    continuous-time filters."""
    ts = np.asarray(timestamps, dtype=np.float64)
    dt = np.empty_like(ts)
    if ts.shape[0] > 1:
        dt[1:] = np.diff(ts)
        dt[0] = dt[1]
    else:
        dt[0] = 1.0
    return dt
