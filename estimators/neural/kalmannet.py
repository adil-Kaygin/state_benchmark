from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from ..base import BaseEstimator

if TYPE_CHECKING:
    import torch
    from datasets.schema import TrajectoryDataset


class _KalmanGainGRU:
    """
    Lazily-defined torch.nn.Module: a GRU-based recurrent gain estimator in
    the spirit of KalmanNet (Revach et al., 2022) and the reference
    `kalman_net_sim2real.py` script, but generalized over an arbitrary
    (nx, ny) state/observation dimension pair via the benchmark's
    `FilterModel.f` / `FilterModel.h` instead of a fixed IMU kinematic prior.

    Predict step (every timestep): x_pred = f(x_prev)  -- supplied by the
    benchmark's process model, run outside the network.

    Update step: the GRU consumes the innovation (y_t - h(x_pred)) and the
    previous state correction (x_prev - x_pred_prev), and emits a Kalman
    gain K (nx x ny). The correction x_post = x_pred + K @ innovation.

    If `predict_log_var=True`, an auxiliary head also predicts per-dimension
    log-variance of the state estimate (uncertainty-aware variant). This
    flag is the only difference between KalmanNetEstimator and
    KalmanNetUncertaintyEstimator -- the architecture/weights are otherwise
    identical, so there is no duplicated network code between the two.
    """

    @staticmethod
    def build(nx: int, ny: int, hidden_size: int, predict_log_var: bool):
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                in_features = ny + nx  # innovation + previous correction
                self.gru = nn.GRU(
                    input_size=in_features,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                )
                self.out_norm = nn.LayerNorm(hidden_size)
                self.fc_gain = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, nx * ny),
                )
                # Zero-init: an untrained network is a pure process-model
                # predictor (K=0), matching the reference script's behavior.
                nn.init.zeros_(self.fc_gain[-1].weight)
                nn.init.zeros_(self.fc_gain[-1].bias)

                self.predict_log_var = predict_log_var
                if predict_log_var:
                    self.fc_logvar = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, nx),
                    )
                    nn.init.zeros_(self.fc_logvar[-1].weight)
                    nn.init.zeros_(self.fc_logvar[-1].bias)

                self.nx = nx
                self.ny = ny
                self.hidden_size = hidden_size

            def step(self, innovation, dx_prev, h):
                """
                innovation: (B, ny), dx_prev: (B, nx), h: (1, B, hidden_size)
                Returns (K, log_var_or_None, h_next).
                """
                inp = torch.cat([innovation, dx_prev], dim=1).unsqueeze(1)
                out, h_next = self.gru(inp, h)
                out = self.out_norm(out.squeeze(1))

                K_flat = self.fc_gain(out)
                K = K_flat.view(-1, self.nx, self.ny)

                log_var = self.fc_logvar(out) if self.predict_log_var else None
                return K, log_var, h_next

            def init_hidden(self, batch_size: int, device) -> "torch.Tensor":
                return torch.zeros(1, batch_size, self.hidden_size, device=device)

        return _Module()


def _torch_batch_step(torch_fn, x_batch: "torch.Tensor", t: float = 0.0) -> "torch.Tensor":
    """Apply a benchmark's BATCHED torch process model (FilterModel.torch.f/h)
    to a batch of state vectors [B, nx] with a single vectorized tensor op on
    the input's own device. No per-row Python loop, no NumPy round-trip -- this
    is what lets KalmanNet's predict step run fully on the GPU during training.
    The batched torch dynamics mirror the NumPy/numba f/h one-for-one."""
    return torch_fn(x_batch, t)


class KalmanNetEstimator(BaseEstimator):
    """
    KalmanNet-style estimator: recurrent (GRU) Kalman gain prediction driven
    by the innovation, composed with the benchmark's own process/observation
    model (via FilterModel). Adapted from `kalman_net_sim2real.py`'s
    architecture but generalized to arbitrary BenchmarkLevel state/obs
    dimensions instead of a fixed 6-dim IMU kinematic prior.

    Hardware-specific execution (deliberate, per the deployment model):
    - fit()/validation: FULLY VECTORIZED, BATCHED torch on the GPU when
      available. The predict step uses FilterModel.torch (batched torch f/h),
      so every timestep is a single on-device tensor op -- no per-row Python
      loop, no NumPy round-trip (`_run_sequence_vectorized`).
    - estimate()/inference: STRICTLY SEQUENTIAL on the CPU -- one trajectory at
      a time, one timestep at a time, using the NumPy f/h on a single state
      vector (`_run_sequence_sequential_cpu`). This simulates microprocessor /
      embedded deployment and measures test-time latency under those conditions.
    """

    estimator_id = "kalmannet"
    _predict_log_var = False

    def __init__(
        self,
        filter_model,
        hidden_size: int = 64,
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
        self._hidden_size = hidden_size
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
        # Populated by fit(): per-epoch training diagnostics.
        self.history_ = {"train_loss": [], "val_loss": [], "lr": [], "epoch": []}
        self.best_epoch_ = None
        self.stopped_epoch_ = None

    @property
    def estimator_name(self) -> str:
        return self.estimator_id

    @property
    def estimator_type(self) -> str:
        return "neural"

    @property
    def best_val_loss(self) -> float:
        return self._best_val_loss

    def _build_scheduler(self, optimizer):
        import torch

        name = (self._scheduler or "none").lower()
        if name in ("none", "off"):
            return None
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self._scheduler_factor,
                patience=self._scheduler_patience,
                min_lr=self._min_lr,
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(self._num_epochs, 1), eta_min=self._min_lr
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(self._scheduler_patience, 1),
                gamma=self._scheduler_factor,
            )
        raise ValueError(f"Unknown scheduler '{self._scheduler}'")

    def _training_device(self):
        import torch
        if self._device_name is not None:
            return torch.device(self._device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_network(self):
        if self._network is None:
            self._network = _KalmanGainGRU.build(
                self._nx, self._ny, self._hidden_size, self._predict_log_var
            )
        return self._network

    def _run_sequence_vectorized(self, network, observations, timestamps, device):
        """
        VECTORIZED predict/update recursion over a batch of full trajectories,
        used for GPU training/validation. Every timestep is a single batched
        torch op on `device` (the batched FilterModel.torch.f/h plus the GRU);
        there is no per-row Python loop and no NumPy round-trip, so the whole
        forward pass stays on the GPU. The only loop is over the T timesteps,
        which is the GRU's intrinsic recurrence (each step depends on the
        previous corrected state) and cannot be removed for a sequential filter.

        observations: torch.Tensor [B, T, ny] on `device`.
        timestamps:   torch.Tensor [T] (used for time-varying f, e.g. nonlinear).
        Returns (estimates [B, T, nx], log_vars [B, T, nx] or None).
        """
        import torch

        if self._model.torch is None:
            raise ValueError(
                f"{self.estimator_name}.fit() needs FilterModel.torch (batched "
                "torch dynamics) for vectorized GPU training; this model provides "
                "none. Add a TorchDynamics to the level (see _torch_dynamics.py)."
            )
        torch_f = self._model.torch.f
        torch_h = self._model.torch.h

        B, T, _ = observations.shape
        x = torch.zeros(B, self._nx, device=device, dtype=observations.dtype)
        x_pred_prev = x.clone()
        h = network.init_hidden(B, device)

        estimates, log_vars = [], []
        for t in range(T):
            t_val = float(timestamps[t])
            x_pred = _torch_batch_step(torch_f, x, t_val)
            y_pred = _torch_batch_step(torch_h, x_pred, t_val)

            innovation = observations[:, t, :] - y_pred
            dx_prev = x - x_pred_prev

            K, log_var, h = network.step(innovation, dx_prev, h)
            correction = torch.bmm(K, innovation.unsqueeze(2)).squeeze(2)
            x_post = x_pred + correction

            x_pred_prev = x_pred
            x = x_post

            estimates.append(x_post)
            if log_var is not None:
                log_vars.append(log_var)

        estimates_seq = torch.stack(estimates, dim=1)
        log_var_seq = torch.stack(log_vars, dim=1) if log_vars else None
        return estimates_seq, log_var_seq

    def fit(self, train_dataset: "TrajectoryDataset", val_dataset: "TrajectoryDataset") -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self._random_seed)
        device = self._training_device()
        network = self._ensure_network().to(device)

        train_obs = torch.as_tensor(np.asarray(train_dataset.observations), dtype=torch.float32)
        train_states = torch.as_tensor(np.asarray(train_dataset.states), dtype=torch.float32)
        val_obs = torch.as_tensor(np.asarray(val_dataset.observations), dtype=torch.float32)
        val_states = torch.as_tensor(np.asarray(val_dataset.states), dtype=torch.float32)

        # Timestamps are shared across all trajectories in a split; thread them
        # into the (possibly time-varying) batched torch f during training.
        train_ts = torch.as_tensor(np.asarray(train_dataset.timestamps), dtype=torch.float32)
        val_ts = torch.as_tensor(np.asarray(val_dataset.timestamps), dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_obs, train_states),
            batch_size=self._batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_obs, val_states),
            batch_size=self._batch_size,
            shuffle=False,
        )

        optimizer = torch.optim.Adam(
            network.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        scheduler = self._build_scheduler(optimizer)
        mse = nn.MSELoss()

        def _loss(pred, log_var, target):
            if self._predict_log_var and log_var is not None:
                var = torch.exp(log_var)
                return nn.functional.gaussian_nll_loss(pred, target, var, eps=1e-6)
            return mse(pred, target)

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

                pred, log_var = self._run_sequence_vectorized(network, obs_b, train_ts, device)
                loss = _loss(pred, log_var, states_b)

                # Skip update if loss is NaN or Inf to prevent weight corruption
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
                    pred, log_var = self._run_sequence_vectorized(network, obs_b, val_ts, device)
                    val_loss_total += _loss(pred, log_var, states_b).item()
                    val_batches += 1
            val_loss = val_loss_total / max(val_batches, 1)

            self.history_["epoch"].append(epoch + 1)
            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)
            self.history_["lr"].append(current_lr)

            improved = False
            # Track best checkpoint in memory (only when loss is finite)
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

            if self._verbose:
                print(
                    f"[{self.estimator_name}] epoch {epoch + 1}/{self._num_epochs} "
                    f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.2e}"
                )

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

        # Load best checkpoint if available; otherwise keep the last state
        if self._best_state_dict is not None:
            network.load_state_dict(self._best_state_dict)

        self._network = network.to("cpu")

    def _run_sequence_sequential_cpu(self, network, observations, timestamps):
        """
        STRICTLY SEQUENTIAL, CPU-ONLY inference recursion used by estimate().

        This deliberately simulates microprocessor / embedded deployment: the
        filter runs one trajectory at a time, one timestep at a time, on the
        CPU, using the benchmark's NumPy process model f/h (one state vector at
        a time, no batching). It is the opposite of the vectorized GPU training
        path on purpose -- test-time latency is measured under deployment-like
        conditions, not under GPU batch throughput.

        observations: torch.Tensor [N, T, ny] (on CPU).
        Returns (estimates [N, T, nx], log_vars [N, T, nx] or None) as CPU tensors.
        """
        import torch

        cpu = torch.device("cpu")
        N, T, _ = observations.shape
        f = self._model.f
        h = self._model.h

        all_estimates = torch.zeros(N, T, self._nx, dtype=observations.dtype, device=cpu)
        all_log_vars = (
            torch.zeros(N, T, self._nx, dtype=observations.dtype, device=cpu)
            if self._predict_log_var else None
        )

        for i in range(N):  # one trajectory at a time
            x = torch.zeros(1, self._nx, dtype=observations.dtype, device=cpu)
            x_pred_prev = x.clone()
            hidden = network.init_hidden(1, cpu)

            for t in range(T):  # one timestep at a time
                t_val = float(timestamps[t])
                # NumPy process model, single state vector (microprocessor-style).
                x_np = x.squeeze(0).numpy()
                x_pred_np = np.asarray(f(x_np, t_val), dtype=np.float64)
                y_pred_np = np.asarray(h(x_pred_np, t_val), dtype=np.float64)
                x_pred = torch.from_numpy(x_pred_np).to(observations.dtype).unsqueeze(0)
                y_pred = torch.from_numpy(y_pred_np).to(observations.dtype).unsqueeze(0)

                innovation = observations[i, t, :].unsqueeze(0) - y_pred
                dx_prev = x - x_pred_prev

                K, log_var, hidden = network.step(innovation, dx_prev, hidden)
                correction = torch.bmm(K, innovation.unsqueeze(2)).squeeze(2)
                x_post = x_pred + correction

                x_pred_prev = x_pred
                x = x_post

                all_estimates[i, t] = x_post.squeeze(0)
                if all_log_vars is not None and log_var is not None:
                    all_log_vars[i, t] = log_var.squeeze(0)

        return all_estimates, all_log_vars

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
            estimates, _ = self._run_sequence_sequential_cpu(network, observations, timestamps)

        return estimates.numpy()

    def save(self, path: Path) -> None:
        import torch
        if self._network is None:
            raise RuntimeError("No trained network to save.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._network.state_dict(),
                "nx": self._nx,
                "ny": self._ny,
                "hidden_size": self._hidden_size,
                "predict_log_var": self._predict_log_var,
                "estimator_name": self.estimator_name,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "KalmanNetEstimator":
        raise NotImplementedError(
            f"{cls.__name__}.load requires a FilterModel. "
            "Reconstruct the estimator from a BenchmarkLevel.get_filter_model(), "
            "then call torch.load(path) and load_state_dict() on its network."
        )


class KalmanNetUncertaintyEstimator(KalmanNetEstimator):
    """
    Uncertainty-aware KalmanNet variant (a.k.a. KalmanNetPlus): identical
    architecture to KalmanNetEstimator plus an auxiliary log-variance head,
    trained with a Gaussian NLL loss instead of plain MSE. No network code
    is duplicated -- this subclass only flips `_predict_log_var` and
    `estimator_id`.

    Not executed by the default benchmark notebook; available for opt-in use.
    """

    estimator_id = "kalmannet_uncertainty"
    _predict_log_var = True

    def estimate_with_uncertainty(self, dataset: "TrajectoryDataset"):
        """Returns (estimates [N,T,nx], variance [N,T,nx]).

        Like estimate(), runs strictly sequentially on the CPU (deployment-like).
        """
        import torch

        if self._network is None:
            raise RuntimeError(f"{self.estimator_name} must be fit() before estimate().")

        network = self._network.to("cpu")
        network.eval()
        observations = torch.as_tensor(
            np.asarray(dataset.observations), dtype=torch.float32, device=torch.device("cpu")
        )
        timestamps = np.asarray(dataset.timestamps)

        with torch.inference_mode():
            estimates, log_var = self._run_sequence_sequential_cpu(network, observations, timestamps)

        return estimates.numpy(), torch.exp(log_var).numpy()
