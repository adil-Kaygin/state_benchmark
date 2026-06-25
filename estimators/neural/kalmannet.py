from __future__ import annotations

import json
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


def _process_model_step(f, x_batch: "torch.Tensor", t: float = 0.0) -> "torch.Tensor":
    """Apply a benchmark's (possibly nonlinear, NumPy-based) process model
    f(x, t) row-wise to a batch of state vectors on the CPU. KalmanNet's
    predict step intentionally reuses the benchmark's own dynamics (matching
    the LLD's FilterModel contract) rather than learning them from scratch."""
    import torch

    x_np = x_batch.detach().cpu().numpy()
    out = np.stack([f(x_np[i], t) for i in range(x_np.shape[0])], axis=0)
    return torch.as_tensor(out, dtype=x_batch.dtype, device=x_batch.device)


class KalmanNetEstimator(BaseEstimator):
    """
    KalmanNet-style estimator: recurrent (GRU) Kalman gain prediction driven
    by the innovation, composed with the benchmark's own process/observation
    model (via FilterModel). Adapted from `kalman_net_sim2real.py`'s
    architecture but generalized to arbitrary BenchmarkLevel state/obs
    dimensions instead of a fixed 6-dim IMU kinematic prior.

    Training runs on GPU when available; estimate() always runs on CPU
    (see `_cpu_inference_session`) so classical and neural estimators are
    benchmarked under identical conditions.
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
        self._network = None

    @property
    def estimator_name(self) -> str:
        return self.estimator_id

    @property
    def estimator_type(self) -> str:
        return "neural"

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

    def _run_sequence(self, network, observations, device, training: bool):
        """
        Run the predict/update recursion over a batch of full trajectories.

        observations: torch.Tensor [B, T, ny] on `device`.
        Returns (estimates [B, T, nx], log_vars [B, T, nx] or None).
        """
        import torch

        B, T, _ = observations.shape
        x = torch.zeros(B, self._nx, device=device, dtype=observations.dtype)
        x_pred_prev = x.clone()
        h = network.init_hidden(B, device)

        estimates, log_vars = [], []
        for t in range(T):
            x_pred = _process_model_step(self._model.f, x, float(t))
            y_pred = _process_model_step(self._model.h, x_pred)

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

        optimizer = torch.optim.Adam(network.parameters(), lr=self._lr)
        mse = nn.MSELoss()

        def _loss(pred, log_var, target):
            if self._predict_log_var and log_var is not None:
                var = torch.exp(log_var)
                return nn.functional.gaussian_nll_loss(pred, target, var, eps=1e-6)
            return mse(pred, target)

        for epoch in range(self._num_epochs):
            network.train()
            for obs_b, states_b in train_loader:
                obs_b = obs_b.to(device)
                states_b = states_b.to(device)

                pred, log_var = self._run_sequence(network, obs_b, device, training=True)
                loss = _loss(pred, log_var, states_b)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()

            network.eval()
            val_loss_total, val_batches = 0.0, 0
            with torch.no_grad():
                for obs_b, states_b in val_loader:
                    obs_b = obs_b.to(device)
                    states_b = states_b.to(device)
                    pred, log_var = self._run_sequence(network, obs_b, device, training=False)
                    val_loss_total += _loss(pred, log_var, states_b).item()
                    val_batches += 1
            val_loss = val_loss_total / max(val_batches, 1)
            print(f"[{self.estimator_name}] epoch {epoch + 1}/{self._num_epochs} val_loss={val_loss:.6f}")

        self._network = network.to("cpu")

    def estimate(self, dataset: "TrajectoryDataset") -> np.ndarray:
        import torch

        if self._network is None:
            raise RuntimeError(f"{self.estimator_name} must be fit() before estimate().")

        network = self._network.to("cpu")
        network.eval()

        observations = torch.as_tensor(np.asarray(dataset.observations), dtype=torch.float32)

        with torch.inference_mode():
            estimates, _ = self._run_sequence(network, observations, torch.device("cpu"), training=False)

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
        """Returns (estimates [N,T,nx], variance [N,T,nx])."""
        import torch

        if self._network is None:
            raise RuntimeError(f"{self.estimator_name} must be fit() before estimate().")

        network = self._network.to("cpu")
        network.eval()
        observations = torch.as_tensor(np.asarray(dataset.observations), dtype=torch.float32)

        with torch.inference_mode():
            estimates, log_var = self._run_sequence(network, observations, torch.device("cpu"), training=False)

        return estimates.numpy(), torch.exp(log_var).numpy()
