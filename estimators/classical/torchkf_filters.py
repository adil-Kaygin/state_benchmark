from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import BaseEstimator
from benchmark_levels.base import FilterModel
from datasets.schema import TrajectoryDataset

# Reference classical filters built on two third-party PyTorch libraries, used as
# independent cross-checks against this repo's custom Numba filters:
#   * torch-kf   (imported as `torch_kf`)  -- a natively-batched *linear* Kalman
#     filter. torch-kf is linear-only (no EKF/UKF of its own), so it backs only
#     the KF reference here.
#   * torchfilter                          -- differentiable Bayesian filters in
#     PyTorch (EKF, UKF, particle filter). It backs the nonlinear references.
# Both are imported lazily (only when a class that needs them is instantiated) so
# the rest of the package works without either installed.


def _require_torchkf():
    try:
        import torch_kf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torch-kf is required for this estimator. Install it with "
            "`pip install torch-kf` (already listed in setup.py install_requires). "
            "The package is imported as `torch_kf`."
        ) from exc


def _require_torchfilter():
    try:
        import torchfilter  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "torchfilter is required for this estimator. Install it with "
            "`pip install torchfilter` (already listed in setup.py install_requires). "
            "It supplies the EKF/UKF/PF reference filters (torch-kf is linear-only)."
        ) from exc


def _angular_indices(filter_model, ny: int) -> np.ndarray:
    """Integer indices of the angular (bearing) observation components from
    FilterModel.angular_obs_mask, or an empty array when there are none (Issues
    5/6). Used to wrap the residual y - h(x) on the EKF/UKF/PF paths so they match
    the custom kernels' angle handling."""
    mask = getattr(filter_model, "angular_obs_mask", None)
    if mask is None:
        return np.empty(0, dtype=np.int64)
    mask = np.asarray(mask)
    if mask.shape != (ny,):
        raise ValueError(
            f"angular_obs_mask must have shape ({ny},); got {mask.shape}."
        )
    return np.nonzero(mask)[0]


def _cholesky_tril(cov: np.ndarray):
    """Lower-triangular Cholesky factor of a covariance (the `scale_tril` that
    torchfilter's models must return). Kept as a float64 numpy array."""
    return np.linalg.cholesky(np.asarray(cov, dtype=np.float64))


# --------------------------------------------------------------------------- #
# torchfilter model wrappers around this repo's FilterModel (f/h/F/H, Q, R).    #
# --------------------------------------------------------------------------- #


def _build_torchfilter_models(filter_model: FilterModel, nx: int, ny: int):
    """Construct torchfilter DynamicsModel / KalmanFilterMeasurementModel classes
    bound to `filter_model`. The dynamics model carries a mutable `t` so the
    per-step timestamp can be threaded into f/F exactly like the custom EKF/UKF.
    Jacobians are supplied explicitly from the model's F/H (the true analytic
    Jacobians), overriding torchfilter's autograd default -- our f/h are NumPy and
    not differentiable by torch."""
    import torch
    import torchfilter

    Q_tril = torch.as_tensor(_cholesky_tril(filter_model.Q), dtype=torch.float32)
    R_tril = torch.as_tensor(_cholesky_tril(filter_model.R), dtype=torch.float32)

    class _Dynamics(torchfilter.base.DynamicsModel):
        def __init__(self):
            super().__init__(state_dim=nx)
            self.t = 0.0

        def forward(self, *, initial_states, controls):
            x = initial_states.detach().cpu().numpy().astype(np.float64)
            fx = np.stack([filter_model.f(xi, self.t) for xi in x])
            pred = torch.as_tensor(fx, dtype=initial_states.dtype, device=initial_states.device)
            N = initial_states.shape[0]
            tril = Q_tril.to(initial_states.device)[None].expand(N, nx, nx)
            return pred, tril

        def jacobian(self, *, initial_states, controls):
            x = initial_states.detach().cpu().numpy().astype(np.float64)
            Fx = np.stack([filter_model.F(xi) for xi in x])
            return torch.as_tensor(Fx, dtype=initial_states.dtype, device=initial_states.device)

    class _Measurement(torchfilter.base.KalmanFilterMeasurementModel):
        def __init__(self):
            super().__init__(state_dim=nx, observation_dim=ny)

        def forward(self, *, states):
            x = states.detach().cpu().numpy().astype(np.float64)
            hx = np.stack([filter_model.h(xi) for xi in x])
            pred = torch.as_tensor(hx, dtype=states.dtype, device=states.device)
            N = states.shape[0]
            tril = R_tril.to(states.device)[None].expand(N, ny, ny)
            return pred, tril

        def jacobian(self, *, states):
            x = states.detach().cpu().numpy().astype(np.float64)
            Hx = np.stack([filter_model.H(xi) for xi in x])
            return torch.as_tensor(Hx, dtype=states.dtype, device=states.device)

    return _Dynamics(), _Measurement()


def _run_torchfilter(
    filt, dynamics, dataset: TrajectoryDataset, filter_model: FilterModel,
    nx: int, with_cov: bool,
):
    """Drive a torchfilter Kalman-style filter (EKF/UKF) one timestep at a time so
    we can (a) thread the per-step timestamp into the dynamics model and (b) wrap
    the bearing innovation to (-pi, pi] (Issues 5/6). torchfilter computes the
    innovation as `obs - h(x_pred)` with no residual hook, so for angular models we
    substitute `obs' = h(x_pred) + wrap(obs - h(x_pred))`, which yields the wrapped
    innovation while leaving torchfilter's own EKF/UKF math untouched."""
    import torch

    observations = np.asarray(dataset.observations, dtype=np.float64)
    timestamps = np.asarray(dataset.timestamps, dtype=np.float64)
    N, T, ny = observations.shape
    angular_idx = _angular_indices(filter_model, ny)

    x0_mean = filter_model.x0_mean if filter_model.x0_mean is not None else np.zeros(nx)
    x0_cov = filter_model.x0_cov if filter_model.x0_cov is not None else np.eye(nx)

    mean0 = torch.as_tensor(np.broadcast_to(x0_mean, (N, nx)).copy(), dtype=torch.float32)
    cov0 = torch.as_tensor(np.broadcast_to(x0_cov, (N, nx, nx)).copy(), dtype=torch.float32)
    filt.initialize_beliefs(mean=mean0, covariance=cov0)

    controls = torch.zeros(N, 1, dtype=torch.float32)  # unused; models ignore controls
    estimates = np.zeros((N, T, nx))
    covs = np.zeros((N, T, nx, nx)) if with_cov else None
    for step in range(T):
        dynamics.t = float(timestamps[step])
        obs_t = torch.as_tensor(observations[:, step], dtype=torch.float32)
        if angular_idx.size:
            filt._predict_step(controls=controls)
            hx, _ = filt.measurement_model(states=filt.belief_mean)
            r = obs_t - hx
            for j in angular_idx:
                r[:, j] = torch.atan2(torch.sin(r[:, j]), torch.cos(r[:, j]))
            filt._update_step(observations=hx + r)
        else:
            filt(observations=obs_t, controls=controls)
        estimates[:, step] = filt.belief_mean.detach().cpu().numpy()
        if with_cov:
            covs[:, step] = filt.belief_covariance.detach().cpu().numpy()

    return (estimates, covs) if with_cov else estimates


# --------------------------------------------------------------------------- #
# KF -- torch-kf (linear, natively batched).                                   #
# --------------------------------------------------------------------------- #


class TorchKFKFEstimator(BaseEstimator):
    """
    Reference KalmanFilterEstimator built on torch_kf.KalmanFilter instead of
    this repo's custom NumPy/Numba KF (estimators/classical/kf.py). Same
    linearize-at-origin contract: only statistically correct on LinearBenchmark.
    Exists as an independent cross-check, not a replacement. Runs every trajectory
    in one batched torch_kf pass (torch-kf is natively batched).
    """

    returns_covariance = True  # Issue 7: torch-kf maintains P internally.

    def __init__(self, filter_model: FilterModel) -> None:
        _require_torchkf()
        self._model = filter_model

    @property
    def estimator_name(self) -> str:
        return "torchkf_kf"

    @property
    def estimator_type(self) -> str:
        return "classical"

    def fit(
        self,
        train_dataset: Optional[TrajectoryDataset],
        val_dataset: Optional[TrajectoryDataset],
    ) -> None:
        pass  # KF requires no training.

    def _run(self, dataset: TrajectoryDataset, with_cov: bool):
        import torch
        from torch_kf import GaussianState, KalmanFilter

        observations = np.asarray(dataset.observations, dtype=np.float64)
        N, T, ny = observations.shape
        nx = self._model.Q.shape[0]

        F = np.asarray(self._model.F(np.zeros(nx)), dtype=np.float64)
        H = np.asarray(self._model.H(np.zeros(nx)), dtype=np.float64)
        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)

        t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)  # noqa: E731
        kf = KalmanFilter(t(F), t(H), t(self._model.Q), t(self._model.R))

        # Batched initial state: (N, nx, 1) mean, (N, nx, nx) covariance.
        state = GaussianState(
            mean=t(x0_mean).reshape(1, nx, 1).expand(N, nx, 1).clone(),
            covariance=t(x0_cov).reshape(1, nx, nx).expand(N, nx, nx).clone(),
        )
        # torch-kf measurements are column vectors (..., ny, 1); time-major so we
        # can predict/update the whole batch at each step.
        meas = t(observations).permute(1, 0, 2).reshape(T, N, ny, 1)

        estimates = np.zeros((N, T, nx))
        covs = np.zeros((N, T, nx, nx)) if with_cov else None
        for step in range(T):
            state = kf.predict(state)
            state = kf.update(state, meas[step])
            estimates[:, step] = state.mean[..., 0].cpu().numpy()
            if with_cov:
                covs[:, step] = state.covariance.cpu().numpy()

        return (estimates, covs) if with_cov else estimates

    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        return self._run(dataset, with_cov=False)

    def estimate_with_covariance(self, dataset: TrajectoryDataset):
        """(estimates, covariances) from torch-kf's maintained P (Issue 7)."""
        return self._run(dataset, with_cov=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"estimator_name": self.estimator_name, "estimator_type": self.estimator_type}, f)

    @classmethod
    def load(cls, path: Path) -> "TorchKFKFEstimator":
        raise NotImplementedError(
            "TorchKFKFEstimator.load requires a FilterModel. "
            "Reconstruct from a BenchmarkLevel.get_filter_model()."
        )


# --------------------------------------------------------------------------- #
# EKF -- torchfilter.filters.ExtendedKalmanFilter.                             #
# --------------------------------------------------------------------------- #


class TorchKFEKFEstimator(BaseEstimator):
    """
    Reference EKFEstimator built on torchfilter.filters.ExtendedKalmanFilter
    instead of this repo's custom EKF (estimators/classical/ekf.py). The dynamics
    and measurement models wrap the level's f/h and return the analytic Jacobians
    F/H directly. Threads the dataset's timestamp into f/F/h/H like the custom
    EKF, so it is valid on every (possibly time-varying) nonlinear level.
    """

    returns_covariance = True  # Issue 7: torchfilter maintains a posterior P.

    def __init__(self, filter_model: FilterModel) -> None:
        _require_torchfilter()
        self._model = filter_model

    @property
    def estimator_name(self) -> str:
        return "torchkf_ekf"

    @property
    def estimator_type(self) -> str:
        return "classical"

    def fit(
        self,
        train_dataset: Optional[TrajectoryDataset],
        val_dataset: Optional[TrajectoryDataset],
    ) -> None:
        pass  # EKF requires no training.

    def _run(self, dataset: TrajectoryDataset, with_cov: bool):
        import torchfilter

        observations = np.asarray(dataset.observations)
        ny = observations.shape[-1]
        nx = self._model.Q.shape[0]
        dynamics, measurement = _build_torchfilter_models(self._model, nx, ny)
        filt = torchfilter.filters.ExtendedKalmanFilter(
            dynamics_model=dynamics, measurement_model=measurement
        )
        return _run_torchfilter(filt, dynamics, dataset, self._model, nx, with_cov)

    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        return self._run(dataset, with_cov=False)

    def estimate_with_covariance(self, dataset: TrajectoryDataset):
        """(estimates, covariances) from the EKF's maintained P (Issue 7)."""
        return self._run(dataset, with_cov=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"estimator_name": self.estimator_name, "estimator_type": self.estimator_type}, f)

    @classmethod
    def load(cls, path: Path) -> "TorchKFEKFEstimator":
        raise NotImplementedError(
            "TorchKFEKFEstimator.load requires a FilterModel. "
            "Reconstruct from a BenchmarkLevel.get_filter_model()."
        )


# --------------------------------------------------------------------------- #
# UKF -- torchfilter.filters.UnscentedKalmanFilter.                            #
# --------------------------------------------------------------------------- #


class TorchKFUKFEstimator(BaseEstimator):
    """
    Reference UKFEstimator built on torchfilter.filters.UnscentedKalmanFilter
    instead of this repo's custom UKF (estimators/classical/ukf.py). Uses
    torchfilter's MerweSigmaPointStrategy with the same alpha/beta/kappa
    convention as the custom UKF.
    """

    returns_covariance = True  # Issue 7: the UKF propagates a real posterior P.

    def __init__(
        self,
        filter_model: FilterModel,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        _require_torchfilter()
        self._model = filter_model
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

    @property
    def estimator_name(self) -> str:
        return "torchkf_ukf"

    @property
    def estimator_type(self) -> str:
        return "classical"

    def fit(
        self,
        train_dataset: Optional[TrajectoryDataset],
        val_dataset: Optional[TrajectoryDataset],
    ) -> None:
        pass  # UKF requires no training.

    def _run(self, dataset: TrajectoryDataset, with_cov: bool):
        import torchfilter

        observations = np.asarray(dataset.observations)
        ny = observations.shape[-1]
        nx = self._model.Q.shape[0]
        dynamics, measurement = _build_torchfilter_models(self._model, nx, ny)
        sigma = torchfilter.utils.MerweSigmaPointStrategy(
            alpha=self._alpha, beta=self._beta, kappa=self._kappa
        )
        filt = torchfilter.filters.UnscentedKalmanFilter(
            dynamics_model=dynamics,
            measurement_model=measurement,
            sigma_point_strategy=sigma,
        )
        return _run_torchfilter(filt, dynamics, dataset, self._model, nx, with_cov)

    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        return self._run(dataset, with_cov=False)

    def estimate_with_covariance(self, dataset: TrajectoryDataset):
        """(estimates, covariances) from the UKF's propagated posterior P (Issue 7)."""
        return self._run(dataset, with_cov=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "estimator_name": self.estimator_name,
                    "estimator_type": self.estimator_type,
                    "alpha": self._alpha,
                    "beta": self._beta,
                    "kappa": self._kappa,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "TorchKFUKFEstimator":
        raise NotImplementedError(
            "TorchKFUKFEstimator.load requires a FilterModel. "
            "Reconstruct from a BenchmarkLevel.get_filter_model()."
        )


# --------------------------------------------------------------------------- #
# PF -- torchfilter.filters.ParticleFilter (added for future use).            #
# --------------------------------------------------------------------------- #


class TorchKFPFEstimator(BaseEstimator):
    """
    Reference particle filter built on torchfilter.filters.ParticleFilter, added
    for future use alongside this repo's custom PF (estimators/classical/pf.py).
    Reuses the same f/h wrappers as the EKF/UKF; the Gaussian measurement model is
    lifted to a likelihood via torchfilter's ParticleFilterMeasurementModelWrapper.

    Not registered in the default sweep. A particle filter carries a weighted
    particle cloud rather than a single Gaussian posterior, so it exposes point
    estimates only (returns_covariance stays False); wire up a
    particle-covariance estimate before opting it into the NEES/NLL metrics.
    """

    def __init__(
        self,
        filter_model: FilterModel,
        num_particles: int = 100,
        seed: int = 0,
    ) -> None:
        _require_torchfilter()
        self._model = filter_model
        self._num_particles = num_particles
        self._seed = seed

    @property
    def estimator_name(self) -> str:
        return "torchkf_pf"

    @property
    def estimator_type(self) -> str:
        return "classical"

    def fit(
        self,
        train_dataset: Optional[TrajectoryDataset],
        val_dataset: Optional[TrajectoryDataset],
    ) -> None:
        pass  # PF requires no training.

    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        import torch
        import torchfilter

        torch.manual_seed(self._seed)  # particle sampling/resampling is stochastic
        observations = np.asarray(dataset.observations, dtype=np.float64)
        timestamps = np.asarray(dataset.timestamps, dtype=np.float64)
        N, T, ny = observations.shape
        nx = self._model.Q.shape[0]
        if _angular_indices(self._model, ny).size:
            raise NotImplementedError(
                "TorchKFPFEstimator does not yet wrap bearing residuals; its "
                "Gaussian likelihood would mis-score angular observations near the "
                "+/-pi branch cut. Add angle handling before using it on levels "
                "with angular_obs_mask set."
            )

        dynamics, measurement = _build_torchfilter_models(self._model, nx, ny)
        filt = torchfilter.filters.ParticleFilter(
            dynamics_model=dynamics,
            measurement_model=torchfilter.base.ParticleFilterMeasurementModelWrapper(
                measurement
            ),
            num_particles=self._num_particles,
        )
        filt.eval()  # enable resampling (torchfilter resamples in eval mode)

        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)
        filt.initialize_beliefs(
            mean=torch.as_tensor(np.broadcast_to(x0_mean, (N, nx)).copy(), dtype=torch.float32),
            covariance=torch.as_tensor(np.broadcast_to(x0_cov, (N, nx, nx)).copy(), dtype=torch.float32),
        )

        controls = torch.zeros(N, 1, dtype=torch.float32)
        estimates = np.zeros((N, T, nx))
        for step in range(T):
            dynamics.t = float(timestamps[step])
            obs_t = torch.as_tensor(observations[:, step], dtype=torch.float32)
            estimates[:, step] = filt(observations=obs_t, controls=controls).detach().cpu().numpy()
        return estimates

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "estimator_name": self.estimator_name,
                    "estimator_type": self.estimator_type,
                    "num_particles": self._num_particles,
                    "seed": self._seed,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "TorchKFPFEstimator":
        raise NotImplementedError(
            "TorchKFPFEstimator.load requires a FilterModel. "
            "Reconstruct from a BenchmarkLevel.get_filter_model()."
        )
