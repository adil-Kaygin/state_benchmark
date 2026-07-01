from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import BaseEstimator
from benchmark_levels.base import FilterModel
from datasets.schema import TrajectoryDataset


def _require_filterpy():
    try:
        import filterpy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "filterpy is required for this estimator. Install it with "
            "`pip install filterpy` (already listed in setup.py install_requires)."
        ) from exc


def _angular_indices(filter_model, ny: int) -> np.ndarray:
    """Integer indices of the angular (bearing) observation components from
    FilterModel.angular_obs_mask, or an empty array when there are none (Issues
    5/6). Used to wrap the residual y - h(x) on the filterpy EKF/UKF paths so they
    match the custom kernels' angle handling."""
    mask = getattr(filter_model, "angular_obs_mask", None)
    if mask is None:
        return np.empty(0, dtype=np.int64)
    mask = np.asarray(mask)
    if mask.shape != (ny,):
        raise ValueError(
            f"angular_obs_mask must have shape ({ny},); got {mask.shape}."
        )
    return np.nonzero(mask)[0]


def _wrap_residual(z, hx, angular_idx):
    """Measurement residual z - hx with the angular components wrapped to
    (-pi, pi]. filterpy's EKF/UKF accept a `residual_z`/`residual` callable of
    exactly this signature."""
    r = np.asarray(z, dtype=np.float64) - np.asarray(hx, dtype=np.float64)
    for j in angular_idx:
        r[j] = np.arctan2(np.sin(r[j]), np.cos(r[j]))
    return r


class FilterpyKFEstimator(BaseEstimator):
    """
    Reference KalmanFilterEstimator built on filterpy.kalman.KalmanFilter
    instead of this repo's custom NumPy/Numba KF (estimators/classical/kf.py).
    Same linearize-at-origin contract: only statistically correct on
    LinearBenchmark. Exists as an independent cross-check, not a replacement.
    """

    returns_covariance = True  # Issue 7: filterpy maintains P internally.

    def __init__(self, filter_model: FilterModel) -> None:
        _require_filterpy()
        self._model = filter_model

    @property
    def estimator_name(self) -> str:
        return "filterpy_kf"

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
        from filterpy.kalman import KalmanFilter

        observations = np.asarray(dataset.observations)
        N, T, ny = observations.shape
        nx = self._model.Q.shape[0]

        F = self._model.F(np.zeros(nx))
        H = self._model.H(np.zeros(nx))
        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)

        kf = KalmanFilter(dim_x=nx, dim_z=ny)
        kf.F = F
        kf.H = H
        kf.Q = self._model.Q
        kf.R = self._model.R

        estimates = np.zeros((N, T, nx))
        covs = np.zeros((N, T, nx, nx)) if with_cov else None
        for i in range(N):
            kf.x = x0_mean.copy()
            kf.P = x0_cov.copy()
            for t in range(T):
                kf.predict()
                kf.update(observations[i, t])
                estimates[i, t] = kf.x
                if with_cov:
                    covs[i, t] = kf.P

        return (estimates, covs) if with_cov else estimates

    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        return self._run(dataset, with_cov=False)

    def estimate_with_covariance(self, dataset: TrajectoryDataset):
        """(estimates, covariances) from filterpy's maintained P (Issue 7)."""
        return self._run(dataset, with_cov=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"estimator_name": self.estimator_name, "estimator_type": self.estimator_type}, f)

    @classmethod
    def load(cls, path: Path) -> "FilterpyKFEstimator":
        raise NotImplementedError(
            "FilterpyKFEstimator.load requires a FilterModel. "
            "Reconstruct from a BenchmarkLevel.get_filter_model()."
        )


class FilterpyEKFEstimator(BaseEstimator):
    """
    Reference EKFEstimator built on filterpy.kalman.ExtendedKalmanFilter
    instead of this repo's custom EKF (estimators/classical/ekf.py). Threads
    the dataset's timestamp into f/F/h/H like the custom EKF, so it is valid
    on every (possibly time-varying) nonlinear level.
    """

    returns_covariance = True  # Issue 7: filterpy maintains P internally.

    def __init__(self, filter_model: FilterModel) -> None:
        _require_filterpy()
        self._model = filter_model

    @property
    def estimator_name(self) -> str:
        return "filterpy_ekf"

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
        from filterpy.kalman import ExtendedKalmanFilter

        observations = np.asarray(dataset.observations)
        timestamps = np.asarray(dataset.timestamps)
        N, T, ny = observations.shape
        nx = self._model.Q.shape[0]
        Q, R = self._model.Q, self._model.R

        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)

        ekf = ExtendedKalmanFilter(dim_x=nx, dim_z=ny)
        ekf.Q = Q
        ekf.R = R
        angular_idx = _angular_indices(self._model, ny)
        residual = (lambda a, b: _wrap_residual(a, b, angular_idx)) if angular_idx.size else np.subtract

        estimates = np.zeros((N, T, nx))
        covs = np.zeros((N, T, nx, nx)) if with_cov else None
        for i in range(N):
            ekf.x = x0_mean.copy()
            ekf.P = x0_cov.copy()
            for t in range(T):
                t_val = float(timestamps[t])
                ekf.F = self._model.F(ekf.x)
                ekf.x = self._model.f(ekf.x, t_val)
                ekf.P = ekf.F @ ekf.P @ ekf.F.T + Q

                Hj = self._model.H(ekf.x)
                ekf.update(
                    observations[i, t],
                    HJacobian=lambda x, Hj=Hj: Hj,
                    Hx=lambda x: self._model.h(x),
                    residual=residual,
                )
                estimates[i, t] = ekf.x
                if with_cov:
                    covs[i, t] = ekf.P

        return (estimates, covs) if with_cov else estimates

    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        return self._run(dataset, with_cov=False)

    def estimate_with_covariance(self, dataset: TrajectoryDataset):
        """(estimates, covariances) from filterpy's maintained P (Issue 7)."""
        return self._run(dataset, with_cov=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"estimator_name": self.estimator_name, "estimator_type": self.estimator_type}, f)

    @classmethod
    def load(cls, path: Path) -> "FilterpyEKFEstimator":
        raise NotImplementedError(
            "FilterpyEKFEstimator.load requires a FilterModel. "
            "Reconstruct from a BenchmarkLevel.get_filter_model()."
        )


class FilterpyUKFEstimator(BaseEstimator):
    """
    Reference UKFEstimator built on filterpy.kalman.UnscentedKalmanFilter
    instead of this repo's custom UKF (estimators/classical/ukf.py). Uses
    filterpy's MerweScaledSigmaPoints with the same alpha/beta/kappa
    convention as the custom UKF.
    """

    def __init__(
        self,
        filter_model: FilterModel,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        _require_filterpy()
        self._model = filter_model
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

    returns_covariance = True  # Issue 7: filterpy maintains P internally.

    @property
    def estimator_name(self) -> str:
        return "filterpy_ukf"

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
        from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

        observations = np.asarray(dataset.observations)
        timestamps = np.asarray(dataset.timestamps)
        N, T, ny = observations.shape
        nx = self._model.Q.shape[0]
        Q, R = self._model.Q, self._model.R

        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)

        points = MerweScaledSigmaPoints(n=nx, alpha=self._alpha, beta=self._beta, kappa=self._kappa)
        angular_idx = _angular_indices(self._model, ny)

        def _residual_z(a, b):
            return _wrap_residual(a, b, angular_idx)

        estimates = np.zeros((N, T, nx))
        covs = np.zeros((N, T, nx, nx)) if with_cov else None
        for i in range(N):
            t_box = {"t": 0.0}

            def fx(x, dt, t_box=t_box):
                return self._model.f(x, t_box["t"])

            def hx(x):
                return self._model.h(x)

            ukf = UnscentedKalmanFilter(dim_x=nx, dim_z=ny, dt=1.0, fx=fx, hx=hx, points=points)
            ukf.x = x0_mean.copy()
            ukf.P = x0_cov.copy()
            ukf.Q = Q
            ukf.R = R
            # Wrap the bearing residual to (-pi, pi] so the innovation and the
            # measurement-covariance sigma spread are correct across the branch
            # cut (Issues 5/6). No-op when there are no angular components.
            if angular_idx.size:
                ukf.residual_z = _residual_z

            for t in range(T):
                t_box["t"] = float(timestamps[t])
                ukf.predict()
                ukf.update(observations[i, t])
                estimates[i, t] = ukf.x
                if with_cov:
                    covs[i, t] = ukf.P

        return (estimates, covs) if with_cov else estimates

    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        return self._run(dataset, with_cov=False)

    def estimate_with_covariance(self, dataset: TrajectoryDataset):
        """(estimates, covariances) from filterpy's maintained P (Issue 7)."""
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
    def load(cls, path: Path) -> "FilterpyUKFEstimator":
        raise NotImplementedError(
            "FilterpyUKFEstimator.load requires a FilterModel. "
            "Reconstruct from a BenchmarkLevel.get_filter_model()."
        )
