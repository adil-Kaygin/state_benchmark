from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import BaseEstimator
from benchmark_levels.base import FilterModel
from datasets.schema import TrajectoryDataset
from ._numba_kernels import kf_loop_batch, kf_loop_batch_cov, assert_linear_model


class KalmanFilterEstimator(BaseEstimator):
    """Optimal linear-Gaussian Kalman filter.

    Statistically valid ONLY on a linear-Gaussian model (f(x)=F@x, h(x)=H@x),
    e.g. LinearBenchmark. Per the "fail fast and loud" rule this estimator
    refuses to run on a nonlinear model instead of silently linearizing it at
    the origin: every estimate() call asserts the model is linear via
    assert_linear_model and raises ValueError otherwise. There is no
    pure-NumPy fallback -- the recursion runs exclusively in the @njit
    kf_loop_batch kernel; use EKF/UKF/PF for nonlinear systems.
    """

    # Issue 7: the KF propagates a real posterior covariance P every step, so it
    # opts into the NEES/NLL consistency scoring via estimate_with_covariance().
    returns_covariance = True

    def __init__(self, filter_model: FilterModel) -> None:
        self._model = filter_model

    @property
    def estimator_name(self) -> str:
        return "kalman_filter"

    @property
    def estimator_type(self) -> str:
        return "classical"

    def fit(
        self,
        train_dataset: Optional[TrajectoryDataset],
        val_dataset: Optional[TrajectoryDataset],
    ) -> None:
        pass  # KF requires no training.

    def _prepare(self, dataset: TrajectoryDataset):
        """Shared argument prep for estimate()/estimate_with_covariance(),
        including the strict linear-model check."""
        observations = np.asarray(dataset.observations)
        N, T, ny = observations.shape
        nx = self._model.Q.shape[0]

        # F and H must be constant matrices (linear model). Evaluating the
        # Jacobian at any point gives the same matrix for a linear model.
        F = self._model.F(np.zeros(nx))
        H = self._model.H(np.zeros(nx))

        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)

        # Strict linear check: crash loudly on a nonlinear model rather than
        # running an origin-linearized filter that would silently diverge.
        assert_linear_model(self._model.f, self._model.h, F, H, nx, ny)

        return dict(
            F=np.ascontiguousarray(F, dtype=np.float64),
            H=np.ascontiguousarray(H, dtype=np.float64),
            Q=np.ascontiguousarray(self._model.Q, dtype=np.float64),
            R=np.ascontiguousarray(self._model.R, dtype=np.float64),
            obs=np.ascontiguousarray(observations, dtype=np.float64),
            x0_mean=np.ascontiguousarray(x0_mean, dtype=np.float64),
            x0_cov=np.ascontiguousarray(x0_cov, dtype=np.float64),
        )

    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        a = self._prepare(dataset)
        return kf_loop_batch(
            a["F"], a["H"], a["Q"], a["R"], a["obs"], a["x0_mean"], a["x0_cov"],
        )

    def estimate_with_covariance(self, dataset: TrajectoryDataset):
        """Return (estimates [N,T,nx], covariances [N,T,nx,nx]) -- the KF's
        propagated posterior P at each step, for the NEES/NLL metrics (Issue 7)."""
        a = self._prepare(dataset)
        return kf_loop_batch_cov(
            a["F"], a["H"], a["Q"], a["R"], a["obs"], a["x0_mean"], a["x0_cov"],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"estimator_name": self.estimator_name,
                 "estimator_type": self.estimator_type},
                f,
            )

    @classmethod
    def load(cls, path: Path) -> KalmanFilterEstimator:
        raise NotImplementedError(
            "KalmanFilterEstimator.load requires a FilterModel. "
            "Reconstruct from a BenchmarkLevel.get_filter_model()."
        )
