from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import BaseEstimator
from benchmark_levels.base import FilterModel
from datasets.schema import TrajectoryDataset
from ._numba_kernels import kf_loop_batch, assert_linear_model


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

    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        observations = np.asarray(dataset.observations)
        N, T, ny = observations.shape
        nx = self._model.Q.shape[0]
        Q = self._model.Q
        R = self._model.R

        # F and H must be constant matrices (linear model). Evaluating the
        # Jacobian at any point gives the same matrix for a linear model.
        F = self._model.F(np.zeros(nx))
        H = self._model.H(np.zeros(nx))

        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)

        # Strict linear check: crash loudly on a nonlinear model rather than
        # running an origin-linearized filter that would silently diverge.
        assert_linear_model(self._model.f, self._model.h, F, H, nx, ny)

        obs64 = np.ascontiguousarray(observations, dtype=np.float64)
        return kf_loop_batch(
            np.ascontiguousarray(F, dtype=np.float64),
            np.ascontiguousarray(H, dtype=np.float64),
            np.ascontiguousarray(Q, dtype=np.float64),
            np.ascontiguousarray(R, dtype=np.float64),
            obs64,
            np.ascontiguousarray(x0_mean, dtype=np.float64),
            np.ascontiguousarray(x0_cov, dtype=np.float64),
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
