from __future__ import annotations  
  
import json  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
from ..base import BaseEstimator
from benchmark_levels.base import FilterModel
from datasets.schema import TrajectoryDataset
from ._numba_kernels import ekf_loop_batch


class EKFEstimator(BaseEstimator):

    def __init__(self, filter_model: FilterModel, use_numba: bool = True) -> None:
        self._model = filter_model
        # use_numba dispatches the inner recursion to an @njit kernel
        # (estimators/classical/_numba_kernels.py:ekf_loop_batch) driven by the
        # level's @njit dynamics (FilterModel.numba). Unlike KF/UKF's linear
        # fast path this is the *general* nonlinear EKF, so it is valid on every
        # level. Falls back to the pure-NumPy loop below if the level ships no
        # numba dynamics or numba isn't installed.
        self._use_numba = use_numba
  
    @property  
    def estimator_name(self) -> str:  
        return "ekf"  
  
    @property  
    def estimator_type(self) -> str:  
        return "classical"  
  
    def fit(  
        self,  
        train_dataset: Optional[TrajectoryDataset],  
        val_dataset: Optional[TrajectoryDataset],  
    ) -> None:  
        pass  # EKF requires no training.  
  
    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        observations = np.asarray(dataset.observations)
        N, T, ny = observations.shape
        nx = self._model.Q.shape[0]
        Q = self._model.Q
        R = self._model.R

        timestamps = np.asarray(dataset.timestamps)

        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)

        if self._use_numba and self._model.numba is not None:
            nd = self._model.numba
            return ekf_loop_batch(
                nd.f, nd.h, nd.F_jac, nd.H_jac,
                np.ascontiguousarray(Q, dtype=np.float64),
                np.ascontiguousarray(R, dtype=np.float64),
                np.ascontiguousarray(observations, dtype=np.float64),
                np.ascontiguousarray(timestamps, dtype=np.float64),
                np.ascontiguousarray(x0_mean, dtype=np.float64),
                np.ascontiguousarray(x0_cov, dtype=np.float64),
            )

        estimates = np.zeros((N, T, nx))

        for i in range(N):
            x = x0_mean.copy()
            P = x0_cov.copy()
            for t in range(T):
                x_pred = self._model.f(x, float(timestamps[t]))
                F = self._model.F(x)
                P_pred = F @ P @ F.T + Q

                H = self._model.H(x_pred)
                y_pred = self._model.h(x_pred)
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)

                x = x_pred + K @ (observations[i, t] - y_pred)
                P = (np.eye(nx) - K @ H) @ P_pred
                estimates[i, t] = x

        return estimates
  
    def save(self, path: Path) -> None:  
        path.parent.mkdir(parents=True, exist_ok=True)  
        with open(path, "w") as f:  
            json.dump(
                {"estimator_name": self.estimator_name,
                 "estimator_type": self.estimator_type,
                 "use_numba": self._use_numba},
                f,
            )
  
    @classmethod  
    def load(cls, path: Path) -> EKFEstimator:  
        raise NotImplementedError(  
            "EKFEstimator.load requires a FilterModel. "  
            "Reconstruct from a BenchmarkLevel.get_filter_model()."  
        )
