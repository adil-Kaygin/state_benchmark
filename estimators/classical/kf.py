from __future__ import annotations  
  
import json  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
from ..base import BaseEstimator
from benchmark_levels.base import FilterModel
from datasets.schema import TrajectoryDataset
from ._numba_kernels import kf_loop_batch


class KalmanFilterEstimator(BaseEstimator):

    def __init__(self, filter_model: FilterModel, use_numba: bool = True) -> None:
        self._model = filter_model
        self._use_numba = use_numba
  
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
  
        # F and H are constant matrices for the linear case.
        F = self._model.F(np.zeros(nx))
        H = self._model.H(np.zeros(nx))

        if self._use_numba:
            obs64 = np.ascontiguousarray(observations, dtype=np.float64)
            return kf_loop_batch(
                np.ascontiguousarray(F, dtype=np.float64),
                np.ascontiguousarray(H, dtype=np.float64),
                np.ascontiguousarray(Q, dtype=np.float64),
                np.ascontiguousarray(R, dtype=np.float64),
                obs64,
            )

        estimates = np.zeros((N, T, nx))

        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)

        for i in range(N):
            x = x0_mean.copy()
            P = x0_cov.copy()
            for t in range(T):
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q
                y = observations[i, t]
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ np.linalg.inv(S)
                x = x_pred + K @ (y - H @ x_pred)
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
    def load(cls, path: Path) -> KalmanFilterEstimator:  
        raise NotImplementedError(  
            "KalmanFilterEstimator.load requires a FilterModel. "  
            "Reconstruct from a BenchmarkLevel.get_filter_model()."  
        )
