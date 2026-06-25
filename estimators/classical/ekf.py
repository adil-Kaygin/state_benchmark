from __future__ import annotations  
  
import json  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
from ..base import BaseEstimator  
from benchmark_levels.base import FilterModel  
from datasets.schema import TrajectoryDataset  
  
  
class EKFEstimator(BaseEstimator):  
  
    def __init__(self, filter_model: FilterModel) -> None:  
        self._model = filter_model  
  
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
        # Not numba-jitted: f/F/h/H are arbitrary Python callables from
        # FilterModel (e.g. pendulum trig), which numba cannot compile
        # without per-benchmark specialization. KF/UKF linear fast paths
        # live in ._numba_kernels for benchmarks with constant F/H.
        observations = np.asarray(dataset.observations)
        N, T, ny = observations.shape  
        nx = self._model.Q.shape[0]  
        Q = self._model.Q  
        R = self._model.R  
  
        estimates = np.zeros((N, T, nx))  
  
        for i in range(N):  
            x = np.zeros(nx)  
            P = np.eye(nx)  
            for t in range(T):  
                x_pred = self._model.f(x)  
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
                 "estimator_type": self.estimator_type},  
                f,  
            )  
  
    @classmethod  
    def load(cls, path: Path) -> EKFEstimator:  
        raise NotImplementedError(  
            "EKFEstimator.load requires a FilterModel. "  
            "Reconstruct from a BenchmarkLevel.get_filter_model()."  
        )
