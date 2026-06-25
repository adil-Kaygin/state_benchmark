from __future__ import annotations  
  
import json  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
from ..base import BaseEstimator  
from benchmark_levels.base import FilterModel  
from datasets.schema import TrajectoryDataset  
  
  
class ParticleFilterEstimator(BaseEstimator):  
  
    def __init__(  
        self,  
        filter_model: FilterModel,  
        num_particles: int = 1000,  
        resample_threshold: float = 0.5,  
    ) -> None:  
        self._model = filter_model  
        self._num_particles = num_particles  
        self._resample_threshold = resample_threshold  
  
    @property  
    def estimator_name(self) -> str:  
        return "particle_filter"  
  
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
        observations = np.asarray(dataset.observations)  
        N, T, ny = observations.shape  
        nx = self._model.Q.shape[0]  
        M = self._num_particles  
        Q = self._model.Q  
        R_inv = np.linalg.inv(self._model.R)  
  
        estimates = np.zeros((N, T, nx))  
  
        rng = np.random.default_rng()  
  
        for i in range(N):  
            particles = rng.multivariate_normal(np.zeros(nx), np.eye(nx), size=M)  
            weights = np.full(M, 1.0 / M)  
  
            for t in range(T):  
                particles = np.array([  
                    self._model.f(p) + rng.multivariate_normal(np.zeros(nx), Q)  
                    for p in particles  
                ])  
  
                y = observations[i, t]  
                log_w = np.array([  
                    -0.5 * (y - self._model.h(p)) @ R_inv @ (y - self._model.h(p))  
                    for p in particles  
                ])  
                log_w -= log_w.max()  
                weights = np.exp(log_w)  
                weights /= weights.sum()  
  
                estimates[i, t] = weights @ particles  
  
                n_eff = 1.0 / (weights ** 2).sum()  
                if n_eff < self._resample_threshold * M:  
                    indices = rng.choice(M, size=M, replace=True, p=weights)  
                    particles = particles[indices]  
                    weights = np.full(M, 1.0 / M)  
  
        return estimates  
  
    def save(self, path: Path) -> None:  
        path.parent.mkdir(parents=True, exist_ok=True)  
        with open(path, "w") as f:  
            json.dump(  
                {  
                    "estimator_name": self.estimator_name,  
                    "estimator_type": self.estimator_type,  
                    "num_particles": self._num_particles,  
                    "resample_threshold": self._resample_threshold,  
                },  
                f,  
            )  
  
    @classmethod  
    def load(cls, path: Path) -> ParticleFilterEstimator:  
        raise NotImplementedError(  
            "ParticleFilterEstimator.load requires a FilterModel. "  
            "Reconstruct from a BenchmarkLevel.get_filter_model()."  
        )
