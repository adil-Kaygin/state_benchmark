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
    """Extended Kalman filter (first-order linearization at the current state).

    Valid on any FilterModel. The recursion runs exclusively in the @njit
    ekf_loop_batch kernel driven by the level's @njit dynamics
    (FilterModel.numba); there is no pure-NumPy fallback. Per the "fail fast and
    loud" rule, a model without numba dynamics raises ValueError rather than
    silently degrading.
    """

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
        if self._model.numba is None:
            raise ValueError(
                "EKFEstimator requires FilterModel.numba (@njit dynamics); this "
                "model provides none. The pure-NumPy EKF path has been removed -- "
                "every level must ship NumbaDynamics (see _numba_dynamics.py)."
            )

        observations = np.asarray(dataset.observations)
        nx = self._model.Q.shape[0]
        Q = self._model.Q
        R = self._model.R
        timestamps = np.asarray(dataset.timestamps)

        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)

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
