from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..base import BaseEstimator
from benchmark_levels.base import FilterModel
from datasets.schema import TrajectoryDataset
from ._numba_kernels import (
    ukf_loop_batch,
    ukf_loop_batch_cov,
    angular_mask_float as _angular_mask,
)


class UKFEstimator(BaseEstimator):
    """Unscented Kalman filter (sigma-point propagation through true f/h).

    Valid on any FilterModel. The recursion runs exclusively in the general
    @njit ukf_loop_batch kernel, which propagates sigma points through the
    level's @njit dynamics (FilterModel.numba); there is no pure-NumPy
    fallback. Per the "fail fast and loud" rule, a model without numba dynamics
    raises ValueError rather than silently degrading.
    """

    # Issue 7: the UKF propagates a real posterior covariance P every step, so it
    # opts into the NEES/NLL consistency scoring via estimate_with_covariance().
    returns_covariance = True

    def __init__(
        self,
        filter_model: FilterModel,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self._model = filter_model
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa

    @property
    def estimator_name(self) -> str:
        return "ukf"

    @property
    def estimator_type(self) -> str:
        return "classical"

    def fit(
        self,
        train_dataset: Optional[TrajectoryDataset],
        val_dataset: Optional[TrajectoryDataset],
    ) -> None:
        pass  # UKF requires no training.

    def _prepare(self, dataset: TrajectoryDataset):
        """Shared argument prep for estimate()/estimate_with_covariance()."""
        if self._model.numba is None:
            raise ValueError(
                "UKFEstimator requires FilterModel.numba (@njit dynamics); this "
                "model provides none. The pure-NumPy UKF path has been removed -- "
                "every level must ship NumbaDynamics (see _numba_dynamics.py)."
            )
        observations = np.asarray(dataset.observations)
        ny = observations.shape[-1]
        nx = self._model.Q.shape[0]
        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)
        nd = self._model.numba
        return dict(
            nd=nd,
            Q=np.ascontiguousarray(self._model.Q, dtype=np.float64),
            R=np.ascontiguousarray(self._model.R, dtype=np.float64),
            obs=np.ascontiguousarray(observations, dtype=np.float64),
            ts=np.ascontiguousarray(np.asarray(dataset.timestamps), dtype=np.float64),
            x0_mean=np.ascontiguousarray(x0_mean, dtype=np.float64),
            x0_cov=np.ascontiguousarray(x0_cov, dtype=np.float64),
            angular_mask=_angular_mask(self._model, ny),
        )

    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        a = self._prepare(dataset)
        nd = a["nd"]
        return ukf_loop_batch(
            nd.f, nd.h, a["Q"], a["R"], a["obs"], a["ts"],
            self._alpha, self._beta, self._kappa,
            a["x0_mean"], a["x0_cov"], a["angular_mask"],
        )

    def estimate_with_covariance(self, dataset: TrajectoryDataset):
        """Return (estimates [N,T,nx], covariances [N,T,nx,nx]) -- the UKF's
        propagated posterior P at each step, for the NEES/NLL metrics (Issue 7)."""
        a = self._prepare(dataset)
        nd = a["nd"]
        return ukf_loop_batch_cov(
            nd.f, nd.h, a["Q"], a["R"], a["obs"], a["ts"],
            self._alpha, self._beta, self._kappa,
            a["x0_mean"], a["x0_cov"], a["angular_mask"],
        )

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
    def load(cls, path: Path) -> UKFEstimator:
        raise NotImplementedError(
            "UKFEstimator.load requires a FilterModel. "
            "Reconstruct from a BenchmarkLevel.get_filter_model()."
        )
