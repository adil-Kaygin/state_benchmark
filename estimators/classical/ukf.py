from __future__ import annotations  
  
import json  
from pathlib import Path  
from typing import Optional, Tuple  
  
import numpy as np  
  
from ..base import BaseEstimator
from benchmark_levels.base import FilterModel
from datasets.schema import TrajectoryDataset
from ._numba_kernels import ukf_loop_batch


class UKFEstimator(BaseEstimator):

    def __init__(
        self,
        filter_model: FilterModel,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        use_numba: bool = True,
    ) -> None:
        self._model = filter_model
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        # use_numba dispatches the sigma-point recursion to the general @njit
        # kernel (estimators/classical/_numba_kernels.py:ukf_loop_batch), which
        # propagates sigma points through the level's @njit f/h
        # (FilterModel.numba) and is therefore valid on every level (linear,
        # pendulum, nonlinear, lorenz). Falls back to the pure-NumPy UKF below
        # when the level ships no numba dynamics or numba isn't installed.
        self._use_numba = use_numba
  
    @property  
    def estimator_name(self) -> str:  
        return "ukf"  
  
    @property  
    def estimator_type(self) -> str:  
        return "classical"  
  
    def _sigma_points(  
        self,  
        x: np.ndarray,  
        P: np.ndarray,  
        nx: int,  
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  
        lam = self._alpha ** 2 * (nx + self._kappa) - nx  
  
        try:  
            L = np.linalg.cholesky((nx + lam) * P)  
        except np.linalg.LinAlgError:  
            L = np.linalg.cholesky((nx + lam) * (P + 1e-6 * np.eye(nx)))  
  
        pts = np.zeros((2 * nx + 1, nx))  
        pts[0] = x  
        for j in range(nx):  
            pts[j + 1] = x + L[:, j]  
            pts[nx + j + 1] = x - L[:, j]  
  
        Wm = np.full(2 * nx + 1, 1.0 / (2.0 * (nx + lam)))  
        Wc = np.full(2 * nx + 1, 1.0 / (2.0 * (nx + lam)))  
        Wm[0] = lam / (nx + lam)  
        Wc[0] = lam / (nx + lam) + (1.0 - self._alpha ** 2 + self._beta)  
  
        return pts, Wm, Wc  
  
    def fit(  
        self,  
        train_dataset: Optional[TrajectoryDataset],  
        val_dataset: Optional[TrajectoryDataset],  
    ) -> None:  
        pass  # UKF requires no training.  
  
    def estimate(self, dataset: TrajectoryDataset) -> np.ndarray:
        observations = np.asarray(dataset.observations)
        N, T, ny = observations.shape
        nx = self._model.Q.shape[0]
        Q = self._model.Q
        R = self._model.R

        x0_mean = self._model.x0_mean if self._model.x0_mean is not None else np.zeros(nx)
        x0_cov = self._model.x0_cov if self._model.x0_cov is not None else np.eye(nx)

        timestamps = np.asarray(dataset.timestamps)

        if self._use_numba and self._model.numba is not None:
            nd = self._model.numba
            return ukf_loop_batch(
                nd.f, nd.h,
                np.ascontiguousarray(Q, dtype=np.float64),
                np.ascontiguousarray(R, dtype=np.float64),
                np.ascontiguousarray(observations, dtype=np.float64),
                np.ascontiguousarray(timestamps, dtype=np.float64),
                self._alpha, self._beta, self._kappa,
                np.ascontiguousarray(x0_mean, dtype=np.float64),
                np.ascontiguousarray(x0_cov, dtype=np.float64),
            )

        estimates = np.zeros((N, T, nx))

        for i in range(N):
            x = x0_mean.copy()
            P = x0_cov.copy()

            for t in range(T):
                pts, Wm, Wc = self._sigma_points(x, P, nx)

                pts_pred = np.array([self._model.f(sp, float(timestamps[t])) for sp in pts])
                x_pred = np.einsum("i,ij->j", Wm, pts_pred)
                P_pred = Q.copy()
                for j in range(2 * nx + 1):
                    d = pts_pred[j] - x_pred
                    P_pred += Wc[j] * np.outer(d, d)

                pts_obs = np.array([self._model.h(sp) for sp in pts_pred])
                y_pred = np.einsum("i,ij->j", Wm, pts_obs)
                S = R.copy()
                Pxy = np.zeros((nx, ny))
                for j in range(2 * nx + 1):
                    d_obs = pts_obs[j] - y_pred
                    d_state = pts_pred[j] - x_pred
                    S += Wc[j] * np.outer(d_obs, d_obs)
                    Pxy += Wc[j] * np.outer(d_state, d_obs)

                K = Pxy @ np.linalg.inv(S)
                x = x_pred + K @ (observations[i, t] - y_pred)
                P = P_pred - K @ S @ K.T
                estimates[i, t] = x

        return estimates
  
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
                    "use_numba": self._use_numba,
                },
                f,  
            )  
  
    @classmethod  
    def load(cls, path: Path) -> UKFEstimator:  
        raise NotImplementedError(  
            "UKFEstimator.load requires a FilterModel. "  
            "Reconstruct from a BenchmarkLevel.get_filter_model()."  
        )
