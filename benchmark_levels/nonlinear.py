from __future__ import annotations  
  
import datetime  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
from .base import (
    BenchmarkLevel,
    BaseSimulator,
    FilterModel,
    split_counts as _split_counts,
    gaussian_noise as _gaussian_noise,
)
from ._numba_dynamics import build_nonlinear_numba_dynamics
from ._torch_dynamics import build_nonlinear_torch_dynamics
  
  
class NonlinearSimulator(BaseSimulator):  
    """Gordon et al. (1993) scalar nonlinear benchmark."""  
  
    def __init__(  
        self,  
        Q: np.ndarray,  
        R: np.ndarray,  
        rng: Optional[np.random.Generator] = None,  
    ) -> None:  
        self._Q = Q  
        self._R = R  
        self._rng = rng if rng is not None else np.random.default_rng()  
  
    def step(  
        self,  
        state: np.ndarray,  
        control: Optional[np.ndarray],  
        dt: float,  
    ) -> np.ndarray:  
        # dt is used as the discrete time index t.  
        x = state[0]  
        new_x = 0.5 * x + 25.0 * x / (1.0 + x ** 2) + 8.0 * np.cos(1.2 * dt)  
        noise = self._rng.multivariate_normal(np.zeros(1), self._Q)  
        return np.array([new_x]) + noise  
  
    def observe(self, state: np.ndarray) -> np.ndarray:  
        noise = self._rng.multivariate_normal(np.zeros(1), self._R)  
        return np.array([state[0] ** 2 / 20.0]) + noise  
  
  
class NonlinearBenchmark(BenchmarkLevel):  
  
    def __init__(  
        self,  
        trajectory_length: int = 100,  
        num_trajectories: int = 500,  
        random_seed: int = 42,  
    ) -> None:  
        self._trajectory_length = trajectory_length  
        self._num_trajectories = num_trajectories  
        self._random_seed = random_seed  
        self._Q = np.eye(1) * 1.0  
        self._R = np.eye(1) * 1.0  
  
    @property  
    def name(self) -> str:  
        return "nonlinear"  
  
    @property  
    def description(self) -> str:  
        return "Strongly nonlinear scalar state estimation benchmark (Gordon et al. 1993)."  
  
    @property
    def state_dimension(self) -> int:
        return 1

    @property
    def observation_dimension(self) -> int:
        return 1

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("x",)

    def generate_dataset(self, output_dir: Path) -> None:
        from datasets.schema import DatasetMetadata
        from datasets.hdf5_writer import HDF5Writer

        rng = np.random.default_rng(self._random_seed)
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = _split_counts(self._num_trajectories)

        nx = self.state_dimension
        ny = self.observation_dimension
        T = self._trajectory_length
        # Uniform init matched in variance to the old standard-normal init
        # (Var[U(-a,a)] = a^2/3 = 1 => a = sqrt(3)); x0_cov stays eye(1) so the
        # filter prior still matches. Wider, more even coverage of the state for
        # the data-driven models; Q/R noise stays Gaussian (the model assumption).
        init_half = float(np.sqrt(3.0))

        for split_name, n_traj in splits.items():
            states = np.zeros((n_traj, T, nx))
            observations = np.zeros((n_traj, T, ny))
            timestamps = np.arange(T, dtype=float)

            x0 = rng.uniform(-init_half, init_half, size=(n_traj, nx))
            proc_noise = _gaussian_noise(rng, self._Q, (n_traj, T))
            obs_noise = _gaussian_noise(rng, self._R, (n_traj, T))

            for i in range(n_traj):
                x = x0[i]
                for t in range(T):
                    states[i, t] = x
                    observations[i, t] = np.array([x[0] ** 2 / 20.0]) + obs_noise[i, t]
                    xv = x[0]
                    new_x = 0.5 * xv + 25.0 * xv / (1.0 + xv ** 2) + 8.0 * np.cos(1.2 * float(t))
                    x = np.array([new_x]) + proc_noise[i, t]

            metadata = DatasetMetadata(  
                benchmark_name=self.name,  
                state_dimension=self.state_dimension,  
                observation_dimension=self.observation_dimension,  
                trajectory_length=self._trajectory_length,  
                num_trajectories=n_traj,  
                random_seed=self._random_seed,  
                generation_time=datetime.datetime.now(datetime.UTC).isoformat(),
            )  
            HDF5Writer(output_dir / f"{split_name}.h5").write(  
                states, observations, timestamps, metadata  
            )  
  
    def get_filter_model(self) -> FilterModel:  
  
        def f(x: np.ndarray, t: float = 0.0) -> np.ndarray:  
            xv = x[0]  
            return np.array([0.5 * xv + 25.0 * xv / (1.0 + xv ** 2) + 8.0 * np.cos(1.2 * t)])  
  
        def h(x: np.ndarray, t: float = 0.0) -> np.ndarray:
            return np.array([x[0] ** 2 / 20.0])
  
        def F_jac(x: np.ndarray) -> np.ndarray:  
            xv = x[0]  
            df_dx = 0.5 + 25.0 * (1.0 - xv ** 2) / (1.0 + xv ** 2) ** 2  
            return np.array([[df_dx]])  
  
        def H_jac(x: np.ndarray) -> np.ndarray:  
            return np.array([[x[0] / 10.0]])  
  
        return FilterModel(
            f=f, h=h, F=F_jac, H=H_jac,
            Q=self._Q.copy(), R=self._R.copy(),
            x0_mean=np.zeros(1), x0_cov=np.eye(1),
            numba=build_nonlinear_numba_dynamics(),
            torch=build_nonlinear_torch_dynamics(),
        )
