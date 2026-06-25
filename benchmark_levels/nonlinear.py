from __future__ import annotations  
  
import datetime  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
from .base import BenchmarkLevel, BaseSimulator, FilterModel
from ._numba_dynamics import build_nonlinear_numba_dynamics
  
  
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
  
    def generate_dataset(self, output_dir: Path) -> None:  
        from datasets.schema import DatasetMetadata  
        from datasets.hdf5_writer import HDF5Writer  
  
        rng = np.random.default_rng(self._random_seed)  
        output_dir.mkdir(parents=True, exist_ok=True)  
        simulator = NonlinearSimulator(self._Q, self._R, rng=rng)  
  
        splits = {  
            "train": int(self._num_trajectories * 0.7),  
            "val": int(self._num_trajectories * 0.15),  
            "test": int(self._num_trajectories * 0.15),  
        }  
  
        for split_name, n_traj in splits.items():  
            states = np.zeros((n_traj, self._trajectory_length, self.state_dimension))  
            observations = np.zeros((n_traj, self._trajectory_length, self.observation_dimension))  
            timestamps = np.arange(self._trajectory_length, dtype=float)  
  
            for i in range(n_traj):  
                x = rng.standard_normal(self.state_dimension)  
                for t in range(self._trajectory_length):  
                    states[i, t] = x  
                    observations[i, t] = simulator.observe(x)  
                    x = simulator.step(x, None, float(t))  
  
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
        )
