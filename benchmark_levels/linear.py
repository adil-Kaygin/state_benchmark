from __future__ import annotations  
  
import datetime  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
from .base import BenchmarkLevel, BaseSimulator, FilterModel  
  
  
class LinearSimulator(BaseSimulator):  
  
    def __init__(  
        self,  
        F: np.ndarray,  
        H: np.ndarray,  
        Q: np.ndarray,  
        R: np.ndarray,  
        rng: Optional[np.random.Generator] = None,  
    ) -> None:  
        self._F = F  
        self._H = H  
        self._Q = Q  
        self._R = R  
        self._rng = rng if rng is not None else np.random.default_rng()  
  
    def step(  
        self,  
        state: np.ndarray,  
        control: Optional[np.ndarray],  
        dt: float,  
    ) -> np.ndarray:  
        noise = self._rng.multivariate_normal(np.zeros(self._Q.shape[0]), self._Q)  
        return self._F @ state + noise  
  
    def observe(self, state: np.ndarray) -> np.ndarray:  
        noise = self._rng.multivariate_normal(np.zeros(self._R.shape[0]), self._R)  
        return self._H @ state + noise  
  
  
class LinearBenchmark(BenchmarkLevel):  
  
    def __init__(
        self,
        trajectory_length: int = 100,
        num_trajectories: int = 500,
        random_seed: int = 42,
        dt: float = 0.1,
        process_noise_var: float = 0.01,
        observation_noise_var: float = 0.1,
        initial_state_var: float = 1.0,
    ) -> None:
        self._trajectory_length = trajectory_length
        self._num_trajectories = num_trajectories
        self._random_seed = random_seed
        self._dt = dt
        self._initial_state_var = initial_state_var

        self._F = np.array([[1.0, dt], [0.0, 1.0]])
        self._H = np.array([[1.0, 0.0]])
        self._Q = np.eye(2) * process_noise_var
        self._R = np.eye(1) * observation_noise_var
  
    @property  
    def name(self) -> str:  
        return "linear"  
  
    @property  
    def description(self) -> str:  
        return "Linear constant-velocity state estimation benchmark."  
  
    @property  
    def state_dimension(self) -> int:  
        return 2  
  
    @property  
    def observation_dimension(self) -> int:  
        return 1  
  
    def generate_dataset(self, output_dir: Path) -> None:  
        from datasets.schema import DatasetMetadata  
        from datasets.hdf5_writer import HDF5Writer  
  
        rng = np.random.default_rng(self._random_seed)  
        output_dir.mkdir(parents=True, exist_ok=True)  
        simulator = LinearSimulator(self._F, self._H, self._Q, self._R, rng=rng)  
  
        splits = {  
            "train": int(self._num_trajectories * 0.7),  
            "val": int(self._num_trajectories * 0.15),  
            "test": int(self._num_trajectories * 0.15),  
        }  
  
        for split_name, n_traj in splits.items():  
            states = np.zeros((n_traj, self._trajectory_length, self.state_dimension))  
            observations = np.zeros((n_traj, self._trajectory_length, self.observation_dimension))  
            timestamps = np.arange(self._trajectory_length, dtype=float) * self._dt  
  
            for i in range(n_traj):
                x = rng.standard_normal(self.state_dimension) * np.sqrt(self._initial_state_var)
                for t in range(self._trajectory_length):  
                    states[i, t] = x  
                    observations[i, t] = simulator.observe(x)  
                    x = simulator.step(x, None, self._dt)  
  
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
        F_mat = self._F.copy()  
        H_mat = self._H.copy()  
  
        def f(x: np.ndarray, t: float = 0.0) -> np.ndarray:
            return F_mat @ x
  
        def h(x: np.ndarray, t: float = 0.0) -> np.ndarray:
            return H_mat @ x  
  
        def F_jac(x: np.ndarray) -> np.ndarray:  
            return F_mat  
  
        def H_jac(x: np.ndarray) -> np.ndarray:  
            return H_mat  
  
        return FilterModel(  
            f=f, h=h, F=F_jac, H=H_jac,  
            Q=self._Q.copy(), R=self._R.copy(),  
            x0_mean=np.zeros(2), x0_cov=np.eye(2) * self._initial_state_var,  
        )
