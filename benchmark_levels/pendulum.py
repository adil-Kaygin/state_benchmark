from __future__ import annotations  
  
import datetime  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
from .base import BenchmarkLevel, BaseSimulator, FilterModel  
  
  
class PendulumSimulator(BaseSimulator):  
  
    def __init__(  
        self,  
        g: float,  
        length: float,  
        Q: np.ndarray,  
        R: np.ndarray,  
        rng: Optional[np.random.Generator] = None,  
    ) -> None:  
        self._g = g  
        self._length = length  
        self._Q = Q  
        self._R = R  
        self._rng = rng if rng is not None else np.random.default_rng()  
  
    def step(  
        self,  
        state: np.ndarray,  
        control: Optional[np.ndarray],  
        dt: float,  
    ) -> np.ndarray:  
        theta, omega = state  
        alpha = -(self._g / self._length) * np.sin(theta)  
        new_state = np.array([theta + omega * dt, omega + alpha * dt])  
        noise = self._rng.multivariate_normal(np.zeros(2), self._Q)  
        return new_state + noise  
  
    def observe(self, state: np.ndarray) -> np.ndarray:  
        noise = self._rng.multivariate_normal(np.zeros(1), self._R)  
        return np.array([state[0]]) + noise  
  
  
class PendulumBenchmark(BenchmarkLevel):  
  
    def __init__(
        self,
        trajectory_length: int = 100,
        num_trajectories: int = 500,
        random_seed: int = 42,
        dt: float = 0.05,
        g: float = 9.81,
        length: float = 1.0,
        process_noise_var: float = 0.001,
        observation_noise_var: float = 0.01,
        initial_angle_range: float = np.pi / 4,
    ) -> None:
        self._trajectory_length = trajectory_length
        self._num_trajectories = num_trajectories
        self._random_seed = random_seed
        self._dt = dt
        self._g = g
        self._length = length
        self._initial_angle_range = initial_angle_range
        self._Q = np.eye(2) * process_noise_var
        self._R = np.eye(1) * observation_noise_var
  
    @property  
    def name(self) -> str:  
        return "pendulum"  
  
    @property  
    def description(self) -> str:  
        return "Nonlinear pendulum state estimation benchmark."  
  
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
        simulator = PendulumSimulator(self._g, self._length, self._Q, self._R, rng=rng)  
  
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
                x = np.array([
                    rng.uniform(-self._initial_angle_range, self._initial_angle_range), 0.0
                ])
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
                generation_time=datetime.datetime.utcnow().isoformat(),  
            )  
            HDF5Writer(output_dir / f"{split_name}.h5").write(  
                states, observations, timestamps, metadata  
            )  
  
    def get_filter_model(self) -> FilterModel:  
        g = self._g  
        length = self._length  
        dt = self._dt  
  
        def f(x: np.ndarray, t: float = 0.0) -> np.ndarray:  
            theta, omega = x  
            alpha = -(g / length) * np.sin(theta)  
            return np.array([theta + omega * dt, omega + alpha * dt])  
  
        def h(x: np.ndarray, t: float = 0.0) -> np.ndarray:
            return np.array([x[0]])
  
        def F_jac(x: np.ndarray) -> np.ndarray:  
            theta = x[0]  
            return np.array([  
                [1.0, dt],  
                [-(g / length) * np.cos(theta) * dt, 1.0],  
            ])  
  
        def H_jac(x: np.ndarray) -> np.ndarray:  
            return np.array([[1.0, 0.0]])  
  
        theta_var = (self._initial_angle_range ** 2) / 3.0
        x0_cov = np.diag([theta_var, 1e-6])

        return FilterModel(  
            f=f, h=h, F=F_jac, H=H_jac,  
            Q=self._Q.copy(), R=self._R.copy(),  
            x0_mean=np.zeros(2), x0_cov=x0_cov,  
        )
