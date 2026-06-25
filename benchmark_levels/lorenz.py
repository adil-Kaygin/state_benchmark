from __future__ import annotations  
  
import datetime  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
from .base import BenchmarkLevel, BaseSimulator, FilterModel
from ._numba_dynamics import build_lorenz_numba_dynamics
  
  
class LorenzSimulator(BaseSimulator):  
  
    def __init__(  
        self,  
        sigma: float,  
        rho: float,  
        beta: float,  
        Q: np.ndarray,  
        R: np.ndarray,  
        rng: Optional[np.random.Generator] = None,  
    ) -> None:  
        self._sigma = sigma  
        self._rho = rho  
        self._beta = beta  
        self._Q = Q  
        self._R = R  
        self._rng = rng if rng is not None else np.random.default_rng()  
  
    def _derivative(self, state: np.ndarray) -> np.ndarray:  
        x, y, z = state  
        return np.array([  
            self._sigma * (y - x),  
            x * (self._rho - z) - y,  
            x * y - self._beta * z,  
        ])  
  
    def step(  
        self,  
        state: np.ndarray,  
        control: Optional[np.ndarray],  
        dt: float,  
    ) -> np.ndarray:  
        k1 = self._derivative(state)  
        k2 = self._derivative(state + 0.5 * dt * k1)  
        k3 = self._derivative(state + 0.5 * dt * k2)  
        k4 = self._derivative(state + dt * k3)  
        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)  
        noise = self._rng.multivariate_normal(np.zeros(3), self._Q)  
        return new_state + noise  
  
    def observe(self, state: np.ndarray) -> np.ndarray:  
        noise = self._rng.multivariate_normal(np.zeros(2), self._R)  
        return np.array([state[0], state[1]]) + noise  
  
  
class LorenzBenchmark(BenchmarkLevel):  
  
    def __init__(  
        self,  
        trajectory_length: int = 200,  
        num_trajectories: int = 500,  
        random_seed: int = 42,  
        dt: float = 0.01,  
        sigma: float = 10.0,  
        rho: float = 28.0,  
        beta: float = 8.0 / 3.0,  
    ) -> None:  
        self._trajectory_length = trajectory_length  
        self._num_trajectories = num_trajectories  
        self._random_seed = random_seed  
        self._dt = dt  
        self._sigma = sigma  
        self._rho = rho  
        self._beta = beta  
        self._Q = np.eye(3) * 0.001  
        self._R = np.eye(2) * 1.0  
  
    @property  
    def name(self) -> str:  
        return "lorenz"  
  
    @property  
    def description(self) -> str:  
        return "Lorenz-63 chaotic system state estimation benchmark."  
  
    @property  
    def state_dimension(self) -> int:  
        return 3  
  
    @property  
    def observation_dimension(self) -> int:  
        return 2  
  
    def generate_dataset(self, output_dir: Path) -> None:  
        from datasets.schema import DatasetMetadata  
        from datasets.hdf5_writer import HDF5Writer  
  
        rng = np.random.default_rng(self._random_seed)  
        output_dir.mkdir(parents=True, exist_ok=True)  
        simulator = LorenzSimulator(  
            self._sigma, self._rho, self._beta, self._Q, self._R, rng=rng  
        )  
  
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
                x = rng.standard_normal(3) + np.array([0.0, 0.0, 25.0])  
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
        sigma = self._sigma  
        rho = self._rho  
        beta = self._beta  
        dt = self._dt  
  
        def _derivative(state: np.ndarray) -> np.ndarray:  
            xv, y, z = state  
            return np.array([  
                sigma * (y - xv),  
                xv * (rho - z) - y,  
                xv * y - beta * z,  
            ])  
  
        def f(x: np.ndarray, t: float = 0.0) -> np.ndarray:  
            k1 = _derivative(x)  
            k2 = _derivative(x + 0.5 * dt * k1)  
            k3 = _derivative(x + 0.5 * dt * k2)  
            k4 = _derivative(x + dt * k3)  
            return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)  
  
        def h(x: np.ndarray, t: float = 0.0) -> np.ndarray:
            return np.array([x[0], x[1]])
  
        def F_jac(x: np.ndarray) -> np.ndarray:  
            xv, y, z = x  
            return np.eye(3) + dt * np.array([  
                [-sigma, sigma, 0.0],  
                [rho - z, -1.0, -xv],  
                [y, xv, -beta],  
            ])  
  
        def H_jac(x: np.ndarray) -> np.ndarray:  
            return np.array([  
                [1.0, 0.0, 0.0],  
                [0.0, 1.0, 0.0],  
            ])  
  
        return FilterModel(
            f=f, h=h, F=F_jac, H=H_jac,
            Q=self._Q.copy(), R=self._R.copy(),
            x0_mean=np.array([0.0, 0.0, 25.0]), x0_cov=np.eye(3),
            numba=build_lorenz_numba_dynamics(sigma, rho, beta, dt),
        )
