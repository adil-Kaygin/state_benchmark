from __future__ import annotations  
  
import datetime  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
from .base import (
    BenchmarkLevel,
    BaseSimulator,
    FilterModel,
    NumbaDynamics,
    split_counts as _split_counts,
    gaussian_noise as _gaussian_noise,
)
from ._numba_dynamics import build_linear_numba_dynamics
from ._torch_dynamics import build_linear_torch_dynamics
  
  
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

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("position", "velocity")

    def generate_dataset(self, output_dir: Path) -> None:
        from datasets.schema import DatasetMetadata
        from datasets.hdf5_writer import HDF5Writer

        rng = np.random.default_rng(self._random_seed)
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = _split_counts(self._num_trajectories)

        nx = self.state_dimension
        ny = self.observation_dimension
        T = self._trajectory_length
        # Uniform initial conditions over a box matched in variance to the old
        # Gaussian init (Var[U(-a,a)] = a^2/3 = initial_state_var => a = sqrt(3*var)).
        # A uniform spread gives the data-driven models a more even coverage of
        # the state space than a Gaussian clustered at the origin; the linear
        # dynamics/observation noise (Q, R) stay Gaussian, matching the KF model.
        init_half = float(np.sqrt(3.0 * self._initial_state_var))

        for split_name, n_traj in splits.items():
            states = np.zeros((n_traj, T, nx))
            observations = np.zeros((n_traj, T, ny))
            timestamps = np.arange(T, dtype=float) * self._dt

            # Draw all noise / initial states up front (one vectorized call each)
            # instead of one rng.multivariate_normal per timestep -- same draws,
            # far less Python/RNG overhead.
            x0 = rng.uniform(-init_half, init_half, size=(n_traj, nx))
            proc_noise = _gaussian_noise(rng, self._Q, (n_traj, T))
            obs_noise = _gaussian_noise(rng, self._R, (n_traj, T))

            for i in range(n_traj):
                x = x0[i]
                for t in range(T):
                    states[i, t] = x
                    observations[i, t] = self._H @ x + obs_noise[i, t]
                    x = self._F @ x + proc_noise[i, t]

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
            # x0_cov = variance of the uniform init box (a^2/3 = initial_state_var),
            # so the filter's prior matches the data-generating distribution.
            x0_mean=np.zeros(2), x0_cov=np.eye(2) * self._initial_state_var,
            numba=build_linear_numba_dynamics(F_mat, H_mat),
            torch=build_linear_torch_dynamics(F_mat, H_mat),
        )
