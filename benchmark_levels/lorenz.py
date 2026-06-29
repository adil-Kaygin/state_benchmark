from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .base import BenchmarkLevel, BaseSimulator, FilterModel
from ._numba_dynamics import (
    build_lorenz_numba_dynamics,
    build_lorenz_fea_numba_dynamics,
)


# The true Lorenz-63 attractor lives within roughly [-20,20]x[-25,25]x[0,50].
# A filter estimate that diverges can drive the chaotic RK4 step to overflow to
# inf/NaN, which then poisons the rest of the run. Clip the *filter's* state to
# a generous multiple of the attractor's extent (~20-50x) so the dynamics stay
# finite without distorting behaviour near the attractor. The data-generating
# simulator is never clipped -- ground truth is untouched.
_STATE_BOUND = 1.0e3


def _lorenz_deriv(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    xv, y, z = state
    return np.array([
        sigma * (y - xv),
        xv * (rho - z) - y,
        xv * y - beta * z,
    ])


def _lorenz_jac(state: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    """Jacobian of the continuous Lorenz vector field at `state`."""
    xv, y, z = state
    return np.array([
        [-sigma, sigma, 0.0],
        [rho - z, -1.0, -xv],
        [y, xv, -beta],
    ])


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
        return _lorenz_deriv(state, self._sigma, self._rho, self._beta)

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


class _BaseLorenzBenchmark(BenchmarkLevel):
    """Shared Lorenz-63 data generation and process model.

    The only difference between the standard `LorenzBenchmark` and the
    `LorenzFEABenchmark` baseline is the *Jacobian* handed to EKF/KF (see
    `get_filter_model`); the simulator, dynamics `f`/`h`, and observation model
    are identical. Subclasses override `name` and `get_filter_model` only.
    """

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
    def description(self) -> str:
        return "Lorenz-63 chaotic system state estimation benchmark."

    @property
    def state_dimension(self) -> int:
        return 3

    @property
    def observation_dimension(self) -> int:
        return 2

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("x", "y", "z")

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

    def _f_h(self):
        sigma = self._sigma
        rho = self._rho
        beta = self._beta
        dt = self._dt

        def f(x: np.ndarray, t: float = 0.0) -> np.ndarray:
            x = np.clip(x, -_STATE_BOUND, _STATE_BOUND)
            k1 = _lorenz_deriv(x, sigma, rho, beta)
            k2 = _lorenz_deriv(x + 0.5 * dt * k1, sigma, rho, beta)
            k3 = _lorenz_deriv(x + 0.5 * dt * k2, sigma, rho, beta)
            k4 = _lorenz_deriv(x + dt * k3, sigma, rho, beta)
            return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        def h(x: np.ndarray, t: float = 0.0) -> np.ndarray:
            return np.array([x[0], x[1]])

        return f, h

    def _H_jac(self):
        def H_jac(x: np.ndarray) -> np.ndarray:
            return np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ])
        return H_jac


class LorenzBenchmark(_BaseLorenzBenchmark):
    """Standard Lorenz-63 level.

    The EKF/KF Jacobian `F` is the **exact Jacobian of the 4-stage RK4 map**
    used by `f` (chain rule through the four stages), so the covariance is
    propagated with the same O(dt^4) accuracy as the mean. This is the
    mathematically consistent linearization. For the lower-order
    forward-Euler-approximation Jacobian, use `LorenzFEABenchmark`.
    """

    @property
    def name(self) -> str:
        return "lorenz"

    def get_filter_model(self) -> FilterModel:
        sigma = self._sigma
        rho = self._rho
        beta = self._beta
        dt = self._dt

        f, h = self._f_h()
        H_jac = self._H_jac()

        def F_jac(x: np.ndarray) -> np.ndarray:
            """Exact Jacobian of the RK4 step x -> x + dt/6 (k1+2k2+2k3+k4).

            Chain rule through the four stages: with g the Lorenz vector field
            and J = dg/dx, each stage k_i = g(arg_i) has sensitivity
            dk_i = J(arg_i) @ d(arg_i)/dx, accumulated through the dependency
            arg_2 = x + dt/2 k1, arg_3 = x + dt/2 k2, arg_4 = x + dt k3.
            """
            x = np.clip(x, -_STATE_BOUND, _STATE_BOUND)
            I = np.eye(3)

            k1 = _lorenz_deriv(x, sigma, rho, beta)
            a2 = x + 0.5 * dt * k1
            k2 = _lorenz_deriv(a2, sigma, rho, beta)
            a3 = x + 0.5 * dt * k2
            k3 = _lorenz_deriv(a3, sigma, rho, beta)
            a4 = x + dt * k3

            dk1 = _lorenz_jac(x, sigma, rho, beta)
            dk2 = _lorenz_jac(a2, sigma, rho, beta) @ (I + 0.5 * dt * dk1)
            dk3 = _lorenz_jac(a3, sigma, rho, beta) @ (I + 0.5 * dt * dk2)
            dk4 = _lorenz_jac(a4, sigma, rho, beta) @ (I + dt * dk3)

            return I + (dt / 6.0) * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4)

        return FilterModel(
            f=f, h=h, F=F_jac, H=H_jac,
            Q=self._Q.copy(), R=self._R.copy(),
            x0_mean=np.array([0.0, 0.0, 25.0]), x0_cov=np.eye(3),
            numba=build_lorenz_numba_dynamics(sigma, rho, beta, dt),
        )


class LorenzFEABenchmark(_BaseLorenzBenchmark):
    """Forward-Euler-Approximation (FEA) baseline Lorenz-63 level.

    Identical to `LorenzBenchmark` except the EKF/KF Jacobian is the
    first-order forward-Euler linearization `F = I + dt*J(x)` of the flow,
    *not* the Jacobian of the RK4 map. Because `f` is RK4 (O(dt^4)) while this
    `F` is O(dt), the mean and covariance are propagated at inconsistent orders
    -- this level is retained only as a baseline to quantify the cost of that
    inconsistency against the standard `LorenzBenchmark`.
    """

    @property
    def name(self) -> str:
        return "lorenz_fea"

    @property
    def description(self) -> str:
        return (
            "Lorenz-63 chaotic system, forward-Euler-approximation (FEA) "
            "Jacobian baseline (first-order I + dt*J linearization)."
        )

    def get_filter_model(self) -> FilterModel:
        sigma = self._sigma
        rho = self._rho
        beta = self._beta
        dt = self._dt

        f, h = self._f_h()
        H_jac = self._H_jac()

        def F_jac(x: np.ndarray) -> np.ndarray:
            x = np.clip(x, -_STATE_BOUND, _STATE_BOUND)
            return np.eye(3) + dt * _lorenz_jac(x, sigma, rho, beta)

        return FilterModel(
            f=f, h=h, F=F_jac, H=H_jac,
            Q=self._Q.copy(), R=self._R.copy(),
            x0_mean=np.array([0.0, 0.0, 25.0]), x0_cov=np.eye(3),
            numba=build_lorenz_fea_numba_dynamics(sigma, rho, beta, dt),
        )
