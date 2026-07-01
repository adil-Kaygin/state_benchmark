from __future__ import annotations  
  
from abc import ABC, abstractmethod  
from dataclasses import dataclass  
from pathlib import Path  
from typing import Callable, Optional  
  
import numpy as np


def split_counts(num_trajectories: int) -> dict[str, int]:
    """70/15/15 train/val/test split that conserves every trajectory.

    The naive ``int(n*0.7) / int(n*0.15) / int(n*0.15)`` silently drops 1-2
    trajectories when n is not divisible (e.g. n=1500 -> 1050+225+225=1500 ok,
    but n=2000 -> 1400+300+300=2000 ok; n=1499 -> 1049+224+224=1497, two lost).
    This assigns the remainder to train so the counts always sum to n exactly.
    """
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be positive; got {num_trajectories}.")
    n_val = int(num_trajectories * 0.15)
    n_test = int(num_trajectories * 0.15)
    n_train = num_trajectories - n_val - n_test
    return {"train": n_train, "val": n_val, "test": n_test}


def gaussian_noise(
    rng: np.random.Generator, cov: np.ndarray, batch_shape: tuple[int, ...]
) -> np.ndarray:
    """Draw zero-mean Gaussian noise of covariance `cov` for a whole batch in one
    vectorized call. Returns shape (*batch_shape, dim).

    Replaces the per-timestep ``rng.multivariate_normal(...)`` loop in dataset
    generation: numerically identical draws (same Cholesky-style construction)
    but one RNG call per split instead of n_traj * T calls. The process and
    observation noise stay Gaussian on purpose -- that is the model assumption
    the Kalman-family filters are derived under; only the *initial-condition*
    sampling is widened to uniform for training robustness.
    """
    dim = cov.shape[0]
    mean = np.zeros(dim)
    return rng.multivariate_normal(mean, cov, size=batch_shape)


@dataclass
class NumbaDynamics:
    """njit-compiled versions of a FilterModel's dynamics.

    f/h/F_jac/H_jac here must be `@njit` functions with the SAME signatures and
    math as the FilterModel's pure-Python f/h/F/H. They exist so the classical
    filters (KF/EKF/UKF) can run their inner recursion fully inside numba
    (see estimators/classical/_numba_kernels.py) instead of calling back into
    Python every timestep. The Python callables remain the source of truth and
    the numpy-fallback path; this is an optional accelerator only.

    Convention (so one njit kernel fits every level):
      f(x, t) -> [nx]   h(x, t) -> [ny]   F_jac(x) -> [nx, nx]   H_jac(x) -> [ny, nx]
    """
    f: Callable
    h: Callable
    F_jac: Callable
    H_jac: Callable


@dataclass
class TorchDynamics:
    """Batched, GPU-friendly torch versions of a FilterModel's f/h.

    f/h here take a batched state tensor [B, nx] (and a scalar timestep t) and
    return [B, nx] / [B, ny] entirely with torch tensor ops on the input's
    device -- no per-row Python loop, no NumPy round-trip. They exist so
    KalmanNet's predict step can run fully vectorized on the GPU during
    training/validation (the classical filters do not use these; they use the
    @njit NumbaDynamics on CPU).

    The math MUST match the FilterModel's NumPy f/h one-for-one so that the
    GPU-trained network and the CPU sequential estimate() see the same process
    model. Convention:
      f(x, t) -> [B, nx]   h(x, t) -> [B, ny]

    time_invariant (Issue 10): True iff f/h ignore the scalar timestep t (their
    output depends only on x). For such a level the teacher-forced precompute can
    flatten the [B, T] grid into one [B*T] batch and call f/h ONCE instead of
    looping T times -- exactly equal because the calls are independent and
    t-invariant. It MUST default to False (the safe per-step path): a level whose
    f actually uses t (e.g. nonlinear's cos(1.2*t)) would be silently corrupted by
    flattening, which feeds a single t to every row. Only set True when f/h
    provably drop t.
    """
    f: Callable
    h: Callable
    time_invariant: bool = False


@dataclass
class FilterModel:
    f: Callable
    h: Callable
    F: Optional[Callable]
    H: Optional[Callable]
    Q: np.ndarray
    R: np.ndarray
    x0_mean: Optional[np.ndarray] = None
    x0_cov: Optional[np.ndarray] = None
    # @njit dynamics for the CPU-optimized classical filter kernels. The
    # classical filters require this (there is no NumPy fallback); only levels
    # consumed solely by neural estimators may leave it None.
    numba: Optional[NumbaDynamics] = None
    # Optional batched-torch dynamics used by KalmanNet for vectorized GPU
    # training. None for levels that don't provide them (KalmanNet then cannot
    # train those levels on GPU and says so loudly).
    torch: Optional["TorchDynamics"] = None
    # Boolean [ny] mask marking which OBSERVATION components are angles that must
    # have their innovation y - h(x) wrapped to (-pi, pi] (Issues 5/6). None or
    # all-False means "no angular components" (every current scalar-obs level).
    # The classical EKF/UKF kernels and the neural innovation features consult
    # this so a bearing residual near the +/-pi branch cut is not ~2*pi wrong.
    angular_obs_mask: Optional[np.ndarray] = None
  
  
class BaseSimulator(ABC):  
  
    @abstractmethod  
    def step(  
        self,  
        state: np.ndarray,  
        control: Optional[np.ndarray],  
        dt: float,  
    ) -> np.ndarray:  
        pass  
  
    @abstractmethod  
    def observe(  
        self,  
        state: np.ndarray,  
    ) -> np.ndarray:  
        pass  
  
  
class BenchmarkLevel(ABC):  
  
    @property  
    @abstractmethod  
    def name(self) -> str:  
        pass  
  
    @property  
    @abstractmethod  
    def description(self) -> str:  
        pass  
  
    @property
    @abstractmethod
    def state_dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def observation_dimension(self) -> int:
        pass

    @property
    @abstractmethod
    def state_names(self) -> tuple[str, ...]:
        """Physical names of each state dimension, e.g. ('x', 'y', 'z') for
        Lorenz or ('theta', 'omega') for the pendulum. len == state_dimension.
        Used by metrics/visualization to label per-dimension RMSE with the real
        physical variable instead of a generic integer index."""
        pass
  
    @abstractmethod  
    def generate_dataset(self, output_dir: Path) -> None:  
        pass  
  
    @abstractmethod  
    def get_filter_model(self) -> FilterModel:  
        pass
