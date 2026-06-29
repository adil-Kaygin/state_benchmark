from __future__ import annotations  
  
from abc import ABC, abstractmethod  
from dataclasses import dataclass  
from pathlib import Path  
from typing import Callable, Optional  
  
import numpy as np


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
class FilterModel:
    f: Callable
    h: Callable
    F: Optional[Callable]
    H: Optional[Callable]
    Q: np.ndarray
    R: np.ndarray
    x0_mean: Optional[np.ndarray] = None
    x0_cov: Optional[np.ndarray] = None
    # Optional @njit dynamics for the CPU-optimized filter kernels. None for
    # levels that don't provide them (filters then use the numpy fallback).
    numba: Optional[NumbaDynamics] = None
  
  
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
