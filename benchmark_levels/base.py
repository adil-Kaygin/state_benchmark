from __future__ import annotations  
  
from abc import ABC, abstractmethod  
from dataclasses import dataclass  
from pathlib import Path  
from typing import Callable, Optional  
  
import numpy as np


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
  
    @abstractmethod  
    def generate_dataset(self, output_dir: Path) -> None:  
        pass  
  
    @abstractmethod  
    def get_filter_model(self) -> FilterModel:  
        pass
