from __future__ import annotations  
  
from abc import ABC, abstractmethod  
from pathlib import Path  
  
  
class BaseEstimator(ABC):  
  
    @property  
    @abstractmethod  
    def estimator_name(self) -> str:  
        pass  
  
    @property  
    @abstractmethod  
    def estimator_type(self) -> str:  
        pass  
  
    @abstractmethod  
    def fit(self, train_dataset, val_dataset) -> None:  
        pass  
  
    @abstractmethod  
    def estimate(self, dataset):  
        pass  
  
    @abstractmethod  
    def save(self, path: Path) -> None:  
        pass  
  
    @classmethod  
    @abstractmethod  
    def load(cls, path: Path) -> BaseEstimator:  
        pass
