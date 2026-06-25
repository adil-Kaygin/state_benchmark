from __future__ import annotations  
  
from pathlib import Path  
from typing import TYPE_CHECKING  
  
from ..base import BaseEstimator  
  
if TYPE_CHECKING:  
    from datasets.schema import TrajectoryDataset  
  
  
class TransformerEstimator(BaseEstimator):  
    """Stub. Deferred to a future milestone."""  
  
    @property  
    def estimator_name(self) -> str:  
        return "transformer"  
  
    @property  
    def estimator_type(self) -> str:  
        return "neural"  
  
    def fit(self, train_dataset: TrajectoryDataset, val_dataset: TrajectoryDataset) -> None:  
        raise NotImplementedError("TransformerEstimator is deferred to a future milestone.")  
  
    def estimate(self, dataset: TrajectoryDataset):  
        raise NotImplementedError("TransformerEstimator is deferred to a future milestone.")  
  
    def save(self, path: Path) -> None:  
        raise NotImplementedError("TransformerEstimator is deferred to a future milestone.")  
  
    @classmethod  
    def load(cls, path: Path) -> TransformerEstimator:  
        raise NotImplementedError("TransformerEstimator is deferred to a future milestone.")
