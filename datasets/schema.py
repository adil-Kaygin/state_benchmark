from __future__ import annotations  
  
from dataclasses import dataclass  
from typing import Optional, TYPE_CHECKING  
  
if TYPE_CHECKING:  
    import torch  
  
  
@dataclass  
class DatasetMetadata:  
    benchmark_name: str  
    state_dimension: int  
    observation_dimension: int  
    trajectory_length: int  
    num_trajectories: int  
    random_seed: int  
    generation_time: str  
  
  
@dataclass  
class TrajectoryDataset:  
    states: torch.Tensor        # [N, T, nx]  
    observations: torch.Tensor  # [N, T, ny]  
    timestamps: torch.Tensor    # [T]  
    metadata: Optional[DatasetMetadata] = None
