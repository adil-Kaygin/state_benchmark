from __future__ import annotations  
  
from dataclasses import dataclass  
  
  
@dataclass  
class ExperimentConfig:  
    benchmark_name: str  
    estimator_name: str  
    random_seed: int  
    device: str  
    save_model: bool = True
