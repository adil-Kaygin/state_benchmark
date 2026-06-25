from __future__ import annotations  
  
from dataclasses import dataclass  
from typing import Optional  
  
  
@dataclass  
class ExperimentResult:  
    experiment_id: str  
    benchmark_name: str  
    estimator_name: str  
    rmse: float  
    runtime_seconds: float  
    runtime_per_step_ms: float  
    memory_mb: Optional[float]
