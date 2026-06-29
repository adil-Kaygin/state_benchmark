from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExperimentResult:
    experiment_id: str
    benchmark_name: str
    estimator_name: str
    # RMSE per named state variable, e.g. {"x": ..., "y": ..., "z": ...}.
    # The single pooled scalar RMSE has been removed as scientifically unsound.
    rmse_per_dim: Dict[str, float]
    runtime_seconds: float
    runtime_per_step_ms: float
