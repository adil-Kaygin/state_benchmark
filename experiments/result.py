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
    # For the mean +/- std / 95% CI over the test trajectories, use
    # metrics.aggregate.aggregate_rmse_per_dim_over_trajectories on the
    # per-trajectory RMSE arrays of a single (large) test set.
    rmse_per_dim: Dict[str, float]
    runtime_seconds: float
    runtime_per_step_ms: float
    random_seed: int = -1
