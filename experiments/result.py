from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


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
    # The base seed of the dataset realization this result came from. Lets a
    # Monte-Carlo sweep tie each single-run result back to its seed.
    random_seed: int = -1


@dataclass
class MonteCarloResult:
    """Aggregate of a benchmark/estimator across N independent dataset
    realizations (seeds). Reports mean +/- std (and a 95% CI half-width) for
    every metric -- the methodologically-sound way to compare estimators on a
    stochastic, possibly-chaotic benchmark (see Single-Run_Methodology_Flaw).
    """

    benchmark_name: str
    estimator_name: str
    seeds: List[int]
    # {state_var: {"mean", "std", "ci95", "n"}}
    rmse_per_dim: Dict[str, Dict[str, float]]
    # {"mean", "std", "ci95", "n"}
    runtime_per_step_ms: Dict[str, float]
    # The individual per-seed results, for drill-down / re-aggregation.
    per_seed_results: List[ExperimentResult] = field(default_factory=list)
