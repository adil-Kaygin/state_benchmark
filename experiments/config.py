from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class ExperimentConfig:
    benchmark_name: str
    estimator_name: str
    random_seed: int
    device: str
    # Physical names of each state dimension (BenchmarkLevel.state_names),
    # required to report RMSE per named variable -- there is no pooled scalar.
    state_names: Sequence[str]
    save_model: bool = True
