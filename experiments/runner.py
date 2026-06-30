from __future__ import annotations  
  
import json
import time
import uuid
from pathlib import Path

import numpy as np

from typing import Callable, List, Sequence, Tuple

from .config import ExperimentConfig
from .result import ExperimentResult, MonteCarloResult
from estimators.base import BaseEstimator
from datasets.schema import TrajectoryDataset
from metrics.rmse import compute_rmse_per_dim
from metrics.runtime import runtime_per_step_ms as _runtime_per_step_ms
from metrics.aggregate import aggregate_rmse_per_dim, aggregate_scalar
from storage.repository import ExperimentRepository

  
def _to_numpy(arr) -> np.ndarray:  
    if isinstance(arr, np.ndarray):  
        return arr  
    return np.asarray(arr)  
  
  
class ExperimentRunner:

    def __init__(
        self,
        repository: ExperimentRepository,
        artifacts_dir: Path,
    ) -> None:
        self._repository = repository
        self._artifacts_dir = artifacts_dir
  
    def run(  
        self,  
        estimator: BaseEstimator,  
        train_dataset: TrajectoryDataset,  
        val_dataset: TrajectoryDataset,  
        test_dataset: TrajectoryDataset,  
        config: ExperimentConfig,  
    ) -> ExperimentResult:  
        experiment_id = str(uuid.uuid4())  
  
        self._repository.create_experiment(  
            experiment_id=experiment_id,  
            benchmark_name=config.benchmark_name,  
            estimator_name=config.estimator_name,  
            random_seed=config.random_seed,  
            status="running",  
        )  
  
        try:  
            estimator.fit(train_dataset, val_dataset)  
  
            t0 = time.perf_counter()  
            estimates = estimator.estimate(test_dataset)  
            runtime_seconds = time.perf_counter() - t0  
  
            N = int(_to_numpy(test_dataset.states).shape[0])
            T = int(_to_numpy(test_dataset.timestamps).shape[0])
            runtime_per_step_ms = _runtime_per_step_ms(runtime_seconds, N * T)

            estimates_np = _to_numpy(estimates)
            targets_np = _to_numpy(test_dataset.states)
            # RMSE per named physical state variable -- the pooled scalar RMSE
            # has been removed. Memory tracking has likewise been removed (see
            # metrics/memory.py); it is no longer recorded.
            rmse_per_dim = compute_rmse_per_dim(
                estimates_np, targets_np, config.state_names
            )

            result = ExperimentResult(
                experiment_id=experiment_id,
                benchmark_name=config.benchmark_name,
                estimator_name=config.estimator_name,
                rmse_per_dim=rmse_per_dim,
                runtime_seconds=runtime_seconds,
                runtime_per_step_ms=runtime_per_step_ms,
                random_seed=config.random_seed,
            )

            self._repository.save_metrics(
                experiment_id=experiment_id,
                rmse_per_dim_json=json.dumps(rmse_per_dim),
                runtime_seconds=runtime_seconds,
                runtime_per_step_ms=runtime_per_step_ms,
            )
  
            if config.save_model:  
                model_path = self._artifacts_dir / experiment_id / "model.json"  
                estimator.save(model_path)  
                self._repository.save_artifact(  
                    experiment_id=experiment_id,  
                    model_path=str(model_path),  
                    figure_path=None,  
                )  
  
            self._repository.update_experiment_status(experiment_id, "completed")

        except Exception:
            self._repository.update_experiment_status(experiment_id, "failed")
            raise

        return result


# A pipeline factory: given a base seed, build everything needed for one
# independent run -- a FRESH estimator, fresh train/val/test datasets generated
# at that seed, and the matching config (config.random_seed must equal `seed`).
# Returns (estimator, train, val, test, config).
PipelineFactory = Callable[
    [int],
    Tuple[BaseEstimator, TrajectoryDataset, TrajectoryDataset, TrajectoryDataset, ExperimentConfig],
]


class MonteCarloRunner:
    """Drives the full pipeline across N independent dataset realizations and
    aggregates the metrics into mean +/- std (+ 95% CI), the methodologically
    sound way to compare estimators on a stochastic / chaotic benchmark
    (see issue Single-Run_Methodology_Flaw).

    For each base seed it calls `build_pipeline(seed)` to regenerate a fresh
    dataset (train/val/test), a fresh estimator, and a config whose random_seed
    is that seed, then runs the existing single-run `ExperimentRunner.run`. Each
    seed is persisted as its own experiment row (unique experiment_id, its own
    random_seed) by the underlying runner, so the SQLite store already tracks the
    seed per run -- this class only adds the seed loop and the aggregation.
    """

    def __init__(self, runner: ExperimentRunner) -> None:
        self._runner = runner

    def run(
        self,
        seeds: Sequence[int],
        build_pipeline: PipelineFactory,
        verbose: bool = True,
    ) -> MonteCarloResult:
        seeds = list(seeds)
        if not seeds:
            raise ValueError("MonteCarloRunner.run requires a non-empty list of seeds.")

        per_seed: List[ExperimentResult] = []
        benchmark_name = None
        estimator_name = None
        for seed in seeds:
            estimator, train_ds, val_ds, test_ds, config = build_pipeline(seed)
            if config.random_seed != seed:
                raise ValueError(
                    f"build_pipeline returned config.random_seed={config.random_seed} "
                    f"for requested seed {seed}; they must match so each run is "
                    "tagged with its own dataset seed."
                )
            if verbose:
                print(
                    f"[montecarlo] {config.benchmark_name}/{config.estimator_name} "
                    f"seed={seed} ({len(per_seed) + 1}/{len(seeds)})"
                )
            result = self._runner.run(estimator, train_ds, val_ds, test_ds, config)
            per_seed.append(result)
            benchmark_name = config.benchmark_name
            estimator_name = config.estimator_name

        rmse_agg = aggregate_rmse_per_dim([r.rmse_per_dim for r in per_seed])
        runtime_agg = aggregate_scalar([r.runtime_per_step_ms for r in per_seed])

        return MonteCarloResult(
            benchmark_name=benchmark_name,
            estimator_name=estimator_name,
            seeds=seeds,
            rmse_per_dim=rmse_agg,
            runtime_per_step_ms=runtime_agg,
            per_seed_results=per_seed,
        )
