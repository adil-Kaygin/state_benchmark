from __future__ import annotations  
  
import json
import time
import uuid
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .result import ExperimentResult
from estimators.base import BaseEstimator
from datasets.schema import TrajectoryDataset
from metrics.rmse import compute_rmse_per_dim
from metrics.runtime import runtime_per_step_ms as _runtime_per_step_ms
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

