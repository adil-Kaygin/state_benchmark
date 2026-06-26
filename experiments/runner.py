from __future__ import annotations  
  
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from .config import ExperimentConfig
from .result import ExperimentResult
from estimators.base import BaseEstimator
from datasets.schema import TrajectoryDataset
from metrics.rmse import compute_rmse
from metrics.memory import measure_memory
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
            rmse = compute_rmse(estimates=estimates_np, targets=targets_np)  
            memory_mb = measure_memory()  
  
            result = ExperimentResult(  
                experiment_id=experiment_id,  
                benchmark_name=config.benchmark_name,  
                estimator_name=config.estimator_name,  
                rmse=rmse,  
                runtime_seconds=runtime_seconds,  
                runtime_per_step_ms=runtime_per_step_ms,  
                memory_mb=memory_mb,  
            )  
  
            self._repository.save_metrics(  
                experiment_id=experiment_id,  
                rmse=rmse,  
                runtime_seconds=runtime_seconds,  
                runtime_per_step_ms=runtime_per_step_ms,  
                memory_mb=memory_mb,  
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
