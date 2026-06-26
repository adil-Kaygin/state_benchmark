from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.result import ExperimentResult
  
  
def get_logger(  
    name: str,  
    level: int = logging.INFO,  
    log_file: Optional[Path] = None,  
) -> logging.Logger:  
    logger = logging.getLogger(name)  
    logger.setLevel(level)  
  
    if not logger.handlers:  
        formatter = logging.Formatter(  
            "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",  
            datefmt="%Y-%m-%d %H:%M:%S",  
        )  
  
        ch = logging.StreamHandler(sys.stdout)  
        ch.setFormatter(formatter)  
        logger.addHandler(ch)  
  
        if log_file is not None:  
            log_file.parent.mkdir(parents=True, exist_ok=True)  
            fh = logging.FileHandler(log_file)  
            fh.setFormatter(formatter)  
            logger.addHandler(fh)  
  
    return logger


class CometExperimentLogger:
    """
    Push-only Comet reporter for already-computed ExperimentResults.

    Does not compute or own any metrics itself (rmse/runtime/memory are
    computed by ExperimentRunner / metrics/*); this class only forwards
    finished ExperimentResult rows to Comet. Disabled by default -- a
    runner with no logger behaves exactly as before.

    Pushes are batched into a single Comet Experiment (one run of the
    benchmark = one Comet experiment with a results table) rather than one
    Experiment per estimator/benchmark pair, since creating a Comet
    Experiment is the slow, network-bound step. Call `flush()` once after
    accumulating results with `log_result`/`log_results`.
    """

    def __init__(
        self,
        api_key: str,
        project_name: str,
        workspace: Optional[str] = None,
        experiment_name: str = "state_benchmark_run",
    ) -> None:
        self._api_key = api_key
        self._project_name = project_name
        self._workspace = workspace
        self._experiment_name = experiment_name
        self._pending: list["ExperimentResult"] = []
        self._pushed_ids: set[str] = set()

    def log_result(self, result: "ExperimentResult") -> None:
        """Queue one result for the next flush(); skips duplicate experiment_ids."""
        if result.experiment_id in self._pushed_ids:
            return
        self._pushed_ids.add(result.experiment_id)
        self._pending.append(result)

    def log_results(self, results: Iterable["ExperimentResult"]) -> None:
        for result in results:
            self.log_result(result)

    def flush(self):
        """Push all queued results to Comet in a single batched Experiment."""
        if not self._pending:
            return None

        from comet_ml import Experiment

        experiment = Experiment(
            api_key=self._api_key,
            project_name=self._project_name,
            workspace=self._workspace,
        )
        experiment.set_name(self._experiment_name)

        rows = [
            {
                "experiment_id": r.experiment_id,
                "benchmark_name": r.benchmark_name,
                "estimator_name": r.estimator_name,
                "rmse": r.rmse,
                "runtime_seconds": r.runtime_seconds,
                "runtime_per_step_ms": r.runtime_per_step_ms,
                "memory_mb": r.memory_mb,
            }
            for r in self._pending
        ]
        experiment.log_table("results.json", rows)

        self._pending.clear()
        experiment.end()
        return experiment
