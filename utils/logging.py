from __future__ import annotations  
  
import logging  
import sys  
from pathlib import Path  
from typing import Optional  
  
  
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
    """Thin wrapper: one Comet experiment per (benchmark_level, model) run."""

    def __init__(
        self,
        api_key: str,
        project_name: str,
        workspace: Optional[str] = None,
    ) -> None:
        self._api_key = api_key
        self._project_name = project_name
        self._workspace = workspace

    def start(self, benchmark_level: str, model_name: str):
        from comet_ml import Experiment

        experiment = Experiment(
            api_key=self._api_key,
            project_name=self._project_name,
            workspace=self._workspace,
        )
        experiment.set_name(f"{benchmark_level}_{model_name}")
        experiment.log_parameter("benchmark_level", benchmark_level)
        experiment.log_parameter("model_name", model_name)
        experiment.add_tag(benchmark_level)
        experiment.add_tag(model_name)
        return experiment
