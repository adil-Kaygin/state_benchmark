from __future__ import annotations  
  
import datetime  
from typing import Any, Dict, List, Optional  
  
from .database import Database  
  
  
class ExperimentRepository:  
  
    def __init__(self, database: Database) -> None:  
        self._db = database  
  
    def create_experiment(  
        self,  
        experiment_id: str,  
        benchmark_name: str,  
        estimator_name: str,  
        random_seed: int,  
        status: str,  
    ) -> None:  
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()
        self._db.execute(  
            """  
            INSERT INTO experiments (id, timestamp, benchmark_name, estimator_name, random_seed, status)  
            VALUES (?, ?, ?, ?, ?, ?)  
            """,  
            (experiment_id, timestamp, benchmark_name, estimator_name, random_seed, status),  
        )  
        self._db.commit()  
  
    def update_experiment_status(self, experiment_id: str, status: str) -> None:  
        self._db.execute(  
            "UPDATE experiments SET status = ? WHERE id = ?",  
            (status, experiment_id),  
        )  
        self._db.commit()  
  
    def save_metrics(  
        self,  
        experiment_id: str,  
        rmse: float,  
        runtime_seconds: float,  
        runtime_per_step_ms: float,  
        memory_mb: Optional[float],  
    ) -> None:  
        self._db.execute(  
            """  
            INSERT INTO metrics (experiment_id, rmse, runtime_seconds, runtime_per_step_ms, memory_mb)  
            VALUES (?, ?, ?, ?, ?)  
            """,  
            (experiment_id, rmse, runtime_seconds, runtime_per_step_ms, memory_mb),  
        )  
        self._db.commit()  
  
    def save_artifact(  
        self,  
        experiment_id: str,  
        model_path: Optional[str],  
        figure_path: Optional[str],  
    ) -> None:  
        self._db.execute(  
            """  
            INSERT INTO artifacts (experiment_id, model_path, figure_path)  
            VALUES (?, ?, ?)  
            """,  
            (experiment_id, model_path, figure_path),  
        )  
        self._db.commit()  
  
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:  
        row = self._db.execute(  
            "SELECT * FROM experiments WHERE id = ?",  
            (experiment_id,),  
        ).fetchone()  
        return dict(row) if row else None  
  
    def get_metrics(self, experiment_id: str) -> Optional[Dict[str, Any]]:  
        row = self._db.execute(  
            "SELECT * FROM metrics WHERE experiment_id = ?",  
            (experiment_id,),  
        ).fetchone()  
        return dict(row) if row else None  
  
    def get_artifacts(self, experiment_id: str) -> Optional[Dict[str, Any]]:  
        row = self._db.execute(  
            "SELECT * FROM artifacts WHERE experiment_id = ?",  
            (experiment_id,),  
        ).fetchone()  
        return dict(row) if row else None  
  
    def list_experiments(self) -> List[Dict[str, Any]]:  
        rows = self._db.execute(  
            "SELECT * FROM experiments ORDER BY timestamp DESC"  
        ).fetchall()  
        return [dict(row) for row in rows]
