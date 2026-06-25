from __future__ import annotations  
  
import sqlite3  
from pathlib import Path  
from typing import Optional  
  
  
class Database:  
  
    def __init__(self, db_path: Path) -> None:  
        self._db_path = db_path  
        self._db_path.parent.mkdir(parents=True, exist_ok=True)  
        self._connection: Optional[sqlite3.Connection] = None  
  
    def connect(self) -> None:  
        self._connection = sqlite3.connect(self._db_path)  
        self._connection.row_factory = sqlite3.Row  
  
    def disconnect(self) -> None:  
        if self._connection is not None:  
            self._connection.close()  
            self._connection = None  
  
    def __enter__(self) -> Database:  
        self.connect()  
        return self  
  
    def __exit__(self, *args) -> None:  
        self.disconnect()  
  
    @property  
    def connection(self) -> sqlite3.Connection:  
        if self._connection is None:  
            raise RuntimeError("Database not connected. Call connect() first.")  
        return self._connection  
  
    def initialize_schema(self, schema_path: Path) -> None:  
        with open(schema_path, "r") as fh:  
            self.connection.executescript(fh.read())  
        self.connection.commit()  
  
    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:  
        return self.connection.execute(sql, params)  
  
    def commit(self) -> None:  
        self.connection.commit()
