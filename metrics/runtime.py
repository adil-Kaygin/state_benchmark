from __future__ import annotations  
  
import time  
from contextlib import contextmanager  
from typing import Generator  
  
  
@contextmanager  
def timer() -> Generator[dict, None, None]:  
    """Context manager that records elapsed_seconds in the yielded dict."""  
    result: dict = {}  
    start = time.perf_counter()  
    try:  
        yield result  
    finally:  
        result["elapsed_seconds"] = time.perf_counter() - start  
  
  
def runtime_per_step_ms(total_seconds: float, num_steps: int) -> float:  
    if num_steps <= 0:  
        return 0.0  
    return (total_seconds / num_steps) * 1000.0
