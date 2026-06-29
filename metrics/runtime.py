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
    # Fail fast: a non-positive step count is an undefined latency, not an
    # "infinitely fast" 0.0. Returning 0.0 here previously made an undefined
    # case read as the best possible result.
    if num_steps <= 0:
        raise ValueError(
            f"num_steps must be positive to compute per-step runtime; got {num_steps}."
        )
    return (total_seconds / num_steps) * 1000.0
