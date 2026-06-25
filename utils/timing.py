from __future__ import annotations  
  
import time  
from contextlib import contextmanager  
from typing import Generator  
  
  
@contextmanager  
def timed(label: str = "") -> Generator[dict, None, None]:  
    """Context manager that stores elapsed_seconds in yielded dict."""  
    result: dict = {}  
    start = time.perf_counter()  
    try:  
        yield result  
    finally:  
        elapsed = time.perf_counter() - start  
        result["elapsed_seconds"] = elapsed  
        if label:  
            print(f"[timing] {label}: {elapsed:.4f}s")  
  
  
def elapsed_ms(start: float) -> float:  
    """Return milliseconds elapsed since `start` (from time.perf_counter)."""  
    return (time.perf_counter() - start) * 1000.0
