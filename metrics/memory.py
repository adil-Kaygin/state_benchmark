from __future__ import annotations  
  
import os  
from typing import Optional  
  
  
def measure_memory() -> Optional[float]:  
    """  
    Return current process RSS memory in megabytes.  
    Returns None when psutil is not installed.  
    """  
    try:  
        import psutil  
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)  
    except ImportError:  
        return None
