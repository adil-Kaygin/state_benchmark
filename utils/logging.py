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
