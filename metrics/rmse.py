from __future__ import annotations  
  
import numpy as np  
  
  
def compute_rmse(estimates: np.ndarray, targets: np.ndarray) -> float:  
    """  
    Compute mean RMSE across all trajectories and timesteps.  
  
    Parameters  
    ----------  
    estimates : np.ndarray, shape [N, T, nx]  
    targets   : np.ndarray, shape [N, T, nx]  
  
    Returns  
    -------  
    float  
    """  
    return float(np.sqrt(np.mean((estimates - targets) ** 2)))
