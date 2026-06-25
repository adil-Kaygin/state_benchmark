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


def compute_rmse_per_dim(estimates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute RMSE per state dimension, pooling over trajectories and timesteps.

    Parameters
    ----------
    estimates : np.ndarray, shape [N, T, nx]
    targets   : np.ndarray, shape [N, T, nx]

    Returns
    -------
    np.ndarray, shape [nx]

    Use this instead of compute_rmse when comparing/aggregating across
    benchmarks whose state dimensions have different physical units (e.g.
    position vs. velocity, or RMSE across linear/pendulum/lorenz) -- pooling
    all dimensions into one scalar (compute_rmse) mixes those units.
    """
    return np.sqrt(np.mean((estimates - targets) ** 2, axis=(0, 1)))
