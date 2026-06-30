from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# NOTE: The single-scalar "pooled" RMSE (sqrt(mean((x̂ - x)²)) over all of
# [N, T, nx] at once) has been DELETED. Pooling state dimensions of different
# physical units/scales into one number is scientifically unsound -- it is
# dominated by the largest-magnitude dimension and is not comparable within or
# across benchmarks. Per the "fail fast and loud" rule there is no scalar
# fallback: callers must report RMSE per named state variable.


def compute_rmse_per_dim(
    estimates: np.ndarray,
    targets: np.ndarray,
    state_names: Sequence[str],
) -> Dict[str, float]:
    """RMSE per state dimension, keyed by the physical variable name.

    Parameters
    ----------
    estimates : np.ndarray, shape [N, T, nx]
    targets   : np.ndarray, shape [N, T, nx]
    state_names : sequence of length nx, the physical name of each state
        dimension (e.g. ("x", "y", "z") for Lorenz, ("theta", "omega") for the
        pendulum). Obtained from BenchmarkLevel.state_names.

    Returns
    -------
    dict mapping state-variable name -> RMSE for that dimension.

    Raises
    ------
    ValueError
        If estimates/targets shapes mismatch, are not 3-D [N, T, nx], or if
        len(state_names) != nx. (Fail fast: a mismatched name list silently
        mislabelling dimensions is exactly the kind of error this guards.)
    """
    estimates = np.asarray(estimates)
    targets = np.asarray(targets)

    if estimates.shape != targets.shape:
        raise ValueError(
            f"estimates and targets must have the same shape; got "
            f"{estimates.shape} vs {targets.shape}."
        )
    if estimates.ndim != 3:
        raise ValueError(
            f"estimates/targets must be 3-D [N, T, nx]; got ndim={estimates.ndim} "
            f"with shape {estimates.shape}."
        )

    nx = estimates.shape[2]
    if len(state_names) != nx:
        raise ValueError(
            f"state_names has length {len(state_names)} but the state dimension "
            f"is {nx}; every dimension must have exactly one physical name."
        )

    per_dim = np.sqrt(np.mean((estimates - targets) ** 2, axis=(0, 1)))
    return {name: float(per_dim[i]) for i, name in enumerate(state_names)}


def compute_rmse_per_trajectory_per_dim(
    estimates: np.ndarray,
    targets: np.ndarray,
    state_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """One RMSE *per trajectory* per named state variable.

    Where ``compute_rmse_per_dim`` pools over all N trajectories into a single
    scalar per variable, this keeps the trajectory axis: for each named variable
    it returns an array of length N, the RMSE of that variable over the T
    timesteps of each trajectory.

    Each test trajectory is an independent realization (its own initial state and
    noise), so this N-vector is the sample used to report a mean +/- std / 95%
    confidence interval on a *single* test set -- the statistically sound
    replacement for re-running the whole pipeline over many random seeds. Feed the
    result to ``metrics.aggregate.aggregate_rmse_per_dim_over_trajectories``.

    Parameters
    ----------
    estimates   : [N, T, nx]
    targets     : [N, T, nx]
    state_names : length-nx physical names (BenchmarkLevel.state_names).

    Returns
    -------
    {state_var: np.ndarray of shape [N]} -- per-trajectory RMSE for that variable.

    Raises
    ------
    ValueError on shape mismatch, non-3-D input, or len(state_names) != nx.
    """
    estimates = np.asarray(estimates)
    targets = np.asarray(targets)

    if estimates.shape != targets.shape:
        raise ValueError(
            f"estimates and targets must have the same shape; got "
            f"{estimates.shape} vs {targets.shape}."
        )
    if estimates.ndim != 3:
        raise ValueError(
            f"estimates/targets must be 3-D [N, T, nx]; got ndim={estimates.ndim} "
            f"with shape {estimates.shape}."
        )

    nx = estimates.shape[2]
    if len(state_names) != nx:
        raise ValueError(
            f"state_names has length {len(state_names)} but the state dimension "
            f"is {nx}; every dimension must have exactly one physical name."
        )

    # RMSE over the time axis only -> [N, nx]
    per_traj = np.sqrt(np.mean((estimates - targets) ** 2, axis=1))
    return {name: per_traj[:, i] for i, name in enumerate(state_names)}


def compute_rmse_per_timestep(estimates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    RMSE per timestep, pooling over trajectories and state dimensions.

    Parameters
    ----------
    estimates : np.ndarray, shape [N, T, nx]
    targets   : np.ndarray, shape [N, T, nx]

    Returns
    -------
    np.ndarray, shape [T]

    Use this to visualize how estimation error evolves over a trajectory
    (e.g. filter convergence/divergence). Pooling across dimensions here is a
    deliberate convenience for a single time-axis curve; for a balanced
    accuracy measure use compute_rmse_per_dim (per named variable).

    Raises
    ------
    ValueError
        If estimates/targets shapes mismatch or are not 3-D [N, T, nx].
    """
    estimates = np.asarray(estimates)
    targets = np.asarray(targets)

    if estimates.shape != targets.shape:
        raise ValueError(
            f"estimates and targets must have the same shape; got "
            f"{estimates.shape} vs {targets.shape}."
        )
    if estimates.ndim != 3:
        raise ValueError(
            f"estimates/targets must be 3-D [N, T, nx]; got ndim={estimates.ndim} "
            f"with shape {estimates.shape}."
        )

    return np.sqrt(np.mean((estimates - targets) ** 2, axis=(0, 2)))
