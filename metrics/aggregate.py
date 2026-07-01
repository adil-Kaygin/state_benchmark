from __future__ import annotations

"""
Single-test-set statistics for the benchmark.

Every test trajectory is an independent realization of the system: it has its
own (uniformly sampled) initial state and its own process/observation noise.
The RMSE of an estimator on one trajectory is therefore one independent sample,
and the N trajectories of a single test set give an N-sample estimate of the
estimator's error distribution.

That means we can report a proper mean +/- std and a 95% confidence interval
from ONE sufficiently large test set -- there is no need to regenerate the whole
dataset and refit the (expensive) learned models over many random seeds. Using
e.g. 7500 test trajectories once is both cheaper and statistically cleaner than
1500 trajectories x 5 seed-realizations: the trajectory count IS the sample size.

These helpers turn the per-trajectory metric arrays
(metrics.rmse.compute_rmse_per_trajectory_per_dim) into
{mean, std, ci95, n} summaries.
"""

import math
from typing import Dict, Sequence, Tuple

import numpy as np


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and (sample) standard deviation of a sequence. std uses ddof=1 when
    there are >= 2 samples (an unbiased estimate of variability); for a single
    sample std is 0.0. Fails fast on an empty sequence."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("mean_std requires at least one value.")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return mean, std


def ci95_halfwidth(std: float, n: int) -> float:
    """Half-width of a normal-approximation 95% confidence interval for the mean
    given the sample std and count: 1.96 * std / sqrt(n). 0.0 for n < 2.

    With N in the thousands of test trajectories the normal approximation is
    well justified (CLT); the half-width shrinks as 1/sqrt(N), which is exactly
    why a single large test set yields a tight, reportable interval.
    """
    if n < 2:
        return 0.0
    return 1.96 * std / math.sqrt(n)


def summarize_samples(values: Sequence[float]) -> Dict[str, float]:
    """Summarize a 1-D sample (e.g. per-trajectory RMSE for one state variable)
    into {"mean", "std", "ci95", "n"}."""
    arr = np.asarray(list(values), dtype=np.float64)
    n = int(arr.size)
    mean, std = mean_std(arr)
    return {"mean": mean, "std": std, "ci95": ci95_halfwidth(std, n), "n": n}


def aggregate_rmse_per_dim_over_trajectories(
    rmse_per_traj_per_dim: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-trajectory RMSE arrays into a mean +/- std / 95% CI per
    named state variable, over the test trajectories.

    Parameters
    ----------
    rmse_per_traj_per_dim : {state_var: np.ndarray[N]} -- the output of
        metrics.rmse.compute_rmse_per_trajectory_per_dim (one RMSE per test
        trajectory for each named variable).

    Returns
    -------
    {state_var: {"mean": ..., "std": ..., "ci95": ..., "n": ...}}, where n is the
    number of test trajectories.

    Raises
    ------
    ValueError if the mapping is empty or any variable's array is empty.
    """
    if not rmse_per_traj_per_dim:
        raise ValueError(
            "aggregate_rmse_per_dim_over_trajectories requires at least one variable."
        )
    out: Dict[str, Dict[str, float]] = {}
    for var, arr in rmse_per_traj_per_dim.items():
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0:
            raise ValueError(f"variable {var!r} has no per-trajectory RMSE samples.")
        out[var] = summarize_samples(arr)
    return out


def aggregate_uncertainty_over_trajectories(
    nees_per_traj: Sequence[float],
    nll_per_traj: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-trajectory NEES / NLL into mean +/- std / 95% CI over the
    test trajectories (Issue 7), the SAME single-test-set treatment as RMSE.

    Parameters
    ----------
    nees_per_traj : [N] per-trajectory mean NEES (metrics.uncertainty.
        compute_nees_per_trajectory).
    nll_per_traj  : [N] per-trajectory mean NLL (compute_nll_per_trajectory).

    Returns
    -------
    {"nees": {"mean","std","ci95","n"}, "nll": {...}}.
    """
    nees = np.asarray(list(nees_per_traj), dtype=np.float64)
    nll = np.asarray(list(nll_per_traj), dtype=np.float64)
    if nees.size == 0 or nll.size == 0:
        raise ValueError("NEES/NLL per-trajectory arrays must be non-empty.")
    return {"nees": summarize_samples(nees), "nll": summarize_samples(nll)}
