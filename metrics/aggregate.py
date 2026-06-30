from __future__ import annotations

"""
Monte-Carlo aggregation across independent dataset realizations (seeds).

A single dataset realization -- however many trajectories it holds -- is a
high-variance point estimate: the process/observation noise and initial states
are random, and chaotic levels (Lorenz) amplify minute differences. Comparing
two estimators on ONE stochastic run is methodologically flawed because the
delta may fall within the noise margin (see issue Single-Run_Methodology_Flaw).

The fix is to run the full pipeline over N independent base seeds and report the
mean +/- std (or a 95% CI) of every metric. These helpers do the aggregation;
`experiments.runner.MonteCarloRunner` drives the seed loop.
"""

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and (sample) standard deviation of a sequence. std uses ddof=1 when
    there are >= 2 samples (an unbiased estimate of run-to-run variability); for
    a single run std is 0.0. Fails fast on an empty sequence."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("mean_std requires at least one value.")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return mean, std


def ci95_halfwidth(std: float, n: int) -> float:
    """Half-width of a normal-approximation 95% confidence interval for the mean
    given the sample std and count: 1.96 * std / sqrt(n). 0.0 for n < 2."""
    if n < 2:
        return 0.0
    return 1.96 * std / math.sqrt(n)


def aggregate_rmse_per_dim(
    rmse_per_dim_runs: List[Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Aggregate per-variable RMSE across runs.

    Parameters
    ----------
    rmse_per_dim_runs : list of {state_var: rmse}, one dict per seed/run (each is
        the output of metrics.rmse.compute_rmse_per_dim).

    Returns
    -------
    {state_var: {"mean": ..., "std": ..., "ci95": ..., "n": ...}}.

    Raises
    ------
    ValueError if the run list is empty or the runs do not share the same set of
        state variables (fail fast: a missing dimension would bias the mean).
    """
    if not rmse_per_dim_runs:
        raise ValueError("aggregate_rmse_per_dim requires at least one run.")

    keys = set(rmse_per_dim_runs[0].keys())
    for i, run in enumerate(rmse_per_dim_runs):
        if set(run.keys()) != keys:
            raise ValueError(
                f"run {i} has state variables {sorted(run.keys())}, expected "
                f"{sorted(keys)}; all runs must share the same variables."
            )

    n = len(rmse_per_dim_runs)
    out: Dict[str, Dict[str, float]] = {}
    for var in rmse_per_dim_runs[0].keys():
        mean, std = mean_std([run[var] for run in rmse_per_dim_runs])
        out[var] = {"mean": mean, "std": std, "ci95": ci95_halfwidth(std, n), "n": n}
    return out


def aggregate_scalar(values: Sequence[float]) -> Dict[str, float]:
    """Aggregate a scalar metric (e.g. runtime_per_step_ms / latency) across runs
    into {"mean", "std", "ci95", "n"}."""
    n = len(list(values))
    mean, std = mean_std(values)
    return {"mean": mean, "std": std, "ci95": ci95_halfwidth(std, n), "n": n}
