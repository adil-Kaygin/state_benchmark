# metrics

Scalar reductions over estimator output. No plotting, no data generation —
consumes already-computed `estimates`/`targets` arrays from `estimators/` and
`benchmark_levels/`.

```
rmse.py     : compute_rmse(estimates, targets) -> float
              sqrt(mean((x̂ - x)²))  over ALL of [N, T, nx] at once  (one scalar)
              compute_rmse_per_dim(estimates, targets) -> np.ndarray [nx]
              sqrt(mean((x̂ - x)², axis=(0, 1)))   per state dimension
              compute_rmse_per_timestep(estimates, targets) -> np.ndarray [T]
              sqrt(mean((x̂ - x)², axis=(0, 2)))   per timestep
runtime.py  : timer() context manager -> {"elapsed_seconds": float}
              runtime_per_step_ms(total_seconds, num_steps) -> float
              = (total_seconds / num_steps) * 1000   (0.0 if num_steps <= 0)
latency.py  : latency_ms_per_step(total_seconds, num_trajectories, trajectory_length)
              = runtime_per_step_ms(total_seconds, num_trajectories * trajectory_length)
memory.py   : measure_memory() -> process RSS in megabytes (psutil), or None if absent
```

## RMSE variants

- `compute_rmse` pools all dimensions and timesteps into one scalar — used as the
  headline ranking number within a benchmark.
- `compute_rmse_per_dim` reports error per state dimension (e.g. Lorenz `z`, which
  is unobserved — `ny=2 < nx=3`).
- `compute_rmse_per_timestep` reports how error evolves along a trajectory (e.g.
  filter convergence/divergence, plotted by `visualization.plot_rmse_per_timestep`).

The pooled `compute_rmse` scalar mixes dimensions and timesteps; see
[Critique.md](Critique.md) for the validity caveats (different physical scales
within a benchmark, and why raw `compute_rmse` is not comparable across
benchmarks).

## Latency / runtime

`runtime_per_step_ms` and `latency_ms_per_step` measure the wall-clock cost of
`estimator.estimate()` divided by the number of steps (`N*T`).
`experiments/runner.py` imports `metrics.runtime.runtime_per_step_ms` rather than
inlining the formula. Training (`fit()`) and numba JIT warm-up are **not** in the
timed window — see [Critique.md](Critique.md) for the fairness discussion of what
this measures and what it leaves out.

## Memory

`measure_memory()` returns the whole process's resident set size (RSS) in MB via
`psutil`, or `None` if `psutil` is not installed. It is a process-level number,
not a per-estimator allocation delta — see [Critique.md](Critique.md).

## Extending with a new metric

- Take already-computed arrays/scalars as input — never recompute anything
  `estimators/` or `benchmark_levels/` already produced.
- Return a plain `float` (or `Optional[float]` if it can be unavailable, like
  `measure_memory`), matching the existing functions' signatures so
  `experiments/runner.py` and `visualization/` can consume it uniformly.
