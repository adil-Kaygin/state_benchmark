# metrics

Scalar/dict reductions over estimator output. No plotting, no data generation —
consumes already-computed `estimates`/`targets`/`covariances` arrays from
`estimators/` and `benchmark_levels/`.

```
rmse.py        : compute_rmse_per_dim(estimates, targets, state_names) -> dict {state_var: rmse}
                 sqrt(mean((x̂ - x)², axis=(0, 1)))   per named physical state variable
                 compute_rmse_per_timestep(estimates, targets) -> np.ndarray [T]
                 sqrt(mean((x̂ - x)², axis=(0, 2)))   per timestep
uncertainty.py : compute_nees(estimates, targets, covariances) -> float
                 mean (x-x̂)ᵀP⁻¹(x-x̂); consistent filter -> E[NEES] ≈ nx
                 compute_nees_chi2_bounds(nx, num_samples, confidence) -> (lower, upper)
                 compute_nll(estimates, targets, covariances) -> float
                 mean Gaussian NLL of truth under N(x̂, P); lower is better
runtime.py     : timer() context manager -> {"elapsed_seconds": float}
                 runtime_per_step_ms(total_seconds, num_steps) -> float
                 = (total_seconds / num_steps) * 1000   (raises if num_steps <= 0)
latency.py     : latency_ms_per_step(total_seconds, num_trajectories, trajectory_length)
                 = runtime_per_step_ms(total_seconds, num_trajectories * trajectory_length)
memory.py      : measure_memory() -> raises NotImplementedError (disabled, see below)
```

## RMSE: no pooled scalar

The single-scalar "pooled" RMSE (`sqrt(mean over all of [N,T,nx] at once)`) has
been **deleted**. Pooling state dimensions of different physical units/scales
into one number is scale-weighted toward the largest-magnitude dimension and is
not a balanced accuracy measure, within a benchmark or across benchmarks. There
is no scalar fallback:

- `compute_rmse_per_dim` is the primary reported metric: a `dict` keyed by the
  physical variable name (`BenchmarkLevel.state_names`, e.g. `x`/`y`/`z` for
  Lorenz, `theta`/`omega` for the pendulum). Raises `ValueError` on a shape
  mismatch or a `state_names` list whose length doesn't match `nx`.
- `compute_rmse_per_timestep` still pools across dimensions, but only along the
  time axis, for plotting how error evolves over a trajectory — a deliberate,
  documented convenience, not the headline number.

## Uncertainty: NEES / NLL

`uncertainty.py` scores the filters' reported **covariance**, which RMSE ignores
entirely. Both require the full posterior covariance `P` (`[N, T, nx, nx]`)
alongside the point estimate:

- `compute_nees` — Normalized Estimation Error Squared, the standard
  Bar-Shalom et al. consistency check. A consistent filter has mean NEES ≈ `nx`;
  `compute_nees_chi2_bounds` gives the chi-square acceptance interval (requires
  `scipy`).
- `compute_nll` — Gaussian negative log-likelihood of the truth under
  `N(x̂, P)`. A proper scoring rule: penalizes both point error and miscalibrated
  covariance, lower is better.

Both raise `ValueError` on shape mismatch, non-finite input, or a
non-positive-definite `P` — a covariance that can't be inverted/Cholesky'd is a
real defect in the filter, not something to silently skip.

## Latency / runtime

`runtime_per_step_ms` and `latency_ms_per_step` measure the wall-clock cost of
`estimator.estimate()` divided by the number of steps (`N*T`). `num_steps <= 0`
raises `ValueError` (an undefined latency, not `0.0`/"infinitely fast").
Training (`fit()`) and numba JIT warm-up are **not** in the timed window — see
[Critique.md](Critique.md) for the fairness discussion of what this measures.

## Memory: disabled

`measure_memory()` raises `NotImplementedError("Memory measurement is currently
unsupported.")`. The previous implementation returned the whole process's RSS
via `psutil`, which is a constant process baseline (numpy/torch/numba
caches + the full dataset + every prior estimator's leftovers) plus noise — not
a per-estimator footprint, and so meaningless for comparing estimators. Per the
"fail fast and loud" rule it raises rather than reporting that misleading number
(or silently returning `None`).

## Extending with a new metric

- Take already-computed arrays/scalars as input — never recompute anything
  `estimators/` or `benchmark_levels/` already produced.
- Never pool state dimensions of different physical units into a single scalar;
  report per named state variable (use `BenchmarkLevel.state_names`).
- Fail fast: raise on shape mismatch, non-finite input, or a configuration that
  can't be scored (e.g. a singular covariance) — never return a dummy `0.0`/`NaN`.
