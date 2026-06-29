# state_benchmark

State-space estimation benchmark: synthetic systems → noisy observations →
classical/neural filters → per-variable RMSE + uncertainty (NEES/NLL) + runtime
metrics → plots.

```
benchmark_levels/  generates (states, observations) per system, defines FilterModel
        │  x_t (latent, ground truth)   y_t (noisy sensor data, the only filter input)
        ▼
estimators/        consumes y_t, produces x̂_t   (never sees x_t)
        │
        ▼
metrics/           compute_rmse_per_dim(x̂_t, x_t, names), compute_nees/nll, runtime_per_step_ms
        │
        ▼
visualization/     plot_trajectory / plot_rmse_comparison_per_dim / plot_runtime_comparison
```

## Architectural rules

These are enforced in code, not just conventions:

- **Fail fast and loud.** No silent fallbacks, implicit type coercions, or dummy
  returns (no `0.0`/`NaN`/`None` for an undefined result). Invalid inputs,
  mismatched arrays, or a scientifically unsound configuration raise a
  descriptive `ValueError`/`RuntimeError`/`ImportError` immediately.
- **Strict Numba.** The classical filters (KF/EKF/UKF) run *exclusively* on the
  custom `@njit` kernels (plus the third-party `filterpy` reference filters).
  There is no pure-NumPy fallback; a missing numba raises `ImportError` at
  import time.
- **Linear KF only on linear models.** `KalmanFilterEstimator` asserts its model
  is linear (`f(x)=F@x`, `h(x)=H@x`) on every `estimate()` and refuses to run on
  a nonlinear system (it does not silently linearize at the origin). Use
  EKF/UKF/PF for nonlinear levels.
- **KalmanNet hardware split.** Training/validation run fully vectorized and
  batched on the GPU; test-time `estimate()` runs strictly sequentially on the
  CPU (microprocessor-deployment simulation).
- **No pooled RMSE.** RMSE is reported per named physical state variable
  (`x,y,z` for Lorenz; `θ,ω` for pendulum); the single scalar across mixed
  dimensions has been removed. Uncertainty is scored with NEES/NLL.
- **Memory metric disabled.** `metrics.measure_memory()` raises
  `NotImplementedError` (whole-process RSS was meaningless per-estimator).

Glue layers, documented inline rather than with their own README (read the
module directly):
- `datasets/` — `TrajectoryDataset`/`DatasetMetadata` schema + HDF5 read/write.
  Each split (`train`/`val`/`test`) is one `.h5` file holding `states`
  `[N,T,nx]`, `observations` `[N,T,ny]`, and one `timestamps` `[T]` array.
- `experiments/` — `ExperimentRunner`: fit → time `estimate()` → score → persist.
- `storage/` — SQLite repository (`schema.sql`) backing the experiment log
  (`experiments` / `metrics` / `artifacts` tables).
- `configs/`, `utils/` — config dataclasses, timing/logging/seeding helpers.

## Module docs (mathematical models, noise vs. state)

- [benchmark_levels/README.md](benchmark_levels/README.md) — `x_{t+1}=f(x_t,t)+w_t`,
  `y_t=h(x_t)+v_t` for linear / nonlinear / pendulum / lorenz (+ `lorenz_fea`,
  the forward-Euler-Jacobian baseline).
- [estimators/README.md](estimators/README.md) — KF/EKF/UKF/PF update equations,
  KalmanNet's learned-gain formulation, filterpy reference filters.
- [metrics/README.md](metrics/README.md) — per-variable RMSE, NEES/NLL,
  latency/runtime formulas.
- [visualization/README.md](visualization/README.md) — plot ↔ metric/array contract.

`summary.md` is the consolidated mathematical reference (every equation with a
file:line citation). The five README files and `summary.md` describe whatever is
currently in the corresponding folder, so they go stale the moment the code
changes — regenerate them rather than diffing line-by-line after a refactor.

## Methodology critique

The READMEs describe **what the code is and does**. Separate `Critique.md` files
review **whether the methodology is sound** — validity of the benchmark design,
correctness of the metrics, and fidelity of each estimator to standard
textbook/industry practice (e.g. running a linear KF on a nonlinear system):

- [Critique.md](Critique.md) — cross-cutting (single-run evaluation, filter
  consistency, latency/memory fairness, time-threading).
- [benchmark_levels/Critique.md](benchmark_levels/Critique.md),
  [estimators/Critique.md](estimators/Critique.md),
  [metrics/Critique.md](metrics/Critique.md),
  [visualization/Critique.md](visualization/Critique.md) — per-module.

## Time-varying dynamics

`FilterModel.f(x, t)` takes a timestep argument so a level can have time-varying
forcing (only `nonlinear`'s `8 cos(1.2 t)` term uses it today). EKF, UKF, PF and
KalmanNet thread the dataset timestamp into `f`; the linear KF does **not** — it
is a time-invariant filter by construction (constant `F`/`H`), so there is no `t`
to thread. See [estimators/README.md](estimators/README.md).

## Extending the benchmark

Adding a new system, estimator, metric, or plot only requires touching one
folder — each module README has an "Extending with a new X" section at the
bottom describing the contract to satisfy (`BenchmarkLevel`/`BaseSimulator`,
`BaseEstimator`, metric function signature, plot function signature) so new
modules stay drop-in compatible with `experiments/runner.py` without changes
to the runner itself.
