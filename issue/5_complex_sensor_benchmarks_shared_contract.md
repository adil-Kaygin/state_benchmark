# Issue 5: Shared Contract for Complex / Multi-Sensor Benchmark Levels (read first)

This is the common reference for the new **harder benchmark levels** whose
difficulty comes from the *measurement side*: multiple heterogeneous sensors,
multiple noise levels, polar (range/bearing) observations, dropouts, biases, and
asynchronous sampling. The first concrete level built on it is
[Issue 6 — Multi-Sensor Vehicle Tracking](6_vehicle_tracking_benchmark.md);
later levels (multi-radar air tracking, GPS+IMU fusion, etc.) should assume
everything below and only state their own deltas.

> The estimator-side contract is unchanged — see
> [Issue 0 — neural shared contract](0_neural_filters_shared_contract.md) for the
> `BaseEstimator` / hardware-split / fail-fast rules. **This issue extends the
> _benchmark-level_ contract** (`benchmark_levels/base.py`,
> `datasets/schema.py`), not the estimator one. Read
> [`linear.py`](../benchmark_levels/linear.py),
> [`pendulum.py`](../benchmark_levels/pendulum.py) and
> [`lorenz.py`](../benchmark_levels/lorenz.py) end to end first — every new level
> mirrors their structure (a `BaseSimulator` + a `BenchmarkLevel`).

## Why this exists

Today's four levels (linear, pendulum, nonlinear, lorenz) all use a **single
sensor with one fixed Gaussian noise**, and three of them observe a state
component directly (`h(x) = x[0]` or `H @ x`). Real estimation problems are hard
because of the *sensors*: you fuse several of them, each with a different noise
floor, a different geometry, sometimes a nonlinear (polar) readout, sometimes a
bias or a dropout. The classic example from the radar/sonar literature is a
vehicle tracked by several stations that each report **range and bearing**
(distance + angle) with **different accuracies** — low-grade nonlinearity in the
measurement, plus a genuine multi-sensor fusion problem. (See the references in
Issue 6.)

These levels exercise parts of the stack the current ones don't:

- EKF/UKF with a **nonlinear, non-identity `h`** and a **block-diagonal stacked
  `R`** (per-sensor noise) — the UKF advantage should finally show up.
- KalmanNet / the four neural filters on observations that are **not** a trivial
  slice of the state, with `ny != nx` and `ny` possibly large.
- Robustness to **missing / NaN observations** (sensor dropout) — currently no
  level produces them.

## What a complex level MAY add (and how to keep it in-contract)

The estimator interface and the dataset on-disk layout do **not** change. A
trajectory is still `states [N,T,nx]`, `observations [N,T,ny]`,
`timestamps [T]`, `metadata`. Everything new is encoded *inside* `ny` and inside
the `FilterModel`:

### 1. Multiple sensors → one stacked observation vector

Stack every sensor's reading into a single `y_t ∈ R^ny`. If `K` sensors each
emit a 2-D `[range, bearing]`, then `ny = 2K` and `h(x)` returns the `2K`-vector
of all sensors' predicted readings. `R` is **block-diagonal** `[ny, ny]`, one
block per sensor — this is exactly how the standard sequential/stacked
multi-sensor fusion is written and keeps the existing single-`h`/single-`R`
`FilterModel` shape intact. Do **not** invent a new multi-`h` field on
`FilterModel`; stacking keeps every existing estimator working unchanged.

### 2. Multiple noise levels

"Different noise levels" = different `R` blocks per sensor (e.g. a precise sensor
with σ_r=0.5 m and a cheap one with σ_r=5 m), **and/or** the same physical level
shipped at several global noise scales as separate sub-levels (mirror the
`process_noise_var` / `observation_noise_var` constructor knobs the existing
levels already expose). Prefer per-sensor heterogeneity as the *interesting* axis
and keep a scalar `noise_scale` multiplier for sweeps.

### 3. Nonlinear polar measurement (`h`, `H` Jacobian)

`range  = sqrt((px - sx)^2 + (py - sy)^2)`,
`bearing = atan2(py - sy, px - sx)` for a sensor at `(sx, sy)`. Provide the
**analytic Jacobian** `H(x)` (the contract already has the `H` slot, currently
trivial on most levels). Bearing residuals MUST be **angle-wrapped** to
`(-π, π]` wherever an innovation `y - h(x)` is formed — document this as a
known footgun for EKF/UKF and for any neural innovation feature.

### 4. Sensor dropout / missing data (optional, gated by a flag)

If a level emits dropouts, encode a missing reading as **`NaN`** in that sensor's
slot of `y_t` (never `0.0` — that is a silent, wrong measurement and violates the
fail-fast rule). A level that produces NaNs MUST set
`metadata`/docstring to say so loudly; estimators that cannot yet handle gating
should raise, not average over NaN. Keep dropout **off by default** so the level
is usable before estimators learn to gate.

## Required `FilterModel` completeness

Same object EKF/UKF/KalmanNet/neural filters consume
([`FilterModel`](../benchmark_levels/base.py)). For a complex level:

- `f, h` — NumPy single-vector process/measurement models. `h` returns the full
  stacked `[ny]` vector. **Angle-wrap any bearing residual at the call site**, not
  inside `h`.
- `F, H` — analytic Jacobians. `H` is `[ny, nx]` (stack the per-sensor
  `[2, nx]` blocks). Do not hand `None` to a level meant for EKF; a `None`
  Jacobian forces UKF-only and must be justified.
- `Q, R` — `[nx,nx]` and the **block-diagonal** `[ny,ny]`.
- `x0_mean`, `x0_cov` — required (these levels diverge easily; a good prior
  matters). The pendulum's analytic `x0_cov` from the init range
  ([`pendulum.py:155-156`](../benchmark_levels/pendulum.py#L155-L156)) is the
  pattern to copy.
- `numba` — `@njit` dynamics for the classical CPU kernels. There is **no NumPy
  fallback** in the classical filters, so a level consumed by EKF/UKF MUST ship
  `NumbaDynamics` (add a `build_<level>_numba_dynamics` to
  [`_numba_dynamics.py`](../benchmark_levels/_numba_dynamics.py), matching the
  stacked-`h` / wrapped-bearing math one-for-one). `atan2`, `sqrt`, `sin/cos` are
  all njit-supported.
- `torch` — batched `[B,nx]->[B,ny]` dynamics for KalmanNet GPU training (add a
  `build_<level>_torch_dynamics` to
  [`_torch_dynamics.py`](../benchmark_levels/_torch_dynamics.py)). Same math,
  `torch.atan2`/`torch.sqrt`. Required for any level you want KalmanNet/neural
  filters to train on GPU; if absent, those estimators must raise the Issue-0
  `ValueError`, not silently fall back.

**The three implementations (NumPy `f`/`h`, `@njit`, batched torch) MUST be
mathematically identical** — this is already the standing invariant for every
level (see the warnings atop `_torch_dynamics.py`) and is the single most common
source of "the neural net trained fine but EKF/the CPU path disagrees" bugs.

## Generation (mirror the existing levels exactly)

- A `BaseSimulator` subclass with `step(state, control, dt)` and
  `observe(state)`; `observe` builds the stacked multi-sensor reading and adds
  the per-sensor noise from the block `R`.
- `generate_dataset` writes the **same** train/val/test 70/15/15 split via
  `HDF5Writer` + `DatasetMetadata` — copy
  [`lorenz.py:130-169`](../benchmark_levels/lorenz.py#L130-L169) verbatim and only
  change the per-trajectory init and the `observe` call. Ground truth (`states`)
  is **never** noised, clipped, or dropped — only `observations` are.
- Seed with `np.random.default_rng(self._random_seed)`; the Monte-Carlo seed loop
  injects `random_seed` per run (see `BENCHMARK_CONFIGS` in
  [`linear.py`](../benchmark_levels/linear.py) and the note that `random_seed` is
  intentionally omitted from the per-level config dict).

## Wiring (do this for each new complex level)

1. New module `benchmark_levels/<name>.py` (Simulator + Benchmark).
2. `build_<name>_numba_dynamics` in `_numba_dynamics.py`,
   `build_<name>_torch_dynamics` in `_torch_dynamics.py`.
3. Register in [`benchmark_levels/__init__.py`](../benchmark_levels/__init__.py)
   (`BENCHMARK_LEVELS` dict, `__init__` imports, `__all__`).
4. Add a per-level entry to the experiment/notebook config dicts
   (`BENCHMARK_CONFIGS`, `BENCHMARK_CLASSES`, and the per-level estimator config
   blocks) so it runs in the standard sweep with the existing estimators.
5. Add `state_names` (e.g. `("px","py","vx","vy")`) so metrics/visualization label
   per-dimension RMSE physically.

## Fail-fast rules (unchanged, restated for the data side)

- No silent fallbacks. Missing sensor → `NaN` slot + loud metadata, never a
  fabricated `0.0`. Bad sensor geometry / non-PSD `R` / `ny != 2K` →
  descriptive `ValueError` at construction.
- Ground truth is untouched; only `observations` carry noise/dropout/bias.
- No `tests/`, no `pip install` on this machine (standing user rule —
  [[feedback_no_tests_no_pip]]). `numba`/`torch` imported lazily as the existing
  levels do.

## Definition of done (every complex level)

- [ ] `BaseSimulator` + `BenchmarkLevel` mirroring linear/pendulum/lorenz;
      70/15/15 HDF5 split; ground truth never corrupted.
- [ ] Stacked multi-sensor `h` → `[ny]`, block-diagonal `R [ny,ny]`, analytic
      `H [ny,nx]`; bearing residuals angle-wrapped at every innovation site.
- [ ] `FilterModel` complete: `f,h,F,H,Q,R,x0_*` **plus** `numba` and `torch`
      with identical math to the NumPy path.
- [ ] Dropout (if any) encoded as `NaN` + documented; off by default.
- [ ] Registered in `__init__.py` and the experiment/notebook config dicts;
      `state_names` set.
- [ ] No `tests/`, no `pip install`; lazy `numba`/`torch` imports.
