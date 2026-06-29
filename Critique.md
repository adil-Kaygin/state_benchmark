# Critique — cross-cutting methodology

This file reviews **whether the benchmark is methodologically valid**, separately
from the READMEs (which document what the code does). Each item is tagged:

- **[bug]** — incorrect relative to its own stated intent.
- **[design-limitation]** — works as written, but falls short of standard
  state-estimation evaluation practice.
- **[debatable-choice]** — a defensible decision with a real downside worth
  naming.
- **[valid-as-is]** — confirmed sound; recorded so it isn't re-litigated.
- **[resolved]** — was one of the above; fixed in code, kept here as a record of
  what changed and why.

Per-module critiques live in `benchmark_levels/Critique.md`,
`estimators/Critique.md`, `metrics/Critique.md`, `visualization/Critique.md`.

---

## 1. Single-run evaluation, no Monte-Carlo averaging — [design-limitation]

`ExperimentRunner.run` ([experiments/runner.py:63-95](experiments/runner.py#L63))
runs each estimator **once** on one `test_dataset` and stores a single scalar
RMSE per `(benchmark, estimator)`. There is no averaging over independent seeds
or trajectory draws, and no variance/error-bar is computed or stored
(`ExperimentResult` has one `rmse` field, [experiments/result.py](experiments/result.py)).

Why this matters: the data is **stochastic** (process + observation noise on
every level) and, for `lorenz`, **chaotic** (positive Lyapunov exponent — two
seeds diverge exponentially). A single realization is therefore a high-variance
point estimate, and a difference between two estimators' RMSE on one run can be
within noise. Standard practice in the state-estimation literature is to report
**mean ± std (or a confidence interval) over ≥10–30 Monte-Carlo runs** with
different seeds.

*Recommendation:* loop `run()` over a list of seeds and aggregate (mean, std,
maybe median) before ranking; surface the spread in `plot_rmse_comparison` as
error bars.

## 2. No filter-consistency check (NEES/NIS) — [resolved]

*Was a design-limitation; fixed.* `metrics/uncertainty.py` adds `compute_nees`
(mean Normalized Estimation Error Squared against the truth, with
`compute_nees_chi2_bounds` for the χ² acceptance interval) and `compute_nll`
(Gaussian negative log-likelihood) — both score whether a filter's reported `P`
is consistent with its actual error, not just how accurate its point estimate
is. Verified analytically: a consistent synthetic filter gives mean NEES ≈ `nx`
and NLL matching the closed-form Gaussian value. NIS (innovation-based, scored
before the update rather than against truth) and a `metrics/`-level wiring of
estimators exposing per-step `P` to the runner remain open if a no-ground-truth
consistency check is wanted in addition.

## 3. No optimality reference beyond the linear level — [design-limitation]

`LinearBenchmark` has a known optimum (the KF is Bayes-optimal there), so "how
close to optimal" is answerable on that level. The three nonlinear levels have
**no reference** — no posterior Cramér-Rao lower bound (PCRLB), no long-run
particle filter treated as ground-truth posterior. So on `nonlinear`/`pendulum`/
`lorenz` the benchmark can rank estimators *relative to each other* but cannot
say how far the best of them is from the achievable optimum.

*Recommendation:* compute the PCRLB for at least one nonlinear level, or run a
very-large-`M` particle filter as an approximate-optimal baseline.

## 4. Latency excludes fit() and JIT warm-up — [debatable-choice, partly resolved]

`runner.run` times only `estimator.estimate()` with `time.perf_counter()`; `fit()`
and numba's first-call JIT compilation are outside the timed window. Read as
**in-flight inference latency**, this is defensible (training/compilation are
amortized, one-time, "pre-flight" costs). Remaining caveats:

1. **It is not disclosed in the stored metric.** `ExperimentResult` records
   `runtime_seconds`/`runtime_per_step_ms` with no field marking that `fit()` and
   warm-up were excluded — a reader of the SQLite log has no way to know.
2. **[resolved] KalmanNet's `estimate()` cost is now intentional, not an
   implementation accident.** The old per-row NumPy round-trip
   (`_process_model_step`) has been removed from *training*, which now runs
   fully vectorized on the GPU (see `estimators/Critique.md §4`). Test-time
   `estimate()` is now **deliberately** strictly sequential on the CPU
   (`_run_sequence_sequential_cpu`), simulating microprocessor deployment — so
   its latency reflects that chosen deployment condition, not a wiring mistake.
   It is still not apples-to-apples against the vectorized classical filters,
   but that asymmetry is now a stated experimental design choice rather than an
   unexamined artifact; it should still be labelled as such in any latency
   comparison table.
3. Numba estimators get their first-call JIT cost excluded only if a warm-up call
   happened earlier in the same process; `run()` does not do a throwaway warm-up,
   so the **first** benchmarked numba estimator in a fresh process can absorb
   compilation time inside its timed window, while a later one does not.

*Recommendation:* record a boolean/column documenting what the timer includes,
and do one warm-up `estimate()` before the timed call.

## 5. Memory metric is whole-process RSS — [resolved]

*Was a design-limitation; the metric is now disabled rather than fixed in place.*
`measure_memory()` previously returned `psutil.Process(os.getpid()).memory_info().rss`
— the **entire process's** resident memory, not the estimator's own allocation,
so it was a constant process baseline plus noise rather than a useful
per-estimator footprint. It now raises `NotImplementedError("Memory measurement
is currently unsupported.")` instead of reporting that misleading number; the
runner no longer calls it or stores a `memory_mb` column. A correct per-estimator
metric (RSS delta, `tracemalloc` peak, or a fresh-subprocess measurement) remains
open if memory tracking is wanted again.

## 6. Time-threading is correct, but asymmetric — [valid-as-is]

EKF, UKF, PF and KalmanNet thread the dataset timestamp into `FilterModel.f(x, t)`
(EKF [ekf.py:82](estimators/classical/ekf.py#L82), UKF
[ukf.py:129](estimators/classical/ukf.py#L129), PF
[pf.py:65](estimators/classical/pf.py#L65), KalmanNet
[kalmannet.py:190](estimators/neural/kalmannet.py#L190)). The linear KF (and
`FilterpyKFEstimator`) does **not** — and that is correct, not a bug: KF is a
linear time-invariant filter that holds `F`/`H` constant, so there is no `t` to
thread. The only level with time-varying forcing (`nonlinear`'s `8 cos(1.2 t)`)
is one the linear KF cannot validly run anyway (see `estimators/Critique.md`).
Recorded here because the asymmetry looks like an omission but is intended.

---

## Summary table

| # | Issue | Tag | Where |
|---|-------|-----|-------|
| 1 | Single run, no seed-averaging / error bars | design-limitation | runner.py |
| 2 | No NEES/NIS filter-consistency check | resolved (NEES + NLL added) | metrics/uncertainty.py |
| 3 | No PCRLB / optimal baseline on nonlinear levels | design-limitation | benchmark_levels/ (absent) |
| 4 | Latency excludes fit/JIT; KalmanNet impl-bound; not disclosed | partly resolved (KalmanNet CPU-sequential by design) | runner.py, kalmannet.py |
| 5 | Memory = whole-process RSS, not per-estimator | resolved (disabled, raises NotImplementedError) | metrics/memory.py |
| 6 | KF correctly does not thread time t | valid-as-is | kf.py |
