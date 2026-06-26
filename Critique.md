# Critique — cross-cutting methodology

This file reviews **whether the benchmark is methodologically valid**, separately
from the READMEs (which document what the code does). Each item is tagged:

- **[bug]** — incorrect relative to its own stated intent.
- **[design-limitation]** — works as written, but falls short of standard
  state-estimation evaluation practice.
- **[debatable-choice]** — a defensible decision with a real downside worth
  naming.
- **[valid-as-is]** — confirmed sound; recorded so it isn't re-litigated.

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

## 2. No filter-consistency check (NEES/NIS) — [design-limitation]

The benchmark scores estimators on point-RMSE only. It never checks whether each
filter's reported covariance `P` is **consistent** with its actual error —
i.e. the Normalized Estimation Error Squared (NEES) against the χ² bounds, the
Normalized Innovation Squared (NIS), or innovation whiteness. These are the
textbook tools (Bar-Shalom et al.) for deciding whether a filter is
*over-confident* or *under-confident*, not just accurate.

Consequence: a filter can post a competitive RMSE while being badly inconsistent
(its `P` does not reflect its true uncertainty), and this benchmark would not
distinguish it from a well-calibrated filter. For Kalman-family filters whose
entire premise is a calibrated posterior, omitting consistency is a real gap.

*Recommendation:* have estimators optionally expose their per-step `P` (and `S`),
add a `metrics/consistency.py` computing average NEES/NIS, and flag values
outside the χ² interval.

## 3. No optimality reference beyond the linear level — [design-limitation]

`LinearBenchmark` has a known optimum (the KF is Bayes-optimal there), so "how
close to optimal" is answerable on that level. The three nonlinear levels have
**no reference** — no posterior Cramér-Rao lower bound (PCRLB), no long-run
particle filter treated as ground-truth posterior. So on `nonlinear`/`pendulum`/
`lorenz` the benchmark can rank estimators *relative to each other* but cannot
say how far the best of them is from the achievable optimum.

*Recommendation:* compute the PCRLB for at least one nonlinear level, or run a
very-large-`M` particle filter as an approximate-optimal baseline.

## 4. Latency excludes fit() and JIT warm-up — [debatable-choice]

`runner.run` times only `estimator.estimate()` with `time.perf_counter()`
([runner.py:66-68](experiments/runner.py#L66)); `fit()` and numba's first-call
JIT compilation are outside the timed window. Read as **in-flight inference
latency**, this is defensible (training/compilation are amortized, one-time,
"pre-flight" costs). Two caveats keep it from being a clean cross-estimator
number:

1. **It is not disclosed in the stored metric.** `ExperimentResult` records
   `runtime_seconds`/`runtime_per_step_ms` with no field marking that `fit()` and
   warm-up were excluded — a reader of the SQLite log has no way to know.
2. **KalmanNet's timed `estimate()` is dominated by an implementation artifact,
   not its algorithm.** `_process_model_step`
   ([estimators/neural/kalmannet.py:100-109](estimators/neural/kalmannet.py#L100))
   round-trips every state vector through NumPy (`tensor → cpu().numpy() → Python
   loop calling f → tensor`) once per timestep. That per-row Python loop, not the
   GRU, dominates the wall-clock time. So KalmanNet's latency reflects how its
   predict step is *wired*, not the cost of the learned-gain idea — making a
   latency comparison against the vectorized classical filters not
   apples-to-apples.
3. Numba estimators get their first-call JIT cost excluded only if a warm-up call
   happened earlier in the same process; `run()` does not do a throwaway warm-up,
   so the **first** benchmarked numba estimator in a fresh process can absorb
   compilation time inside its timed window, while a later one does not.

*Recommendation:* record a boolean/columns documenting what the timer includes;
do one warm-up `estimate()` before the timed call; and either vectorize
KalmanNet's predict or annotate its latency as implementation-bound.

## 5. Memory metric is whole-process RSS — [design-limitation]

`measure_memory()` returns `psutil.Process(os.getpid()).memory_info().rss`
([metrics/memory.py:14](metrics/memory.py#L14)) — the **entire process's**
resident memory at the moment it is called, not the estimator's own allocation.
By the time it runs, the process has loaded numpy, torch, numba caches, the full
dataset, and every prior estimator's leftovers. So `memory_mb` is essentially a
constant process baseline plus noise, not a per-estimator footprint, and should
not be used to compare estimators' memory cost.

*Recommendation:* measure a *delta* (RSS after − RSS before the estimator runs,
or `tracemalloc` peak around `estimate()`), and run each estimator in a fresh
subprocess if a clean number is needed.

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
| 1 | Single run, no seed-averaging / error bars | design-limitation | runner.py:63-95 |
| 2 | No NEES/NIS filter-consistency check | design-limitation | metrics/ (absent) |
| 3 | No PCRLB / optimal baseline on nonlinear levels | design-limitation | benchmark_levels/ (absent) |
| 4 | Latency excludes fit/JIT; KalmanNet impl-bound; not disclosed | debatable-choice | runner.py:66-68, kalmannet.py:100 |
| 5 | Memory = whole-process RSS, not per-estimator | design-limitation | metrics/memory.py:14 |
| 6 | KF correctly does not thread time t | valid-as-is | kf.py:44-45 |
