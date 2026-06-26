# Critique — metrics

Methodology review of how results are scored, addressing the project's question:
**are the metrics calculated correctly?** Tags: **[bug]** /
**[design-limitation]** / **[debatable-choice]** / **[valid-as-is]**. See the root
[Critique.md](../Critique.md) for cross-cutting issues (single-run variance,
latency/memory fairness).

---

## 1. Pooled RMSE mixes dimensions of different physical scale — [design-limitation]

`compute_rmse` reduces the entire `[N, T, nx]` error array to one scalar:

```python
np.sqrt(np.mean((estimates - targets) ** 2))   # rmse.py:19
```

The arithmetic is *correct* (it is a valid root-mean-square). The methodological
problem is what it pools. Within a single benchmark the state dimensions can have
**different units and magnitudes**:

- `pendulum`: `θ` in radians (O(1)) vs `ω` in rad/s — different physical units.
- `lorenz`: `x`, `y` ~ O(10) vs `z` ~ O(25) — same units but very different scale.

A plain mean-square over those dimensions is **dominated by the largest-magnitude
dimension**, so the headline RMSE is effectively "how well did we track `z`" on
Lorenz, with `x`/`y` contributing little. This is a known subtlety the previous
docs framed only as a *cross-benchmark* problem ("don't average linear's RMSE
with Lorenz's"). It is also a *within-benchmark* problem: the single scalar is a
scale-weighted blend, not a balanced accuracy measure.

`compute_rmse_per_dim` and `compute_rmse_per_timestep` exist and are the right
tools, but `compute_rmse` is what `runner.py`/`ExperimentResult`/the bar chart use
as the headline number.

*Recommendation:* make per-dimension RMSE (or a per-dimension **normalized**
RMSE, dividing each dim by its own std/range) the primary reported metric, and
treat the pooled scalar as a convenience only.

## 2. RMSE ignores the reported posterior (no NLL / uncertainty scoring) — [design-limitation]

Every metric here scores the **point estimate** `x̂` against `x`. None scores the
filters' reported **uncertainty** `P`. A Bayesian filter's output is a
*distribution*, and the standard way to score a distribution is negative
log-likelihood (NLL) of the truth under `N(x̂, P)`, or the consistency checks
(NEES/NIS) in the root critique.

This is most visible for `KalmanNetUncertaintyEstimator`, which produces a
variance head specifically so its uncertainty can be evaluated — but there is **no
metric that consumes it**. Its calibration is unmeasured.

*Recommendation:* add a `metrics/nll.py` (Gaussian NLL of truth under the filter's
posterior) so the uncertainty-aware estimators can be scored on what they're for.

## 3. Unobserved dimensions are pooled into the headline RMSE — [debatable-choice]

On `lorenz`, `z` is unobserved (`ny = 2 < nx = 3`), yet it enters
`compute_rmse` with equal weight. Tracking an unobserved dimension purely from
dynamics is a fundamentally different (harder) task than tracking an observed one,
and folding both into one scalar conflates them. A reader cannot tell from the
headline number whether a filter is good at the observed states and bad at the
hidden one, or vice versa.

*Recommendation:* report observed-vs-unobserved RMSE separately (the per-dim
metric already enables this; the convention just isn't enforced).

## 4. `runtime_per_step_ms` returns 0.0 for `num_steps <= 0` — [debatable-choice]

```python
if num_steps <= 0: return 0.0    # runtime.py
```

Returning `0.0` for an undefined case (zero steps) silently produces a value that
reads as "infinitely fast" rather than "not applicable." It is unlikely to fire in
the normal pipeline (`N*T > 0` always), but if it ever does, a `0.0` in the
results table is misleading.

*Recommendation:* return `float("nan")` (or raise) so an undefined latency is not
mistaken for an excellent one.

## 5. Within-benchmark RMSE comparison is fair — [valid-as-is]

Subject to §1's scale caveat, every estimator scored on a given benchmark sees the
same `targets` array and the same reduction, so the *ranking* among estimators on
one benchmark is fair. The cross-benchmark caveat (don't average raw `compute_rmse`
across levels with different units) is correctly documented in the README. Recorded
here so the within-benchmark fairness isn't second-guessed: the metric ranks
correctly within a level; the open issues are scale-weighting (§1) and the absence
of uncertainty scoring (§2), not unfairness between estimators on the same data.

---

## Summary table

| # | Issue | Tag | Where |
|---|-------|-----|-------|
| 1 | Pooled RMSE scale-weighted across dims (even within a benchmark) | design-limitation | rmse.py:19 |
| 2 | No NLL / posterior scoring; uncertainty head unmeasured | design-limitation | metrics/ (absent) |
| 3 | Unobserved `z` pooled into headline RMSE | debatable-choice | rmse.py + lorenz |
| 4 | `runtime_per_step_ms` returns 0.0 (not NaN) on n<=0 | debatable-choice | runtime.py |
| 5 | Within-benchmark ranking is fair | valid-as-is | rmse.py |
