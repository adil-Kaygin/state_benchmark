# Critique — metrics

Methodology review of how results are scored, addressing the project's question:
**are the metrics calculated correctly?** Tags: **[bug]** /
**[design-limitation]** / **[debatable-choice]** / **[valid-as-is]** /
**[resolved]**. See the root [Critique.md](../Critique.md) for cross-cutting
issues (single-run variance, latency/memory fairness).

---

## 1. Pooled RMSE mixed dimensions of different physical scale — [resolved]

*Was a design-limitation; fixed.* The pooled-scalar `compute_rmse` (which mixed
state dimensions of different units/scales into one number, dominated by the
largest-magnitude dimension) has been **deleted entirely** — there is no scalar
fallback. `compute_rmse_per_dim` is now the only per-dimension reducer and
returns a `dict` keyed by the physical variable name
(`BenchmarkLevel.state_names`), used as the primary reported metric everywhere
(`runner.py`/`ExperimentResult`/`plot_rmse_comparison_per_dim`). Raises
`ValueError` if `state_names`'s length doesn't match `nx` (fail fast against a
silently mislabelled dimension).

## 2. RMSE ignored the reported posterior (no NLL / uncertainty scoring) — [resolved]

*Was a design-limitation; fixed.* `metrics/uncertainty.py` now scores the
filters' reported covariance `P`: `compute_nees` (Normalized Estimation Error
Squared, consistent filter ⇒ mean NEES ≈ `nx`, with `compute_nees_chi2_bounds`
for the acceptance interval) and `compute_nll` (Gaussian negative
log-likelihood of the truth under `N(x̂, P)` — a proper scoring rule). Both
raise on a non-positive-definite `P` rather than silently skipping. This gives
`KalmanNetUncertaintyEstimator`'s variance head (and any Kalman-family filter's
`P`) a metric that actually consumes it.

## 3. Unobserved dimensions are pooled into the headline RMSE — [resolved by §1]

On `lorenz`, `z` is unobserved (`ny = 2 < nx = 3`). Now that there is no pooled
scalar, `compute_rmse_per_dim`'s `{"x":..., "y":..., "z":...}` always reports
`z`'s error separately by name — a reader can directly compare observed (`x`,
`y`) vs unobserved (`z`) accuracy without any extra convention to remember or
enforce.

## 4. `runtime_per_step_ms` returned 0.0 for `num_steps <= 0` — [resolved]

*Was a debatable-choice; fixed.* `runtime_per_step_ms` now raises `ValueError`
for `num_steps <= 0` instead of returning `0.0` — an undefined latency can no
longer read as "infinitely fast" in a results table.

## 5. Within-benchmark RMSE comparison is fair — [valid-as-is]

Every estimator scored on a given benchmark sees the same `targets` array and
the same per-dimension reduction, so the *ranking* among estimators on one
benchmark is fair. With §1 resolved (no pooled scalar at all), the
cross-benchmark caveat is moot by construction — there is no single number to
mistakenly average across levels with different units.

---

## Summary table

| # | Issue | Tag | Where |
|---|-------|-----|-------|
| 1 | Pooled RMSE scale-weighted across dims | resolved (deleted; per-named-variable only) | rmse.py |
| 2 | No NLL / posterior scoring; uncertainty head unmeasured | resolved (NEES + NLL added) | uncertainty.py |
| 3 | Unobserved `z` pooled into headline RMSE | resolved by §1 (named per-dim dict) | rmse.py + lorenz |
| 4 | `runtime_per_step_ms` returns 0.0 (not NaN) on n<=0 | resolved (raises ValueError) | runtime.py |
| 5 | Within-benchmark ranking is fair | valid-as-is | rmse.py |
