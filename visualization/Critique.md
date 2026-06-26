# Critique — visualization

Brief methodology review of the plotting layer. Tags: **[bug]** /
**[design-limitation]** / **[debatable-choice]** / **[valid-as-is]**. See the root
[Critique.md](../Critique.md) for cross-cutting issues.

The visualization code itself is sound — it is pure rendering with no hidden
computation, and the array/metric contract is clean. The critiques below are about
what the plots *don't* show, which ties directly to the metric gaps in
[metrics/Critique.md](../metrics/Critique.md) and the root critique.

---

## 1. No uncertainty / confidence bands — [design-limitation]

`plot_trajectory` and `plot_states_all_dims` overlay the point estimate `x̂` on
the truth `x`, but never draw the filter's reported covariance (a `±√P` band). So
there is no visual way to see whether a filter is **over- or under-confident** —
exactly the consistency question the benchmark never measures
(root [Critique.md §2](../Critique.md)). A filter that tracks the mean well but
reports an absurd `P` looks identical to a well-calibrated one in these plots.

*Recommendation:* let `plot_trajectory` optionally take a per-step variance array
and shade a confidence band (the uncertainty estimator already produces one).

## 2. Bar charts hide variance because there is only one run — [design-limitation]

`plot_rmse_comparison` and `plot_runtime_comparison` draw one bar per estimator
from a single scalar. Because the pipeline runs each estimator once (root
[Critique.md §1](../Critique.md)), there is no spread to plot, so the bars imply a
precision the single-seed experiment does not have. Two bars differing by a few
percent could easily reorder on another seed.

*Recommendation:* once results are averaged over seeds, add error bars; until then,
the bar charts should be read as illustrative, not as ranked-with-confidence.

## 3. Observations not plotted by default — [debatable-choice]

`plot_trajectory` deliberately plots only `x` vs `x̂`, not the raw `y`
observations (the README documents the workaround of passing observations in the
`estimates` slot, valid only when `ny == nx`). For levels where `ny < nx`
(`pendulum`, `lorenz`) there is no built-in way to visualize how noisy the sensor
signal was relative to the filtered estimate, which is a useful sanity check when
diagnosing why a filter under- or over-trusts its measurements.

*Recommendation:* add an optional `observations` overlay (plotting only the
observed dimensions) so the measurement noise is visible alongside the estimate.

---

## Summary table

| # | Issue | Tag | Where |
|---|-------|-----|-------|
| 1 | No covariance/confidence bands (can't see consistency) | design-limitation | trajectory.py |
| 2 | Single-run bars imply false precision | design-limitation | rmse.py / runtime.py |
| 3 | Observations not overlaid for `ny < nx` levels | debatable-choice | trajectory.py |
