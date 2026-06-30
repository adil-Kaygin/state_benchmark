# Issue 7: Wire NEES / NLL uncertainty metrics into the benchmark

The posterior-consistency metrics already exist in
[`metrics/uncertainty.py`](../metrics/uncertainty.py) — `compute_nees`,
`compute_nees_chi2_bounds`, `compute_nll` — but **nothing in the pipeline calls
them**, because they need each estimator's reported covariance `P` and no
estimator currently returns one. Point RMSE ignores the covariance entirely, so
right now a filter that is wildly over-confident (P far too small) scores the
same as a well-calibrated one. This issue closes that gap.

## Why

A Kalman-family filter outputs a *distribution* `N(x̂, P)`, not just a point
`x̂`. Two estimators with identical RMSE can have very different posterior
quality: one reports a covariance the errors actually live in, the other lies
about its confidence. NEES and NLL are the standard scores for that (see the
module docstring in [`metrics/uncertainty.py`](../metrics/uncertainty.py)):

- **NEES** `e = (x-x̂)ᵀ P⁻¹ (x-x̂)`; a consistent filter has mean NEES ≈ `nx`,
  with the chi-square band from `compute_nees_chi2_bounds`. Below ⇒
  under-confident, above ⇒ over-confident.
- **NLL** `½[(x-x̂)ᵀ P⁻¹ (x-x̂) + ln det(2π P)]`; a proper scoring rule that
  penalizes both point error and miscalibrated covariance.

This pairs naturally with the new single-test-set methodology: NEES/NLL are also
per-trajectory-averageable, so they get the same mean ± std / 95% CI over the
test trajectories (extend
[`metrics/aggregate.py`](../metrics/aggregate.py) the same way as RMSE).

## The blocker: estimators do not return `P`

`BaseEstimator.estimate(dataset)` returns only the point estimates `[N, T, nx]`.
The uncertainty metrics need the covariance sequence `[N, T, nx, nx]`. The
classical filters (KF/EKF/UKF) **already compute `P` every step** internally and
throw it away — exposing it is cheap. The neural filters mostly do **not** have a
calibrated covariance, so they need either a covariance head, an ensemble, or to
be explicitly excluded.

## Scope (recommended: classical filters first, honest exclusion of the rest)

1. **Covariance-returning API.** Add an opt-in covariance output to the estimator
   contract. Suggested: a method
   `estimate_with_covariance(dataset) -> (estimates [N,T,nx], covariances
   [N,T,nx,nx])`, with a capability flag (e.g. `returns_covariance: bool`).
   Keep `estimate()` unchanged so nothing else breaks. Per the fail-fast rule,
   `estimate_with_covariance` on an estimator without a covariance must raise
   `NotImplementedError`, not return a dummy `P`.

2. **Classical filters (KF/EKF/UKF, + the filterpy variants).** Surface the `P`
   they already maintain. These are the levels where NEES/NLL are most meaningful
   (their `P` is the actual propagated covariance). The metrics fail loudly on a
   non-PD `P`, which is the correct behaviour — a non-invertible posterior is a
   real defect, not something to paper over.

3. **Neural filters.** They have no calibrated `P` by default ⇒ set
   `returns_covariance = False` and **skip** them in the NEES/NLL table (document
   it; do not fabricate a covariance). A proper neural-covariance head / deep
   ensemble is a separate follow-up issue.

4. **Wire into the experiment + notebook.** For every estimator with
   `returns_covariance`, compute per-trajectory NEES/NLL on the test set,
   aggregate to mean ± std / 95% CI, log to Comet, and add to the summary table.
   Show the chi-square acceptance band (`compute_nees_chi2_bounds`) alongside the
   NEES column so over/under-confidence is readable at a glance.

5. **Visualization (optional).** A NEES-vs-`nx` plot with the chi-square band,
   and an NLL bar chart with error bars (reuse the
   `std_*` / `runtime_errors` error-bar arguments already in
   [`visualization/`](../visualization/)).

## Acceptance criteria

- [ ] Estimator contract gains an opt-in covariance output + capability flag;
      `estimate()` is unchanged; missing covariance raises `NotImplementedError`
      (no dummy `P`).
- [ ] KF/EKF/UKF (and the filterpy variants) return their `P` sequence
      `[N, T, nx, nx]`.
- [ ] `metrics/aggregate.py` aggregates per-trajectory NEES/NLL to
      mean ± std / 95% CI over the test trajectories (same pattern as RMSE).
- [ ] Experiment/notebook computes + logs NEES/NLL for every covariance-returning
      estimator, with the chi-square band shown next to NEES; neural filters
      without a covariance are explicitly skipped and documented.
- [ ] No `tests/`, no `pip install`; lazy `scipy` import for the chi-square band
      (already the case in `compute_nees_chi2_bounds`).
