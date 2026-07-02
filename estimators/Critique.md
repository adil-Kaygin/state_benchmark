# Critique — estimators

Methodology and fidelity review of the estimators. The question the project
explicitly raised — **is running a linear Kalman filter on a nonlinear system
valid, and how is it handled?** — is now answered in code: it is *not* valid, and
the KF refuses to do it (§1). Tags: **[bug]** / **[design-limitation]** /
**[debatable-choice]** / **[valid-as-is]** / **[resolved]**. See the root
[Critique.md](../Critique.md) for cross-cutting issues.

---

## 1. KF on a nonlinear system: now refused on both paths — [resolved]

*Was a design-limitation; fixed.* `KalmanFilterEstimator.estimate()` now calls
`assert_linear_model` on **every** run and raises `ValueError` on a nonlinear
`f`/`h`, so the KF refuses `pendulum`/`nonlinear`/`lorenz` instead of silently
running an origin-linearized filter that diverges. The pure-NumPy KF path (the
previously-unguarded one) and the `use_numba` flag have been **deleted**
entirely — the recursion runs only in the `@njit` `kf_loop_batch` kernel, and the
check is hardened to validate `F`/`H` shapes and probe linearity at the origin,
basis directions, and random points. The default notebook now includes KF only
on the linear level. KF-on-nonlinear can no longer be mistaken for a real result
because it cannot run.

## 2. Numba ↔ NumPy parity — [largely resolved]

*The dual-path risk is mostly gone.* KF/EKF/UKF no longer have a pure-NumPy
path at all — the NumPy fallbacks and the linear-only fast UKF path were deleted,
so there is a single implementation per filter (the `@njit` kernel) and nothing
to drift out of parity. What remains is the level dynamics being written twice
(Python `f/h/F/H` in each level **and** their `@njit`/torch mirrors in
`_numba_dynamics.py` / `_torch_dynamics.py`), plus the custom-vs-`torch-kf`/`torchfilter`
cross-check. There is still no automated parity script (the project does not keep
a `tests/` dir), so keeping those mirrors in sync remains a manual discipline; a
standalone parity script on `LinearBenchmark` would close the gap.

## 3. EKF uses the Euler-order Lorenz Jacobian — [resolved, cross-ref]

*Fixed.* The standard `LorenzBenchmark` now supplies the exact RK4-step Jacobian,
so EKF's covariance step is integration-order consistent with its RK4 mean step
(see [benchmark_levels/Critique.md §1](../benchmark_levels/Critique.md)). EKF is
no longer penalized on Lorenz for a reason unrelated to the EKF algorithm. The
old Euler-order Jacobian survives only on the `lorenz_fea` baseline.

## 4. KalmanNet's predict step is a per-row NumPy loop — [resolved (training); inference CPU-sequential by design]

*Was a design-limitation; reworked.* The per-row NumPy round-trip
(`_process_model_step`) has been **removed from training**. `fit()`/validation now
run fully vectorized and batched on the GPU using `FilterModel.torch` (batched
torch `f`/`h`), so each timestep is a single on-device tensor op — no Python
per-row loop, no NumPy round-trip. Test-time `estimate()` is now **deliberately**
strictly sequential on the CPU (one trajectory/timestep at a time, NumPy `f`/`h`),
simulating microprocessor deployment — so its latency reflects that intended
deployment condition rather than an accidental implementation artifact. Note this
also reframes root [Critique.md §4](../Critique.md): the inference cost is now an
intentional measurement, not a wiring accident.

## 5. `load()` cannot reconstruct a classical estimator — [valid-as-is]

`save()`/`load()` on the classical (and neural) estimators persist parameters as
JSON, but `load()` raises because a `FilterModel` is live Python callables (`f`,
`h`, Jacobians) that JSON cannot rebuild. This is a genuine limitation of the
persistence layer, but it is correctly surfaced (it raises rather than silently
returning a broken estimator) and documented at the call sites. Recorded as a
known constraint, not a defect: re-instantiate from the `BenchmarkLevel` instead
of `load()`-ing a classical filter.

## 6. Neural baselines are stubs — [valid-as-is]

`neural_ode.py` and `transformer.py` raise `NotImplementedError` and are confined
to `EXPERIMENTAL_ESTIMATORS`, so default sweeps don't crash on them. This is the
right handling for unfinished work; just note the benchmark's "neural" coverage is
currently **only** KalmanNet — there is no learned filter that drops the analytic
process model, so the neural-vs-classical comparison is narrower than the directory
structure suggests.

---

## Summary table

| # | Issue | Tag | Where |
|---|-------|-----|-------|
| 1 | KF-on-nonlinear: origin linearization, NumPy path unguarded | resolved (KF refuses nonlinear; NumPy path deleted) | kf.py |
| 2 | Numba/NumPy & custom/torchfilter parity untested | largely resolved (no dual filter path); dynamics mirrors still manual | _numba_kernels.py |
| 3 | EKF uses Euler-order Jacobian on RK4 Lorenz | resolved (exact RK4 Jacobian) | (cross-ref) lorenz.py |
| 4 | KalmanNet predict is per-row NumPy loop | resolved for training (batched torch GPU); inference CPU-sequential by design | kalmannet.py |
| 5 | `load()` cannot rebuild a FilterModel | valid-as-is | classical/neural `load` |
| 6 | neural_ode / transformer are stubs | valid-as-is | neural/*.py |
