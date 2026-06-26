# Critique — estimators

Methodology and fidelity review of the estimators, including the question the
project explicitly raised: **is running a linear Kalman filter on a nonlinear
system valid, and how is it handled?** Tags: **[bug]** / **[design-limitation]**
/ **[debatable-choice]** / **[valid-as-is]**. See the root
[Critique.md](../Critique.md) for cross-cutting issues.

---

## 1. KF on a nonlinear system: fixed-point linearization, NumPy path unguarded — [design-limitation]

`KalmanFilterEstimator` linearizes **once at the origin** and holds the result
constant for the whole trajectory:

```python
F = self._model.F(np.zeros(nx))   # kf.py:44
H = self._model.H(np.zeros(nx))   # kf.py:45
```

Is this valid? **Only for `LinearBenchmark`**, where `F`/`H` are genuinely
constant matrices that ignore their input — there the KF is Bayes-optimal and
this is exactly right. On any nonlinear level it is a *fixed-point linearization
about `x = 0`* that never re-linearizes at the current estimate.

Relative to standard industry/textbook practice, that is **not** how you run a
Kalman filter on a nonlinear system. The standard answer is:

- **EKF** — re-linearize `F = ∂f/∂x` at the current estimate every step (present
  here, [ekf.py:83](classical/ekf.py#L83)).
- **UKF** — propagate sigma points through the true `f`/`h`, no Jacobian
  (present).
- **PF** — sample the true posterior, no Gaussian assumption (present).

So the repo *has* the correct nonlinear filters; the issue is the **guarding** of
the plain KF:

- The **numba path** refuses a nonlinear model: `assert_linear_model` probes
  `f`/`h` against `F@x`/`H@x` at a random point and raises `ValueError`
  ([kf.py:51](classical/kf.py#L51)). Good.
- The **pure-NumPy path has no such guard** ([kf.py:63-89](classical/kf.py#L63)).
  Call `KalmanFilterEstimator(model, use_numba=False)` with a `pendulum` /
  `nonlinear` / `lorenz` model and it will silently run the origin-linearized KF,
  almost certainly diverge, and report a number. The only thing that keeps it from
  `NaN` is the `1e12` covariance clamp — which keeps the output **finite but not
  correct**.

Consequence: KF appears in the default `ESTIMATORS` registry alongside EKF/UKF/PF/
KalmanNet and is run on every level. On the three nonlinear levels its RMSE is the
RMSE of a *deliberately-wrong baseline*, but nothing in the metric or the plot
labels it as such, so a reader can misread "KF did poorly on Lorenz" as a fair
algorithmic result rather than a known invalid configuration.

*Recommendation (pick one):*
1. Add the same `assert_linear_model` guard to the NumPy path so KF refuses
   nonlinear models on both paths; **or**
2. Keep KF-on-nonlinear deliberately, but mark it explicitly as a
   "linear-baseline (expected to diverge)" entry in results/plots so its RMSE is
   never mistaken for a real method's.

This is the cleanest single improvement to the benchmark's validity.

## 2. Numba ↔ NumPy parity is asserted but has no regression test — [design-limitation]

KF and UKF each have two code paths (custom NumPy and `@njit` kernels in
`_numba_kernels.py`) that are *supposed* to be numerically identical up to
`fastmath` reordering. The previous documentation claimed this was "covered by
`tests/test_classical_filters.py`" — **that file (and the entire `tests/`
directory) does not exist** in the repo. There is no automated check that the two
paths agree, that the linear-only fast paths match the general path on linear
data, or that the filterpy reference filters match the custom ones.

Two independent implementations of the same math with no parity test is a
standing correctness risk: a future edit to one path can silently diverge from
the other, and `use_numba=True` (the default) vs `False` could then give
different RMSE for the same estimator on the same data.

*Recommendation:* add a parity check (even a single script, not a test suite if
the project avoids `tests/`) asserting NumPy-vs-numba and custom-vs-filterpy
agreement on `LinearBenchmark` within `fastmath` tolerance.

## 3. EKF uses the Euler-order Lorenz Jacobian — [design-limitation, cross-ref]

EKF's covariance step uses `F_jac` from the level. On Lorenz that Jacobian is the
first-order `I + dt·J` while EKF's mean step uses the level's RK4 `f` — an
internally inconsistent linearization. Detailed in
[benchmark_levels/Critique.md §1](../benchmark_levels/Critique.md). This penalizes
EKF specifically on Lorenz for a reason unrelated to the EKF algorithm itself.

## 4. KalmanNet's predict step is a per-row NumPy loop — [design-limitation]

`_process_model_step` ([neural/kalmannet.py:100-109](neural/kalmannet.py#L100))
applies `FilterModel.f` to a batch by detaching to CPU, looping over rows in
Python, and re-wrapping as a tensor — every timestep. This is a deliberate fidelity
choice (KalmanNet's process model is then bit-identical to the classical filters'),
but it has two methodological consequences:

- It is the **dominant runtime cost** of KalmanNet, so the latency metric measures
  the NumPy round-trip, not the learned-gain network (see root
  [Critique.md §4](../Critique.md)).
- It **breaks GPU/batch acceleration** of the predict step and forces inference to
  CPU, so KalmanNet cannot be scaled the way a fully-torch model could — limiting
  how seriously its training/throughput can be compared to a real neural baseline.

*Recommendation:* for any level intended for KalmanNet at scale, also provide a
vectorized/torch `f`; document current numbers as CPU-Python-bound.

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
| 1 | KF-on-nonlinear: origin linearization, NumPy path unguarded | design-limitation | kf.py:44-45, 63-89 |
| 2 | Numba/NumPy & custom/filterpy parity untested (no `tests/`) | design-limitation | _numba_kernels.py |
| 3 | EKF uses Euler-order Jacobian on RK4 Lorenz | design-limitation | (cross-ref) lorenz.py |
| 4 | KalmanNet predict is per-row NumPy loop (latency/GPU) | design-limitation | kalmannet.py:100-109 |
| 5 | `load()` cannot rebuild a FilterModel | valid-as-is | classical/neural `load` |
| 6 | neural_ode / transformer are stubs | valid-as-is | neural/*.py |
