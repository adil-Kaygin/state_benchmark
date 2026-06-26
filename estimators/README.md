# estimators

State estimators consume `observations` (noisy sensor data, `y_t`) and produce
`estimates` of the latent `states` (`x_t`), which `metrics.rmse.compute_rmse`
scores against the ground truth written by `benchmark_levels`. No estimator
reads `dataset.states` inside `estimate()` — that would leak the target.

```
BaseEstimator
 ├── estimator_name / estimator_type             # "classical" | "neural"
 ├── fit(train_dataset, val_dataset) -> None      # no-op for classical filters
 ├── estimate(dataset) -> ndarray [N, T, nx]       # the only inference call
 └── save(path) / load(path)                       # classical: params only,
                                                    # cannot reconstruct FilterModel
```

All classical estimators take a `FilterModel(f, h, F, H, Q, R, x0_mean, x0_cov,
numba)` produced by `BenchmarkLevel.get_filter_model()` — they are
benchmark-agnostic; swapping the benchmark swaps the model, not the filter code.
Every classical estimator initialises from `FilterModel.x0_mean`/`x0_cov`
(falling back to `zeros(nx)`/`eye(nx)` only when the model leaves them `None`).

**Registries** (`estimators/__init__.py`):
- `ESTIMATORS` — run by default sweeps: `kf`, `ekf`, `ukf`, `pf`, `kalmannet`.
- `EXPERIMENTAL_ESTIMATORS` — opt-in: `kalmannet_uncertainty`, `neural_ode`,
  `transformer`.
- `REFERENCE_ESTIMATORS` — third-party cross-checks: `filterpy_kf`,
  `filterpy_ekf`, `filterpy_ukf`.

## classical/kf.py — `KalmanFilterEstimator`

Optimal linear-Gaussian filter; statistically correct only on `LinearBenchmark`
(assumes `f(x) = Fx`, `h(x) = Hx` exactly).

```
predict:  x⁻ = F x,        P⁻ = F P Fᵀ + Q
update:   S  = H P⁻ Hᵀ + R
          K  = P⁻ Hᵀ S⁻¹
          x  = x⁻ + K(y - H x⁻)
          P  = (I - K H) P⁻
```

- `F = self._model.F(zeros(nx))`, `H = self._model.H(zeros(nx))` — evaluates the
  Jacobians once at the origin and holds them constant for the whole trajectory.
  Exact for `LinearBenchmark` (constant Jacobians); a fixed-point linearization
  everywhere else. KF is time-invariant — it does not thread the dataset
  timestamp into `f`.
- Two code paths: a numba-jitted batched loop (`_numba_kernels.kf_loop_batch`,
  default `use_numba=True`) and a pure-NumPy fallback (`use_numba=False` or numba
  absent), intended to be numerically identical up to `fastmath` float-reordering.
- `use_numba=True` calls `_numba_kernels.assert_linear_model`, which probes `f`/`h`
  against `F@x`/`H@x` at a random point and raises `ValueError` if they disagree —
  so the numba fast path refuses a nonlinear `FilterModel` instead of silently
  linearizing at the origin. The pure-NumPy path has no such guard.
- **Covariance bound:** after each predict/update, `_bound_cov` symmetrizes `P`
  (`0.5(P+Pᵀ)`) and clips it to `±1e12`. Under a model mismatch (e.g. this linear
  KF on chaotic Lorenz data) `P` grows geometrically; the clip keeps it finite
  (avoids `inf - inf = NaN` poisoning the run) while leaving a well-matched
  filter untouched.

## classical/ekf.py — `EKFEstimator`

First-order linearization at the current estimate; works on any `FilterModel`.

```
predict:  x⁻ = f(x, t),     F = ∂f/∂x|_x        P⁻ = F P Fᵀ + Q
update:   H  = ∂h/∂x|_x⁻
          S  = H P⁻ Hᵀ + R,  K = P⁻ Hᵀ S⁻¹
          x  = x⁻ + K(y - h(x⁻))
          P  = (I - K H) P⁻
```

- Re-evaluates `F = F(x)` every step at the current estimate (unlike KF).
- Threads the dataset timestamp: `x_pred = self._model.f(x, float(timestamps[t]))`
  — required for `nonlinear`'s `8 cos(1.2 t)` forcing term to take effect.
- Optional numba path (`ekf_loop_batch`) driven by the level's `@njit` dynamics
  when `use_numba=True` and `FilterModel.numba` is present; pure-NumPy otherwise.
  Unlike KF/UKF there is no linear-only fast path, so no `assert_linear_model`
  guard is needed.
- Same `_bound_cov` `±1e12` covariance ceiling as KF.

## classical/ukf.py — `UKFEstimator`

Sigma-point (unscented) filter — propagates a deterministic set of points
through the *true* nonlinear `f`/`h` instead of linearizing, capturing
second-order curvature EKF misses.

```
λ = α²(nx + κ) - nx
χ₀ = x,  χᵢ = x ± [√((nx+λ)P)]ᵢ              i = 1..2nx
Wm₀ = λ/(nx+λ),  Wmᵢ = 1/(2(nx+λ))
Wc₀ = Wm₀ + (1 - α² + β),  Wcᵢ = Wmᵢ
predict:  χ' = f(χ, t),  x⁻ = Σ Wm·χ',  P⁻ = Q + Σ Wc·(χ'-x⁻)(χ'-x⁻)ᵀ
          γ  = h(χ'),    y⁻ = Σ Wm·γ
update:   S = R + Σ Wc·(γ-y⁻)(γ-y⁻)ᵀ,   Pxy = Σ Wc·(χ'-x⁻)(γ-y⁻)ᵀ
          K = Pxy S⁻¹,  x = x⁻ + K(y - y⁻),  P = P⁻ - K S Kᵀ
```

- Defaults `alpha=1e-3, beta=2.0, kappa=0.0` (standard scaled-unscented-transform
  choice for Gaussian posteriors, Wan & van der Merwe 2000); not tuned per-level.
- Threads the dataset timestamp into `f(χ, t)`.
- **PD handling:** `P` is symmetrized before every Cholesky and after every
  covariance update; on a Cholesky failure the filter retries with escalating
  jitter scaled to `trace(P)/nx` (×10 per retry, ~10 retries) and falls back to
  eigenvalue-clipping as a last resort — all to keep `P` numerically
  positive-definite despite float64 rounding drift.
- `use_numba=True` dispatches to `ukf_linear_loop`, which hardcodes `f(x)=Fx`,
  `h(x)=Hx` and is correct only for `LinearBenchmark`; it calls
  `assert_linear_model` first, raising `ValueError` on a nonlinear model. The
  general numba path (`ukf_loop`) propagates sigma points through the level's
  actual `@njit` `f`/`h` and is valid on every level.

## classical/pf.py — `ParticleFilterEstimator`

Sequential Importance Resampling (SIR / bootstrap) particle filter — the only
estimator here that makes **no Gaussian-posterior assumption**, asymptotically
correct (`M → ∞`) even on `nonlinear`'s bimodal observation model.

```
init:       x⁽ⁱ⁾₀ ~ N(x0_mean, x0_cov),  i = 1..M        (M = num_particles)
propagate:  x⁽ⁱ⁾ₜ = f(x⁽ⁱ⁾ₜ₋₁, t) + w⁽ⁱ⁾,  w⁽ⁱ⁾ ~ N(0, Q)
weight:     log w⁽ⁱ⁾ = -½(yₜ-h(x⁽ⁱ⁾ₜ))ᵀ R⁻¹ (yₜ-h(x⁽ⁱ⁾ₜ))
            log w⁽ⁱ⁾ -= max(log w);  w⁽ⁱ⁾ = exp(log w)/Σexp(log w)
estimate:   x̂ₜ = Σᵢ w⁽ⁱ⁾ x⁽ⁱ⁾ₜ                            (weighted mean)
resample:   N_eff = 1/Σ(w⁽ⁱ⁾)²;  if N_eff < resample_threshold·M:
                multinomial resample with replacement, w⁽ⁱ⁾ ← 1/M
```

- Initialises particles from `FilterModel.x0_mean`/`x0_cov` (correct for every
  level, including Lorenz's `N([0,0,25], I)`); threads the dataset timestamp into
  `f`.
- **Log-sum-exp weighting:** `log_w -= log_w.max()` before exponentiating, so far-
  off particles whose raw likelihood underflows to `0.0` don't produce a
  `0/0 = NaN` weight sum.
- **Resampling trigger:** effective sample size `N_eff = 1/Σw²`, threshold
  `resample_threshold·M` (default `0.5·M`) — resamples only when weights have
  degenerated, avoiding the extra variance of resampling every step.
- `random_seed: int = 0` constructor param seeds `np.random.default_rng`, so runs
  are reproducible. No covariance-ceiling guard is needed — a particle that
  wanders off only loses weight, it has no `P` matrix to diverge.

## neural/kalmannet.py — `KalmanNetEstimator`, `KalmanNetUncertaintyEstimator`

Learned Kalman gain (Revach et al. 2022): the process-model prediction `f`/`h`
stays the analytic benchmark model; only the **gain** `K` is learned by a GRU,
conditioned on the innovation and the previous correction.

```
predict (analytic, not learned):  x⁻ₜ = f(xₜ₋₁, t),   ŷₜ = h(x⁻ₜ)
innovation:                       eₜ = yₜ - ŷₜ
GRU input:                        [eₜ ; xₜ₋₁ - x⁻ₜ₋₁]
GRU output:                       Kₜ ∈ ℝ^(nx×ny)  (learned, not Kalman-optimal)
update:                           xₜ = x⁻ₜ + Kₜ eₜ
```

- Architecture: 1-layer GRU → LayerNorm → 2-layer MLP head → flattened `K`. The
  gain head is **zero-initialized**, so an untrained network outputs `K = 0`
  exactly and `estimate()` reduces to pure process-model rollout (`xₜ = f(xₜ₋₁, t)`)
  — a checkable invariant that the GRU/gain wiring is correct.
- Training (`fit()`): Adam + `ReduceLROnPlateau(factor=0.5, patience=2,
  min_lr=1e-6)`, gradient-norm clipping via `clip_grad_norm_(params,
  grad_clip_norm)` (constructor param, default `0.5`) before each
  `optimizer.step()`, NaN/Inf losses skipped without updating weights, and the
  best validation-loss checkpoint restored at the end of training.
- `KalmanNetUncertaintyEstimator` (`_predict_log_var=True`) adds a log-variance
  head trained with Gaussian NLL instead of MSE (`gaussian_nll_loss(..., eps=1e-6)`)
  and exposes `estimate_with_uncertainty()` returning `(estimates, variance)`. In
  `EXPERIMENTAL_ESTIMATORS`, not run by default sweeps.
- `_process_model_step` round-trips every state vector through NumPy
  (`x_batch.cpu().numpy()` → Python loop calling `f` → back to a tensor) once per
  timestep, because `FilterModel.f` is a plain NumPy callable shared with the
  classical filters — this guarantees an identical process model but is the
  dominant cost of training/inference for this estimator.
- `estimate()` always runs on CPU regardless of training device, so inference-time
  runtime/latency is measured under the same hardware condition as the
  (CPU-only) classical filters.

## classical/filterpy_filters.py — `FilterpyKFEstimator`, `FilterpyEKFEstimator`, `FilterpyUKFEstimator`

Reference re-implementations of KF/EKF/UKF on top of the third-party
[`filterpy`](https://github.com/rlabbe/filterpy) library, used as an independent
cross-check against this repo's custom NumPy/Numba filters — same `FilterModel`
contract, same linearization behavior as their custom counterparts (KF
linearizes once at the origin, EKF re-linearizes every step, UKF uses
`MerweScaledSigmaPoints` with the same `alpha=1e-3, beta=2.0, kappa=0.0`). On
`LinearBenchmark`, `FilterpyKFEstimator`'s RMSE should match
`KalmanFilterEstimator`'s up to floating-point noise.

`filterpy` is imported lazily (only when one of these classes is instantiated;
`_require_filterpy()` raises a clear `ImportError` if it is missing), so the rest
of the package works without it installed. Registered in `REFERENCE_ESTIMATORS`.

## neural/neural_ode.py, neural/transformer.py

Stubs — every method raises `NotImplementedError`. Registered only in
`EXPERIMENTAL_ESTIMATORS`, never in `ESTIMATORS`, so default sweeps don't include
them. Pull them into `ESTIMATORS` once real `fit()`/`estimate()` land.

## Extending with a new estimator

1. Subclass `BaseEstimator`; classical filters take a `FilterModel` in
   `__init__` and ignore `fit()`. Neural estimators implement real `fit()`.
2. Never read `dataset.states` inside `estimate()`.
3. If adding a numba-accelerated path, keep a pure-NumPy fallback (pattern in
   `_numba_kernels.py`) so the estimator has no hard numba dependency, and
   document which benchmarks the fast path is valid for (linear-only fast paths
   must guard against silently activating on nonlinear `FilterModel`s).
4. Seed any internal `np.random.Generator`/`torch.manual_seed` from a
   constructor argument so runs are reproducible.
5. Register in `estimators/__init__.py`'s `ESTIMATORS` (`EXPERIMENTAL_ESTIMATORS`
   if opt-in only, `REFERENCE_ESTIMATORS` if it is a third-party-backed
   re-implementation of an existing estimator rather than new behavior).
