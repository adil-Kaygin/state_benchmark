# estimators

State estimators consume `observations` (noisy sensor data, `y_t`) and produce
`estimates` of the latent `states` (`x_t`), which `metrics.rmse.compute_rmse_per_dim`
scores per named state variable against the ground truth written by
`benchmark_levels`. No estimator reads `dataset.states` inside `estimate()` —
that would leak the target.

```
BaseEstimator
 ├── estimator_name / estimator_type             # "classical" | "neural"
 ├── fit(train_dataset, val_dataset) -> None      # no-op for classical filters
 ├── estimate(dataset) -> ndarray [N, T, nx]       # the only inference call
 └── save(path) / load(path)                       # classical: params only,
                                                    # cannot reconstruct FilterModel
```

All classical estimators take a `FilterModel(f, h, F, H, Q, R, x0_mean, x0_cov,
numba, torch)` produced by `BenchmarkLevel.get_filter_model()` — they are
benchmark-agnostic; swapping the benchmark swaps the model, not the filter code.
Every classical estimator initialises from `FilterModel.x0_mean`/`x0_cov`
(falling back to `zeros(nx)`/`eye(nx)` only when the model leaves them `None`).
`numba` is required by every classical filter; `torch` is required only by
KalmanNet's GPU training.

**Registries** (`estimators/__init__.py`):
- `ESTIMATORS` — run by default sweeps: `ekf`, `ukf`, `pf`, `kalmannet` on every
  level, plus `kf` on `LinearBenchmark` only (`KalmanFilterEstimator` raises
  `ValueError` on a nonlinear model, so it is never swept there).
- `EXPERIMENTAL_ESTIMATORS` — opt-in: `kalmannet_uncertainty`, `neural_ode`,
  `transformer`.
- `REFERENCE_ESTIMATORS` — third-party cross-checks: `torchkf_kf`,
  `torchkf_ekf`, `torchkf_ukf`.

## classical/kf.py — `KalmanFilterEstimator`

Optimal linear-Gaussian filter; valid **only** on a linear model (`f(x)=Fx`,
`h(x)=Hx` exactly, e.g. `LinearBenchmark`).

```
predict:  x⁻ = F x,        P⁻ = F P Fᵀ + Q
update:   S  = H P⁻ Hᵀ + R
          K  = P⁻ Hᵀ S⁻¹
          x  = x⁻ + K(y - H x⁻)
          P  = (I - K H) P⁻
```

- **Strict linear check (fail fast).** Every `estimate()` calls
  `_numba_kernels.assert_linear_model`, which validates `F`/`H` shapes and probes
  `f`/`h` against `F@x`/`H@x` at the origin, the basis directions, and random
  points. A nonlinear `FilterModel` raises `ValueError` — the KF **refuses** to
  run on `pendulum`/`nonlinear`/`lorenz` rather than silently linearizing at the
  origin. Use EKF/UKF/PF there.
- `F = self._model.F(zeros(nx))`, `H = self._model.H(zeros(nx))` — for a linear
  model the Jacobian is the same constant matrix everywhere. KF is
  time-invariant; it does not thread the dataset timestamp into `f`.
- **Single code path:** the recursion runs exclusively in the numba-jitted
  `_numba_kernels.kf_loop_batch`. The pure-NumPy fallback, the `use_numba` flag,
  and the `±1e12` covariance clamp have all been removed — on a linear model the
  covariance is well-behaved, and a missing numba is an `ImportError`, not a
  silent degrade.

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
- Threads the dataset timestamp: `f(x, float(timestamps[t]))` — required for
  `nonlinear`'s `8 cos(1.2 t)` forcing term to take effect.
- Runs **exclusively** on the `@njit ekf_loop_batch` kernel driven by the level's
  `FilterModel.numba` dynamics (general nonlinear EKF, valid on every level). The
  pure-NumPy fallback and `use_numba` flag are gone; a model without
  `FilterModel.numba` raises `ValueError` (fail fast). The kernel still keeps `P`
  symmetric and bounded internally to survive a diverging chaotic trajectory.

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
- Runs **exclusively** on the general `@njit ukf_loop_batch` kernel, which
  propagates sigma points through the level's actual `FilterModel.numba` `f`/`h`
  and is valid on every level. The pure-NumPy fallback, the `use_numba` flag, and
  the linear-only fast path (`ukf_linear_loop`/`ukf_sigma_points`) have all been
  removed; a model without `FilterModel.numba` raises `ValueError`.

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
- **Hardware-specific execution (deliberate split):**
  - `fit()`/validation run **fully vectorized and batched on the GPU**
    (`_run_sequence_vectorized`). The predict step uses `FilterModel.torch`
    (batched torch `f`/`h`, `[B,nx]→[B,nx]/[B,ny]`), so every timestep is a
    single on-device tensor op — no per-row Python loop, no NumPy round-trip.
    The only remaining loop is over `T` (the GRU's intrinsic recurrence: each
    step depends on the previous corrected state). Requires
    `FilterModel.torch`; raises `ValueError` if a level provides none.
  - `estimate()`/`estimate_with_uncertainty()` run **strictly sequentially on
    the CPU** (`_run_sequence_sequential_cpu`): one trajectory at a time, one
    timestep at a time, using the NumPy `f`/`h` on a single state vector. This
    simulates microprocessor/embedded deployment and measures inference latency
    under that condition, not under GPU batch throughput.

## classical/torchkf_filters.py — `TorchKFKFEstimator`, `TorchKFEKFEstimator`, `TorchKFUKFEstimator`, `TorchKFPFEstimator`

Reference re-implementations of KF/EKF/UKF (and a PF, for future use) built on
two third-party PyTorch libraries, used as an independent cross-check against
this repo's custom Numba filters — same `FilterModel` contract, same
linearization behavior as their custom counterparts (KF linearizes once at the
origin -- valid only on a linear model, same as the strict
`KalmanFilterEstimator` -- EKF re-linearizes every step, UKF uses the Merwe
scaled sigma points with the same `alpha=1e-3, beta=2.0, kappa=0.0`):

- **KF** → [`torch-kf`](https://github.com/raphaelreme/torch-kf) (imported as
  `torch_kf`), a natively-batched **linear** Kalman `predict`/`update` over a
  `GaussianState`. Every trajectory is filtered in one batched pass.
- **EKF / UKF / PF** → [`torchfilter`](https://github.com/stanford-iprl-lab/torchfilter),
  which supplies `ExtendedKalmanFilter`, `UnscentedKalmanFilter`
  (`MerweSigmaPointStrategy`), and `ParticleFilter`. torch-kf is linear-only, so
  the nonlinear filters come from torchfilter. The dynamics/measurement models
  wrap the level's `f`/`h` and return the analytic Jacobians `F`/`H` directly
  (our `f`/`h` are NumPy, not torch-differentiable), and thread the per-step
  timestamp through a mutable `t` on the dynamics model. For angular levels the
  bearing innovation is wrapped to `(-π, π]` by substituting
  `obs' = h(x_pred) + wrap(obs - h(x_pred))` before torchfilter's update step
  (Issues 5/6).

`TorchKFKFEstimator` has no `assert_linear_model` guard of its own; only run it
on `LinearBenchmark`, where its RMSE should match `KalmanFilterEstimator`'s up to
floating-point noise. `TorchKFPFEstimator` is registered in
`EXPERIMENTAL_ESTIMATORS` (not the default sweep) and reports point estimates
only (`returns_covariance` is False), so it is excluded from the NEES/NLL table
until a particle-covariance estimate is wired up.

Both libraries are imported lazily (only when a class that needs them is
instantiated; `_require_torchkf()` / `_require_torchfilter()` raise a clear
`ImportError` if missing), so the rest of the package works without either
installed. The KF/EKF/UKF are registered in `REFERENCE_ESTIMATORS`.

## neural/neural_ode.py, neural/transformer.py

Stubs — every method raises `NotImplementedError`. Registered only in
`EXPERIMENTAL_ESTIMATORS`, never in `ESTIMATORS`, so default sweeps don't include
them. Pull them into `ESTIMATORS` once real `fit()`/`estimate()` land.

## Extending with a new estimator

1. Subclass `BaseEstimator`; classical filters take a `FilterModel` in
   `__init__` and ignore `fit()`. Neural estimators implement real `fit()`.
2. Never read `dataset.states` inside `estimate()`.
3. Classical filters run **exclusively** on `@njit` kernels (`_numba_kernels.py`)
   — there is no pure-NumPy fallback. A missing numba must raise `ImportError`
   (fail fast), not silently degrade. A linear-only fast path must call
   `assert_linear_model` and raise `ValueError` on a nonlinear `FilterModel`
   rather than silently activating on it.
4. Seed any internal `np.random.Generator`/`torch.manual_seed` from a
   constructor argument so runs are reproducible.
5. Register in `estimators/__init__.py`'s `ESTIMATORS` (`EXPERIMENTAL_ESTIMATORS`
   if opt-in only, `REFERENCE_ESTIMATORS` if it is a third-party-backed
   re-implementation of an existing estimator rather than new behavior).
