# Mathematical & Design summary

---

## 0. Architectural rules (apply throughout)

- **Fail fast and loud.** No silent fallbacks, implicit coercions, or dummy
  returns (`0.0`/`NaN`/`None` for an undefined result). Invalid input, mismatched
  shapes, or a scientifically unsound configuration raise a descriptive
  `ValueError`/`RuntimeError`/`ImportError` immediately, not a degraded number.
- **Strict numba.** The classical filters (KF/EKF/UKF) run *exclusively* on
  `@njit` kernels (plus the third-party `filterpy` reference filters) — there is
  no pure-NumPy fallback. A missing numba raises `ImportError` at import time.
- **Linear KF only on linear models.** `KalmanFilterEstimator` asserts
  `f(x)=F@x`, `h(x)=H@x` on every `estimate()` call and raises `ValueError`
  otherwise — it never falls back to an origin-linearization.

---

## 1. Generative model (shared contract)

[benchmark_levels/base.py](benchmark_levels/base.py)

> `x_t`: true latent state at time `t`. `y_t`: noisy observation at time `t`. `f`/`h`: deterministic process/observation maps. `w_t`/`v_t`: process/observation noise. `Q`/`R`: process/observation noise covariances.

```
x_{t+1} = f(x_t, t) + w_t,   w_t ~ N(0, Q)     [latent state, ground truth]
y_t     = h(x_t)    + v_t,   v_t ~ N(0, R)     [noisy observation, only estimator input]
```

- `f`/`h` in `FilterModel` are **noise-free** deterministic maps; `Q`/`R` carry all stochasticity, so the same `f`/`h` is reused for both data generation and filter prediction without double-counting noise.
- `FilterModel(f, h, F, H, Q, R, x0_mean, x0_cov, numba, torch)`:
  - `F`/`H` are exact Jacobians (constant matrices when the system is linear).
  - `x0_mean`/`x0_cov` expose the **true generative prior** so every estimator initializes identically.
  - `numba` (`NumbaDynamics`: `@njit` `f`/`h`/`F_jac`/`H_jac`) is **required** by every classical filter — there is no pure-NumPy fallback; EKF/UKF raise `ValueError` if it is `None`.
  - `torch` (`TorchDynamics`: batched torch `f`/`h`, `[B,nx]→[B,nx]/[B,ny]`) is **optional**, consumed only by KalmanNet's vectorized GPU training.
- Every state dimension has a physical name via `BenchmarkLevel.state_names` (e.g. `("x","y","z")` for Lorenz, `("theta","omega")` for pendulum) — used by `metrics.rmse.compute_rmse_per_dim` and the plots instead of a bare integer index.

---

## 2. Benchmark levels

### 2.1 linear — `LinearBenchmark` ([linear.py](benchmark_levels/linear.py))

> `x`: state vector `[position, velocity]`. `F`/`H`: linear state-transition/observation matrices. `Q`/`R`: process/observation noise covariances. `x_0`: initial state.

```
x = [position, velocity]ᵀ                     nx=2, ny=1   state_names=(position, velocity)
F = [[1, dt], [0, 1]]      H = [1, 0]
x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q),  Q = I·process_noise_var
y_t     = H x_t + v_t,   v_t ~ N(0, R),  R = I·observation_noise_var
x_0 = standard_normal(2) · sqrt(initial_state_var)
```

- Defaults: `process_noise_var=0.01, observation_noise_var=0.1, initial_state_var=1.0, dt=0.1`.
- Fully linear ⇒ **KF is the Bayes-optimal estimator here** — the only level on which `KalmanFilterEstimator` runs (it raises `ValueError` on every nonlinear level).

### 2.2 nonlinear — `NonlinearBenchmark` ([nonlinear.py](benchmark_levels/nonlinear.py))

Gordon, Salmond & Smith (1993) scalar benchmark.

```
x_{t+1} = 0.5 x_t + 25 x_t/(1+x_t²) + 8 cos(1.2 t) + w_t,  w_t ~ N(0, Q)
y_t     = x_t² / 20 + v_t,                                  v_t ~ N(0, R)
nx=1, ny=1,  state_names=(x,)  Q=R=1.0,  x_0 ~ N(0,1)
```

- `F_jac = 0.5 + 25(1-x²)/(1+x²)²`; `H_jac = x/10` — the squaring in `h` makes the observation **bimodal in sign** (±x give the same `y`), which breaks EKF/UKF's unimodal-Gaussian assumption by design and favors the particle filter.
- Time-forcing term `t` is threaded into every nonlinear-capable estimator's call to `f(x, t)` (EKF/UKF/PF, and KalmanNet's batched torch `f` during training).

### 2.3 pendulum — `PendulumBenchmark` ([pendulum.py](benchmark_levels/pendulum.py))

Euler-integrated nonlinear pendulum, angle-only sensor.

```
x = [θ, ω]ᵀ,  nx=2, ny=1   state_names=(theta, omega)
α(θ) = -(g/length) sin(θ)
x_{t+1} = [θ + ω·dt,  ω + α(θ)·dt] + w_t,  w_t ~ N(0, Q)
y_t     = θ_t + v_t,                        v_t ~ N(0, R)
θ_0 ~ U(-initial_angle_range, +initial_angle_range),  ω_0 = 0
```

- Exact Jacobian of the Euler step: `F_jac = [[1, dt], [-(g/length)cos(θ)·dt, 1]]`.
- Defaults: `process_noise_var=0.001, observation_noise_var=0.01, dt=0.05, g=9.81, length=1.0, initial_angle_range=π/4`.
- Prior moment-matching: θ uniform on `[-r, r]` has variance `r²/3`; `x0_cov = diag([r²/3, 1e-6])` (the `1e-6` keeps the deterministic `ω_0=0` diagonal entry positive-definite, not exactly zero).
- Default `π/4` keeps the system near the small-angle (linear) regime by design; widening `initial_angle_range` is the documented way to push EKF/UKF toward divergence.

### 2.4 lorenz — `LorenzBenchmark` ([lorenz.py](benchmark_levels/lorenz.py))

Lorenz-63 chaotic attractor, **RK4**-integrated for both data generation and the filter's process model.

> `x,y,z`: the three Lorenz state components (unrelated to the generic `x_t`/`y_t` state/observation notation; only `x,y` are observed). `σ,ρ,β`: Lorenz system parameters.

```
ẋ = σ(y-x),  ẏ = x(ρ-z)-y,  ż = xy-βz         nx=3, ny=2 (z unobserved)
state_names=(x, y, z)
x_{t+1} = RK4_step(x_t, dt) + w_t,   w_t ~ N(0, Q),  Q = I·0.001
y_t     = [x_t, y_t] + v_t,           v_t ~ N(0, R),  R = I·1.0
x_0 = standard_normal(3) + [0, 0, 25]
```

- Classic chaotic parameters `σ=10, ρ=28, β=8/3` — positive Lyapunov exponent, so RMSE on this level is expected to be trajectory-length- and seed-sensitive; a property of the dynamics, not a bug.
- The simulator step and `get_filter_model().f` use the **identical** 4-stage RK4 integrator, so the filter's process model matches the data generator exactly (no integration-scheme mismatch silently absorbed into "process noise").
- **Analytic Jacobian — exact RK4-step Jacobian.** `F_jac` is computed by the **chain rule through the four RK4 stages**, so it matches `f` to the same `O(dt⁴)` accuracy (verified numerically against a finite-difference Jacobian of `f` to ~1e-9). This is the standard `lorenz` registration.
- `x0_mean=[0,0,25], x0_cov=I` — matches the data generator's actual initialization.

#### `lorenz_fea` — `LorenzFEABenchmark`: the forward-Euler-approximation baseline

Identical data generation and `f`/`h` to `LorenzBenchmark`. The **only**
difference: `F_jac(x) = I + dt·J(x)` — the first-order forward-Euler
linearization of the continuous flow, *not* the Jacobian of the RK4 map. Since
`f` is `O(dt⁴)` while this `F` is `O(dt)`, the mean and covariance are
integration-order **inconsistent**. Registered as `lorenz_fea` and retained
*only* so the cost of that inconsistency can be measured against the standard
`lorenz`; prefer `lorenz` for any real filter comparison.

**Overflow guard (state clipping, not noise):** `f` clips its input to `±1e3`
before integrating, mirrored in both the numba and torch dynamics, for both the
standard and FEA Jacobian builders. The true attractor lives within roughly
`[-20,20]×[-25,25]×[0,50]`; `1e3` is ~20–50× that extent, so this bound never
touches legitimate trajectories. It exists purely to stop a *diverging filter
estimate* from producing a state so large that `xy`/`xz` terms in the RK4
derivative overflow float64 to `inf`/`NaN`, which would otherwise poison every
subsequent step of that trajectory irrecoverably. This is a guard on the
**filter's internal estimate**, not the data-generating simulator (which has no
such clip — ground truth is never altered).

---

## 3. Classical estimators

All consume a `FilterModel`; benchmark-agnostic by construction (swap the model, not the filter code). **Strict numba:** every classical filter runs exclusively on its `@njit` kernel in `_numba_kernels.py` — there is no pure-NumPy fallback, and a missing numba raises `ImportError` at import time.

### 3.1 KF — `KalmanFilterEstimator` ([kf.py](estimators/classical/kf.py))

> `x`: state estimate, `x⁻`: predicted (prior) state estimate. `P`: state covariance, `P⁻`: predicted (prior) covariance. `K`: Kalman gain. `S`: innovation covariance. `y`: observation.

```
predict:  x⁻ = F x,        P⁻ = F P Fᵀ + Q
update:   S  = H P⁻ Hᵀ + R
          K  = P⁻ Hᵀ S⁻¹
          x  = x⁻ + K(y - H x⁻)
          P  = (I - K H) P⁻
```
[_numba_kernels.py](estimators/classical/_numba_kernels.py) (`kf_loop_batch`, the sole implementation).

- Optimal only when `f(x)=Fx, h(x)=Hx` exactly (true for `LinearBenchmark`).
- **Strict linear guard, on every call.** `estimate()` always calls `assert_linear_model(f, h, F, H, nx, ny)`, which validates `F`/`H` shapes and probes `f`/`h` against `F@x`/`H@x` at the origin, the basis directions, and several random points. A mismatch raises `ValueError` — KF refuses to run on `pendulum`/`nonlinear`/`lorenz` rather than silently linearizing at the origin. There is no `use_numba` flag and no unguarded fallback path; this is the only way `estimate()` runs.

### 3.2 EKF — `EKFEstimator` ([ekf.py](estimators/classical/ekf.py))

> `x`: state estimate, `x⁻`: predicted state estimate. `F`/`H`: Jacobians of `f`/`h`, evaluated at the current estimate (not constant, unlike KF). `P`/`P⁻`: covariance/predicted covariance. `K`: Kalman gain.

```
predict:  x⁻ = f(x, t),     F = ∂f/∂x|_x        P⁻ = F P Fᵀ + Q
update:   H  = ∂h/∂x|_x⁻
          S  = H P⁻ Hᵀ + R,  K = P⁻ Hᵀ S⁻¹
          x  = x⁻ + K(y - h(x⁻))
          P  = (I - K H) P⁻
```
[_numba_kernels.py](estimators/classical/_numba_kernels.py) (`ekf_loop_batch`, the sole implementation — driven by `FilterModel.numba`, required).

- General — works on any `FilterModel` with numba dynamics; `estimate()` raises `ValueError` if `FilterModel.numba` is `None`.
- Time threaded explicitly: `f(x, timestamps[t])` — required for `nonlinear`'s `8 cos(1.2 t)` forcing term to take effect.
- On Lorenz, EKF now uses the exact RK4-step Jacobian (§2.4) — its covariance propagation is integration-order consistent with its RK4 mean propagation.

### 3.3 UKF — `UKFEstimator` ([ukf.py](estimators/classical/ukf.py))

Sigma-point filter — propagates a deterministic set of points through the *true* nonlinear `f`/`h` instead of linearizing, so it captures second-order curvature effects EKF misses.

> `χᵢ`: sigma points (deterministic samples around `x`). `Wm`/`Wc`: mean/covariance weights for combining sigma points. `α,β,κ,λ`: unscented-transform tuning parameters. `χ'`/`γ`: sigma points propagated through `f`/`h`. `Pxy`: cross-covariance between state and predicted observation.

```
λ = α²(nx+κ) - nx
χ₀ = x,  χᵢ = x ± [√((nx+λ)P)]ᵢ                  i=1..2nx
Wm₀ = λ/(nx+λ),  Wc₀ = Wm₀ + (1-α²+β),  Wmᵢ=Wcᵢ=1/(2(nx+λ))
predict:  χ' = f(χ),  x⁻ = ΣWm·χ',  P⁻ = Q + ΣWc·(χ'-x⁻)(χ'-x⁻)ᵀ
          γ = h(χ'), y⁻ = ΣWm·γ
update:   S = R + ΣWc·(γ-y⁻)(γ-y⁻)ᵀ,  Pxy = ΣWc·(χ'-x⁻)(γ-y⁻)ᵀ
          K = Pxy S⁻¹,  x = x⁻+K(y-y⁻),  P = P⁻-K S Kᵀ
```
[_numba_kernels.py](estimators/classical/_numba_kernels.py) (`ukf_loop_batch`, the sole implementation — general nonlinear path, propagates sigma points through `FilterModel.numba`'s `f`/`h`; required).

- Defaults `alpha=1e-3, beta=2.0, kappa=0.0` — standard scaled-unscented-transform choice for Gaussian posteriors (Wan & van der Merwe 2000); not tuned per-level.
- No linear-only fast path remains (`ukf_linear_loop`/`ukf_sigma_points` removed) — every level, including `LinearBenchmark`, runs the same general sigma-point kernel.
- **PD-matrix handling** (covariance losing positive-definiteness from accumulated floating-point error): `P` is symmetrized before every Cholesky and after every covariance update; on a Cholesky failure, an escalating jitter scaled to `trace(P)/nx` is retried (×10 per attempt, ~10 retries), with eigenvalue-clipping as the final, guaranteed-PD fallback.

### 3.4 PF — `ParticleFilterEstimator` ([pf.py](estimators/classical/pf.py))

Sequential Importance Resampling (bootstrap particle filter) — the only estimator here making **no Gaussian-posterior assumption**, asymptotically correct (`M→∞`) even on `nonlinear`'s bimodal observation model. Pure NumPy throughout — this is its sole implementation (there is no numba kernel to fall back from; PF is not subject to the "no NumPy fallback" rule because it has no numba alternative).

> `x⁽ⁱ⁾ₜ`: state of particle `i` at time `t`. `M`: number of particles. `w⁽ⁱ⁾`: normalized weight of particle `i`. `N_eff`: effective sample size.

```
init:       x⁽ⁱ⁾₀ ~ N(x0_mean, x0_cov),  i=1..M
propagate:  x⁽ⁱ⁾ₜ = f(x⁽ⁱ⁾ₜ₋₁, t) + w⁽ⁱ⁾,  w⁽ⁱ⁾ ~ N(0,Q)
weight:     log w⁽ⁱ⁾ = -½(yₜ-h(x⁽ⁱ⁾ₜ))ᵀR⁻¹(yₜ-h(x⁽ⁱ⁾ₜ))      [unnormalized log-weight]
            log w⁽ⁱ⁾ -= max(log w)                            [log-sum-exp shift]
            w⁽ⁱ⁾ = exp(log w⁽ⁱ⁾) / Σexp(log w)                [normalize]
estimate:   x̂ₜ = Σᵢ w⁽ⁱ⁾ x⁽ⁱ⁾ₜ                                  [weighted mean]
resample:   N_eff = 1/Σ(w⁽ⁱ⁾)²;  if N_eff < resample_threshold·M:
                multinomial resample with replacement, w⁽ⁱ⁾ ← 1/M
```

- **Log-sum-exp weighting:** `log_w -= log_w.max()` before exponentiating, so a far-off particle's likelihood underflowing to `0.0` cannot make the weight sum `0/0 = NaN`.
- **Resampling trigger:** `N_eff = 1/Σw²`, threshold `resample_threshold·M` (default `0.5·M`).
- **Reproducibility:** `rng = np.random.default_rng(self._random_seed)`, default `random_seed=0`.
- No covariance-ceiling guard needed/present — PF has no `P` matrix to diverge; a particle that wanders off only loses weight.

---

## 4. Neural estimator — KalmanNet ([kalmannet.py](estimators/neural/kalmannet.py))

Learned Kalman gain (Revach et al. 2022): analytic process/observation model stays fixed; only the **gain matrix** is learned.

> `x⁻ₜ`: predicted state at time `t`. `ŷₜ`: predicted observation. `eₜ`: innovation (observation residual). `Kₜ`: learned (not analytically optimal) gain matrix output by the GRU.

```
predict (analytic, not learned):  x⁻ₜ = f(xₜ₋₁, t),   ŷₜ = h(x⁻ₜ)
innovation:                       eₜ = yₜ - ŷₜ
GRU input:                        [eₜ ; xₜ₋₁ - x⁻ₜ₋₁]
GRU output:                       Kₜ ∈ ℝ^(nx×ny)         (learned, not Kalman-optimal)
update:                           xₜ = x⁻ₜ + Kₜ·eₜ
```

- Architecture: 1-layer GRU → LayerNorm → 2-layer MLP head → flattened `K`.
- **Zero-initialized gain head:** an untrained network outputs `K=0` exactly, so `estimate()` on an unfit model reduces to pure process-model rollout (`x_t = f(x_{t-1}, t)`) — a checkable invariant that the GRU/gain wiring is correct.
- **Uncertainty variant** (`KalmanNetUncertaintyEstimator`, `_predict_log_var=True`): adds a log-variance head, trained with Gaussian NLL instead of MSE (`gaussian_nll_loss(..., eps=1e-6)`); exposes `estimate_with_uncertainty()` returning `(estimates, variance)`.
- **Training-time stability control:** gradient-norm clipping (`clip_grad_norm_`, default `0.5`) before every optimizer step; NaN/Inf losses are skipped without updating weights; the best validation-loss checkpoint is restored at the end of training.

### Hardware-specific execution (the deliberate GPU/CPU split)

- **`fit()`/validation: fully vectorized, batched, on the GPU.** `_run_sequence_vectorized` uses `FilterModel.torch` — batched torch `f`/`h` (`[B,nx]→[B,nx]/[B,ny]`) built per level in `_torch_dynamics.py`. Every timestep is a *single on-device tensor op*; there is no per-row Python loop and no NumPy round-trip. The only remaining loop is over `T`, the GRU's intrinsic time recurrence (each step depends on the previous corrected state) — that cannot be removed for a sequential filter. Raises `ValueError` if the level provides no `FilterModel.torch`.
- **`estimate()`/`estimate_with_uncertainty()`: strictly sequential, on the CPU.** `_run_sequence_sequential_cpu` processes one trajectory at a time, one timestep at a time, calling the benchmark's plain NumPy `f`/`h` on a single state vector. This deliberately simulates microprocessor/embedded deployment, so test-time latency is measured under that condition rather than GPU batch throughput.
- This replaces the old design, where the same per-row NumPy round-trip (`_process_model_step`) ran during *both* training and inference — that accidentally made training Python-bound and slow, while giving inference numbers that were "CPU-only" only by side effect, not intent.

`neural_ode.py`, `transformer.py` are stubs — every method raises `NotImplementedError`; excluded from default `ESTIMATORS`, present only in `EXPERIMENTAL_ESTIMATORS`.

---

## 5. Metrics ([metrics/](metrics/))

> `x̂`: estimated state, `x`: true state, `P`: filter's reported covariance. `N`: number of trajectories, `T`: number of timesteps, `nx`: state dimension. `s`: elapsed seconds, `n`: number of steps.

```
compute_rmse_per_dim(x̂, x, names) = {name_i: sqrt(mean((x̂ᵢ-xᵢ)², axis=(0,1))) for i, name_i}
compute_rmse_per_timestep(x̂, x)   = sqrt(mean((x̂-x)², axis=(0,2)))            → array [T]
compute_nees(x̂, x, P)             = mean over [N,T] of (x-x̂)ᵀP⁻¹(x-x̂)         consistent ⇒ ≈ nx
compute_nees_chi2_bounds(nx,n,c)  = chi2 acceptance interval on mean NEES (requires scipy)
compute_nll(x̂, x, P)              = mean over [N,T] of ½[(x-x̂)ᵀP⁻¹(x-x̂) + ln det(2πP)]
runtime_per_step_ms(s, n)         = (s/n) * 1000   (raises ValueError if n<=0)
latency_ms_per_step(s,N,T)        = runtime_per_step_ms(s, N*T)
measure_memory()                  = raises NotImplementedError (disabled)
```

- **No pooled scalar RMSE.** The old `compute_rmse` (`sqrt(mean((x̂-x)²))` over *all* of `[N,T,nx]` at once) has been **deleted**. Pooling state dimensions of different physical units/scales into one number is dominated by the largest-magnitude dimension — scientifically unsound, within a benchmark or across benchmarks. `compute_rmse_per_dim` is the primary reported metric and is keyed by `BenchmarkLevel.state_names`; it raises `ValueError` if the name list's length doesn't match `nx`.
- **Uncertainty scoring (new).** `compute_nees`/`compute_nll` score the filter's *covariance*, which RMSE never touched. Both require the full `[N,T,nx,nx]` posterior and raise `ValueError` on a non-positive-definite `P` rather than silently skipping it.
- **Memory is disabled, not approximated.** `measure_memory()` raises `NotImplementedError("Memory measurement is currently unsupported.")`; the previous whole-process-RSS number (via `psutil`) was a constant baseline plus noise, not a per-estimator footprint, so it was removed rather than left to mislead.
- `runtime_per_step_ms` raises on `n<=0` instead of returning `0.0` (an undefined latency must not read as "infinitely fast").

---

## 6. Cross-cutting numerical-stability pattern (summary)

Independent overflow/convergence problems handled by matched fixes — **bound the
quantity that can blow up, but never replace a finite-but-bad estimate with a
crash** — except where the "fail fast and loud" rule explicitly calls for a
crash instead (an invalid configuration, not a numerically-rough-but-valid one):

| Failure mode | Where | Guard | Bound |
|---|---|---|---|
| `P` loses PD-ness from float rounding → Cholesky fails | UKF (sigma points / `_robust_chol`) | symmetrize, then escalating jitter scaled to `trace(P)/nx`, eigval-clip as last resort | starts `1e-9·scale`, ×10 per retry, 10 retries |
| Diverging state estimate → RK4 derivative overflows | Lorenz `f` (filter only, not the simulator) | `np.clip(x, -1e3, 1e3)` before integrating | `±1e3` (≈20–50× attractor extent) |
| Particle weights underflow to 0/0 = NaN | PF weighting | log-sum-exp shift before `exp` | exact, not a tunable bound |
| **Nonlinear model passed to the linear KF** | `KalmanFilterEstimator.estimate()` | **`assert_linear_model` raises `ValueError`** | crashes — no numeric bound; this configuration is invalid, not numerically rough |
| **Numba missing** | `_numba_kernels.py`, `_numba_dynamics.py` | **`ImportError` at import time** | crashes — there is no fallback to bound toward |

The first three rows are guards on a filter's own internal state estimate during
an otherwise-valid run (none alter the data-generating simulators, none alter
`Q`/`R`). The last two rows are deliberately *not* numerically bounded: per the
fail-fast rule, an invalid configuration (wrong filter for the model, missing
hard dependency) must crash immediately rather than produce a number that looks
plausible but isn't. The KF's old `_bound_cov` `±1e12` covariance ceiling — which
previously let a misconfigured linear-KF-on-nonlinear-data run silently produce
a "legitimately bad but numeric" RMSE — has been removed along with the
pure-NumPy fallback it lived in; that failure mode is now refused outright.
