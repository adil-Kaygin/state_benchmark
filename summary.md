# Mathematical & Design summary

---

## 0. Architectural rules (apply throughout)

- **Fail fast and loud.** No silent fallbacks, implicit coercions, or dummy
  returns (`0.0`/`NaN`/`None` for an undefined result). Invalid input, mismatched
  shapes, or a scientifically unsound configuration raise a descriptive
  `ValueError`/`RuntimeError`/`ImportError` immediately, not a degraded number.
- **Strict numba.** The classical filters (KF/EKF/UKF) run *exclusively* on
  `@njit` kernels (plus the third-party `torch-kf`/`torchfilter` reference filters) — there is
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
- `FilterModel(f, h, F, H, Q, R, x0_mean, x0_cov, numba, torch, angular_obs_mask)`:
  - `F`/`H` are exact Jacobians (constant matrices when the system is linear).
  - `x0_mean`/`x0_cov` expose the **true generative prior** so every estimator initializes identically.
  - `numba` (`NumbaDynamics`: `@njit` `f`/`h`/`F_jac`/`H_jac`) is **required** by every classical filter — there is no pure-NumPy fallback; EKF/UKF raise `ValueError` if it is `None`.
  - `torch` (`TorchDynamics`: batched torch `f`/`h`, `[B,nx]→[B,nx]/[B,ny]`) is **optional**, consumed by KalmanNet **and the four newer neural filters** (Neural-ODE, PINN, Transformer, Mamba) for vectorized GPU training. It also carries a `time_invariant: bool` flag (§7) marking levels whose `f`/`h` ignore the scalar timestep `t`, so the teacher-forced precompute can flatten `[B,T]→[B·T]` and call `f`/`h` once.
  - `angular_obs_mask` (optional boolean `[ny]`, default `None`) marks which **observation components are angles** whose innovation `y - h(x)` must be wrapped to `(-π, π]` (§2.5 vehicle-tracking bearings). `None`/all-`False` (every scalar-observation level) means no wrapping.
- Every state dimension has a physical name via `BenchmarkLevel.state_names` (e.g. `("x","y","z")` for Lorenz, `("theta","omega")` for pendulum, `("px","py","vx","vy")` for vehicle tracking) — used by `metrics.rmse.compute_rmse_per_dim` and the plots instead of a bare integer index.

---

## 2. Benchmark levels

### 2.1 linear — `LinearBenchmark` ([linear.py](benchmark_levels/linear.py))

> `x`: state vector `[position, velocity]`. `F`/`H`: linear state-transition/observation matrices. `Q`/`R`: process/observation noise covariances. `x_0`: initial state.

```
x = [position, velocity]ᵀ                     nx=2, ny=1   state_names=(position, velocity)
F = [[1, dt], [0, 1]]      H = [1, 0]
x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q),  Q = I·process_noise_var
y_t     = H x_t + v_t,   v_t ~ N(0, R),  R = I·observation_noise_var
x_0 ~ U(-a, +a),  a = sqrt(3·initial_state_var)   (per component)
```

- Defaults: `process_noise_var=0.01, observation_noise_var=0.1, initial_state_var=1.0, dt=0.1`.
- **Uniform initial-condition box.** `x_0` is drawn from a uniform box `U(-a, a)` with `a = sqrt(3·initial_state_var)`, chosen so `Var[U(-a,a)] = a²/3 = initial_state_var` — a wider, more even coverage of the start region for the data-driven models than a Gaussian blob, while the filter's `x0_mean=0`, `x0_cov=I·initial_state_var` still matches the box's mean/variance exactly.
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
x_0 ~ U(center ± half),  center = [0, 0, 25],  half = [8, 8, 8]
```

- Classic chaotic parameters `σ=10, ρ=28, β=8/3` — positive Lyapunov exponent, so RMSE on this level is expected to be trajectory-length- and seed-sensitive; a property of the dynamics, not a bug.
- The simulator step and `get_filter_model().f` use the **identical** 4-stage RK4 integrator, so the filter's process model matches the data generator exactly (no integration-scheme mismatch silently absorbed into "process noise").
- **Analytic Jacobian — exact RK4-step Jacobian.** `F_jac` is computed by the **chain rule through the four RK4 stages**, so it matches `f` to the same `O(dt⁴)` accuracy (verified numerically against a finite-difference Jacobian of `f` to ~1e-9). This is the standard `lorenz` registration.
- **Uniform initial-condition box.** `x_0 ~ U(center ± half)` with `center=[0,0,25]`, `half=[8,8,8]` — even coverage of the start region for the data-driven models. The filter's `x0_mean=[0,0,25]`, `x0_cov=diag(half²/3)=diag(64/3)≈diag(21.33)` matches that box's per-axis mean/variance exactly.

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

### 2.5 vehicle_tracking — `VehicleTrackingBenchmark` ([vehicle_tracking.py](benchmark_levels/vehicle_tracking.py))

Multi-sensor **range/bearing** target tracking — the canonical radar/sonar
fusion problem, and the first level whose difficulty comes from the *measurement*
side (multiple heterogeneous sensors, a nonlinear polar readout) rather than the
dynamics. The state moves under a **linear constant-velocity (CV)** model, so the
only new difficulty is the sensors.

> `px,py`: Cartesian position. `vx,vy`: Cartesian velocity. `K`: number of sensor stations at fixed positions `(sxₖ,syₖ)`. `range/bearing`: polar readout of the target relative to each sensor.

```
x = [px, py, vx, vy]ᵀ                          nx=4, ny=2K   state_names=(px, py, vx, vy)
f(x) = [px+vx·dt, py+vy·dt, vx, vy]            (linear ⇒ F is the constant CV matrix)
Q = process_noise_intensity · DWNA(dt)         (per-axis discrete white-noise-acceleration)
h(x)[2k]   = sqrt((px-sxₖ)² + (py-syₖ)²)       (range,   sensor k)
h(x)[2k+1] = atan2(py-syₖ, px-sxₖ)             (bearing, sensor k; wrapped to (-π,π])
R = blkdiag( diag(σ_r,k², σ_b,k²)  for k )     (block-diagonal [2K,2K])
```

- **Multiple noise levels (heterogeneous fusion).** The per-sensor `(σ_r,k, σ_b,k)` differ (e.g. defaults `K=3`, `σ_r=(0.5, 2.0, 5.0) m`, `σ_b=(0.5°, 2°, 5°)`) and a global `noise_scale` multiplies all of them for sweeps — so the filter must **weight a precise sensor over the noisy ones**, which none of the other levels test. Sensors are placed around the scene box so the observing geometry (GDOP) varies along the trajectory.
- **Analytic Jacobian `H` `[2K,4]`** stacks the per-sensor `[2,4]` blocks: `∂range/∂(px,py)=[dx/r, dy/r]`, `∂bearing/∂(px,py)=[-dy/r², dx/r²]` (velocity columns zero; `r` floored at `1e-9` to guard a target passing exactly over a sensor). `F` is the constant CV matrix.
- **Bearing angle-wrap (the polar-measurement footgun).** Every innovation `y - h(x)` with a bearing component is wrapped to `(-π,π]` via `atan2(sin·, cos·)` — otherwise a residual near the `±π` branch cut is ~2π wrong and silently wrecks the gain. This is signalled by `FilterModel.angular_obs_mask` (odd indices = bearings) and honored in the EKF/UKF `@njit` kernels, the torchfilter EKF/UKF references, and **every neural innovation feature** (KalmanNet/Transformer/Mamba). `h` itself returns bearings already on `(-π,π]`; wrapping is only at the innovation site, never inside `h`.
- **Defaults:** `trajectory_length=200, num_trajectories=2000, dt=0.1, num_sensors=3, process_noise_intensity=0.1, dropout_prob=0.0`. Prior: `x0_mean=[½scene, ½scene, 0, 0]`, `x0_cov=diag([initial_state_var, initial_state_var, vel_var, vel_var])`.
- **Optional sensor dropout** (`dropout_prob`, **OFF by default**): when `>0`, a sensor's range+bearing slot is emitted as **`NaN`** (never `0.0` — a fabricated zero is a silent wrong measurement) with that per-step, per-sensor probability, and the metadata/docstring say so loudly. Off by default so EKF/UKF and every current estimator run unmodified; NaN-gating estimators are a follow-up.
- Ground truth (Cartesian states) is **never** noised, clipped, or dropped — only `observations` carry noise/dropout. numpy `f`/`h`, `@njit`, and batched-torch dynamics are **mathematically identical** (verified: `f`/`h`/`F`/`H` agree to 0 across paths; analytic `H` matches a finite-difference Jacobian to ~1e-8).
- A coordinated-turn variant (state `[px,py,vx,vy,ω]`, nonlinear `f`) is noted as a follow-up but out of scope — `f` is kept linear so the sensors are the only new difficulty.

---

## 3. Classical estimators

All consume a `FilterModel`; benchmark-agnostic by construction (swap the model, not the filter code). **Strict numba:** every classical filter runs exclusively on its `@njit` kernel in `_numba_kernels.py` — there is no pure-NumPy fallback, and a missing numba raises `ImportError` at import time.

**Reported covariance (`returns_covariance=True`).** KF/EKF/UKF (and the three `torch-kf`/`torchfilter` reference variants) already propagate a posterior `P` every step and now **expose** it: they set `returns_covariance=True` and implement `estimate_with_covariance(dataset) → (estimates [N,T,nx], covariances [N,T,nx,nx])` via `*_loop_batch_cov` kernels that mirror the point-estimate loops one-for-one plus `covs[t]=P`. `estimate()` is unchanged (point estimates only). This is what feeds the NEES/NLL consistency metrics (§5); a filter with no calibrated `P` (the neural filters, PF) leaves `returns_covariance=False` and `estimate_with_covariance` raises `NotImplementedError` rather than fabricating a `P`.

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

## 4. Neural estimators

Five learned filters, all sharing the **GPU-train / CPU-infer hardware split**
(§4.6) and the `BaseEstimator` interface, so `experiments/runner.py` and the
notebooks swap them in without special-casing. KalmanNet is the canonical
template (§4.1); the other four (Neural-ODE, PINN, Transformer, Mamba) subclass a
shared `SequentialNeuralFilter` scaffold ([_neural_base.py](estimators/neural/_neural_base.py))
that factors out the per-epoch train/val loop, best-checkpoint-by-val-loss,
gradient clipping, LR scheduler, early stopping, NaN/Inf-loss skip, seeding, and
the `save()`/`load_weights()` recipe. All are registered in the default
`ESTIMATORS` and run in the standard sweep alongside the classical filters.

### 4.1 KalmanNet ([kalmannet.py](estimators/neural/kalmannet.py))

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

**KalmanNet execution.** `fit()`/validation is fully vectorized/batched on the GPU: `_run_sequence_vectorized` uses `FilterModel.torch` (batched `f`/`h`, `[B,nx]→[B,nx]/[B,ny]`); the only loop is over `T` (the GRU's intrinsic time recurrence, unavoidable for a sequential filter). `estimate()`/`estimate_with_uncertainty()` runs strictly sequentially on the CPU with the plain NumPy `f`/`h`, one trajectory / one timestep at a time (embedded-deployment latency). Raises `ValueError` if the level provides no `FilterModel.torch`.

**Two-phase curriculum (speed hack for the recurrence).** KalmanNet is an inherently sequential GRU recurrence, so its free-running training loop cannot be parallelized over `T`. `curriculum_epochs > 0` runs a **Phase-1 teacher-forced** warm-start (inputs built from the ground-truth previous state ⇒ the whole `[B,T,*]` sequence runs in one parallel GRU call) before annealing into the **Phase-2 free-running** objective (`_run_sequence_vectorized`, self-propagated state) that matches deployment; `curriculum_epochs=0` is free-running only. Phase 2 can run on CPU (`phase2_device=None`) where the launch-bound `T`-loop is faster.

### 4.2 Neural-ODE ([neural_ode.py](estimators/neural/neural_ode.py))

Continuous-time filter. Between observations the latent state is propagated by a **learned ODE** `dx/dt = g_θ(x, t)` integrated over each inter-observation `dt` with a **dependency-free explicit RK4** (`n_substeps` fixed steps); at each observation a learned innovation-driven correction is applied — a continuous-discrete EKF with both drift and correction learned.

```
x_pred(t_k) = x(t_{k-1}) + ∫_{t_{k-1}}^{t_k} g_θ(x, t) dt      (RK4, n_substeps)
innov       = y_k - h(x_pred(t_k))                             (h = true filter_model.h)
x_post(t_k) = x_pred(t_k) + c_φ([innov, x_pred])
```

- **Process-model usage:** the true `h` always forms the innovation; the learned drift `g_θ` **replaces** `f` by default, or with `use_model_drift=True` is learned as a **residual** on top of the known dynamics.
- **No teacher-forcing curriculum** — the forward is already free-running (it carries its own `x_post` forward through the learned RK4), so training input = inference input.
- **Integrator:** default `solver="rk4"` (plain-PyTorch, differentiable); `solver="dopri5"` uses `torchdiffeq` if importable, else a clear `ImportError` (never a silent fallback).

### 4.3 PINN — Physics-Informed filter ([pinn.py](estimators/neural/pinn.py))

A causal GRU maps innovation features to a state estimate; the training loss adds **residual terms that force the estimate to obey the benchmark's known `f`/`h`** (same `f`/`h` EKF gets — no model mismatch), not just state-MSE.

```
r_data = x̂_t - x_t                         (supervised state error)
r_dyn  = x̂_t - f(x̂_{t-1}, t-1)             (process-model consistency)
r_meas = y_t - h(x̂_t, t)                    (observation consistency, self-supervised — uses only y)
loss   = ‖r_data‖² + λ_dyn‖r_dyn‖² + λ_meas‖r_meas‖²
```

- Both `f` and `h` are used in the loss (GPU, batched, differentiable); `filter_model.torch` is **required** for `fit()`. The forward also uses `f`/`h` to build the innovation conditioning, like KalmanNet.
- **No teacher-forcing curriculum** — `_forward_train` is already free-running (feeds `x = x_pred + dx` into the next `f(x)`). Setting `λ_dyn = λ_meas = 0` recovers a plain supervised filter (the ablation baseline).

### 4.4 Transformer ([transformer.py](estimators/neural/transformer.py))

Causal (decoder-style) Transformer mapping the observation sequence `y_{1:t}` to `x̂_t` — self-attention gives each estimate explicit long-range access to the past window, a different inductive bias from a recurrent filter. A causal mask enforces `x̂_t` depends only on `y_{≤t}` (a filter, not a smoother); built from plain PyTorch (`nn.TransformerEncoder` + causal mask), **no third-party dependency**.

- **Process-model usage:** with `use_innovation_features=True` (default) the per-step input is `[y_t, innovation_t, x_pred_t, dt_t]` where `x_pred=f(x_prev)`, `innovation=y-h(x_pred)` (requires `filter_model.torch`); with `False` it is a pure black-box on `[y, dt]`.
- **Training is parallel over `T`** — one masked forward computes `x̂_t` for all `t`. `estimate()` runs causally on the CPU: for each `t` the model is re-run on the prefix `y_{1:t}` (no KV cache — the honest deployment cost of an attention model).

### 4.5 Mamba — selective SSM ([mamba.py](estimators/neural/mamba.py))

A learned linear state-space recursion with input-dependent (selective) `A,B,C,Δ` — the natural generalization of the Kalman recursion. It trains in parallel via an **associative selective scan** (numerically-stable log-space Heinsen scan) and runs as a cheap constant-memory `O(T)` linear recurrence at inference (likely the cheapest learned filter at deployment).

- **Dependency-free:** a from-scratch selective SSM in plain PyTorch — a parallel scan for training and a plain `O(T)` recurrence (`step`) for inference, numerically the same math. There is **no `mamba_ssm` requirement** (the flag `use_mamba_ssm_kernels` is honored only if the CUDA kernels happen to be importable).
- Same innovation-feature / black-box options as the Transformer; same GPU-train (parallel scan) / CPU-infer (`O(T)` recurrence) split.

### 4.6 Exposure-bias curriculum (Transformer & Mamba)

The Transformer and Mamba parallelize over `T` only when the input is built from the **ground-truth** previous state (teacher forcing). But at deployment the same input is built from the model's **own** previous estimate, so a teacher-forced-only model suffers **exposure bias** — its errors compound on a distribution it never trained on (worst on chaotic/weakly-observable levels). Both estimators take a `curriculum_epochs` knob: `>0` adds a **free-running fine-tune** for the trailing epochs — a sequential `T`-loop (shared `SequentialNeuralFilter._forward_free_running`) that feeds the model's own `x̂_{t-1}` into `f`/`h` **exactly as the CPU inference path does**, so training matches deployment. `curriculum_epochs=0` keeps the fast teacher-forced-only path; it is a no-op with `use_innovation_features=False` (no state is fed back, so no exposure bias). KalmanNet/PINN/Neural-ODE are already free-running/curriculum'd and are unchanged.

---

## 5. Metrics ([metrics/](metrics/))

> `x̂`: estimated state, `x`: true state, `P`: filter's reported covariance. `N`: number of trajectories, `T`: number of timesteps, `nx`: state dimension. `s`: elapsed seconds, `n`: number of steps.

```
compute_rmse_per_dim(x̂, x, names) = {name_i: sqrt(mean((x̂ᵢ-xᵢ)², axis=(0,1))) for i, name_i}
compute_rmse_per_timestep(x̂, x)   = sqrt(mean((x̂-x)², axis=(0,2)))            → array [T]
compute_nees(x̂, x, P)             = mean over [N,T] of (x-x̂)ᵀP⁻¹(x-x̂)         consistent ⇒ ≈ nx
compute_nees_chi2_bounds(nx,n,c)  = chi2 acceptance interval on mean NEES (requires scipy)
compute_nll(x̂, x, P)              = mean over [N,T] of ½[(x-x̂)ᵀP⁻¹(x-x̂) + ln det(2πP)]
compute_nees_per_trajectory(...)  = [N] mean NEES per trajectory (the per-traj sample for aggregation)
compute_nll_per_trajectory(...)   = [N] mean NLL  per trajectory
runtime_per_step_ms(s, n)         = (s/n) * 1000   (raises ValueError if n<=0)
latency_ms_per_step(s,N,T)        = runtime_per_step_ms(s, N*T)
measure_memory()                  = raises NotImplementedError (disabled)
```

- **No pooled scalar RMSE.** The old `compute_rmse` (`sqrt(mean((x̂-x)²))` over *all* of `[N,T,nx]` at once) has been **deleted**. Pooling state dimensions of different physical units/scales into one number is dominated by the largest-magnitude dimension — scientifically unsound, within a benchmark or across benchmarks. `compute_rmse_per_dim` is the primary reported metric and is keyed by `BenchmarkLevel.state_names`; it raises `ValueError` if the name list's length doesn't match `nx`.
- **Uncertainty scoring (wired end-to-end).** `compute_nees`/`compute_nll` score the filter's *covariance*, which RMSE never touched (RMSE ignores `P`, so an over-confident filter scores the same as a well-calibrated one). Both require the full `[N,T,nx,nx]` posterior and raise `ValueError` on a non-positive-definite `P` rather than silently skipping it. The per-trajectory variants + `aggregate_uncertainty_over_trajectories` (§5.1) give NEES/NLL the same **mean ± std / 95% CI over the test trajectories** as RMSE, from a single test set. In the experiment/notebook every `returns_covariance=True` estimator (§3) gets NEES/NLL computed via `estimate_with_covariance`, logged with the chi-square acceptance band shown next to the mean NEES (inside ⇒ consistent, above ⇒ over-confident `P`, below ⇒ under-confident); neural filters and PF are explicitly skipped, not given a fabricated `P`. Verified end-to-end: the linear KF sits at mean NEES ≈ nx inside its chi-square band.
- **Memory is disabled, not approximated.** `measure_memory()` raises `NotImplementedError("Memory measurement is currently unsupported.")`; the previous whole-process-RSS number (via `psutil`) was a constant baseline plus noise, not a per-estimator footprint, so it was removed rather than left to mislead.
- `runtime_per_step_ms` raises on `n<=0` instead of returning `0.0` (an undefined latency must not read as "infinitely fast").

### 5.1 Single-test-set statistics ([aggregate.py](metrics/aggregate.py))

Every test trajectory is an independent realization (its own uniformly-sampled `x_0`, its own process/observation noise), so the RMSE (or NEES/NLL) on one trajectory is one independent sample and the `N` trajectories of a single test set give an `N`-sample estimate of the error distribution. That yields a proper **mean ± std and 95% CI from ONE sufficiently large test set** — no Monte-Carlo seed loop / dataset regeneration is needed (the trajectory count *is* the sample size; the CI half-width `1.96·std/√N` shrinks as `1/√N`). `aggregate_rmse_per_dim_over_trajectories` and `aggregate_uncertainty_over_trajectories` turn the per-trajectory metric arrays into `{mean, std, ci95, n}` summaries the notebook tables/plots consume.

---

## 6. Cross-cutting numerical-stability pattern (summary)

Independent overflow/convergence problems handled by matched fixes — **bound the
quantity that can blow up, but never replace a finite-but-bad estimate with a
crash** — except where the "fail fast and loud" rule explicitly calls for a
crash instead (an invalid configuration, not a numerically-rough-but-valid one):

| Failure mode | Where | Guard | Bound |
|---|---|---|---|
| `P` loses PD-ness from float rounding → Cholesky fails | UKF (sigma points / `_robust_chol`) | symmetrize, then escalating jitter scaled to `trace(P)/nx`, eigval-clip as last resort | starts `1e-9·scale`, ×10 per retry, 10 retries |
| Diverging `P` → `F P Fᵀ` predict overflows to inf/NaN | KF/EKF njit loops (`_bound_cov`) | symmetrize, then clip entries to `±ceiling` each predict/update | `±1e12` |
| Diverging state estimate → RK4 derivative overflows | Lorenz `f` (filter only, not the simulator) | `np.clip(x, -1e3, 1e3)` before integrating | `±1e3` (≈20–50× attractor extent) |
| Particle weights underflow to 0/0 = NaN | PF weighting | log-sum-exp shift before `exp` | exact, not a tunable bound |
| **Nonlinear model passed to the linear KF** | `KalmanFilterEstimator.estimate()` | **`assert_linear_model` raises `ValueError`** | crashes — no numeric bound; this configuration is invalid, not numerically rough |
| **Numba missing** | `_numba_kernels.py`, `_numba_dynamics.py` | **`ImportError` at import time** | crashes — there is no fallback to bound toward |

The first four rows are guards on a filter's own internal state estimate during
an otherwise-valid run (none alter the data-generating simulators, none alter
`Q`/`R`). The last two rows are deliberately *not* numerically bounded: per the
fail-fast rule, an invalid configuration (wrong filter for the model, missing
hard dependency) must crash immediately rather than produce a number that looks
plausible but isn't. In particular the KF's old **pure-NumPy fallback path** —
which previously let a misconfigured linear-KF-on-nonlinear-data run silently
produce a "legitimately bad but numeric" RMSE — has been removed: that
configuration is now refused outright by `assert_linear_model`. The `_bound_cov`
`±1e12` ceiling itself survives in the shared `@njit` KF/EKF loops (row 2 above),
where it keeps a legitimately-diverging-but-valid run finite rather than letting
it poison later steps with inf/NaN.

---

## 7. Neural training-speed design (vectorization, exact — not approximate)

The neural filters' training cost is reduced by removing *redundant* work, never
by changing the math (every optimization below is bit-identical to the naive
path):

- **Teacher-forced feature caching.** The teacher-forced predictions
  `x_pred[:,t]=f(states[:,t-1])`, `y_pred[:,t]=h(x_pred[:,t])` are a pure function
  of `(states, timestamps, model)` — none of which change during a `fit()` — so
  they are computed **once per fit** (`precompute_teacher_forced`) and carried
  through the `DataLoader` as a third tensor (shuffled in lockstep with
  `obs`/`states`) instead of rebuilt every epoch. No gradients flow through them
  (ground-truth constants), so caching is numerically identical.
- **`time_invariant` flatten (`TorchDynamics.time_invariant`).** For a level whose
  `f`/`h` ignore the scalar `t` (linear/pendulum/lorenz — only the baked-in `dt`
  matters; `nonlinear` is the exception with `cos(1.2 t)`), the per-step
  `for t in range(T)` build collapses: reshape `x_prev` `[B,T,nx]→[B·T,nx]`, call
  `f` **once**, reshape back (one kernel-launch set instead of `T`). This is
  exactly equal because the calls are independent and `t`-invariant; it defaults
  to `False`, so a new/forgotten level takes the safe per-step path rather than
  silently feeding a single `t` to every row.
- **Device split for launch-bound loops.** The single-phase sequential nets
  (Neural-ODE/PINN) and KalmanNet's free-running Phase 2 are launch-bound on
  short/narrow levels, where a CPU can beat a GPU (no per-step kernel-launch
  overhead). This is a config lever (`device="cpu"` / `phase2_device=None`), not a
  code change — the same numbers, chosen per (level, estimator).
