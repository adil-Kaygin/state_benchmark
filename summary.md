# Mathematical & Design summary

---

## 1. Generative model (shared contract)

[benchmark_levels/base.py:32-43](benchmark_levels/base.py#L32)

> `x_t`: true latent state at time `t`. `y_t`: noisy observation at time `t`. `f`/`h`: deterministic process/observation maps. `w_t`/`v_t`: process/observation noise. `Q`/`R`: process/observation noise covariances.

```
x_{t+1} = f(x_t, t) + w_t,   w_t ~ N(0, Q)     [latent state, ground truth]
y_t     = h(x_t)    + v_t,   v_t ~ N(0, R)     [noisy observation, only estimator input]
```

- `f`/`h` in `FilterModel` are **noise-free** deterministic maps; `Q`/`R` carry all stochasticity. Design choice: keeps "process model" and "noise model" orthogonal so the same `f`/`h` is reused for both data generation and filter prediction without double-counting noise.
- `FilterModel(f, h, F, H, Q, R, x0_mean, x0_cov, numba)` — `F`/`H` are exact Jacobians (constant matrices when the system is linear). `x0_mean`/`x0_cov` expose the **true generative prior** so every estimator initializes identically.
- `NumbaDynamics` ([base.py:11-28](benchmark_levels/base.py#L11)) mirrors `f/h/F_jac/H_jac` as `@njit` closures, used only as a speed accelerator — Python callables remain the source of truth.

**Design rationale — why `x0_mean`/`x0_cov` exist at all:** previously every classical filter hardcoded `x=0, P=I`. On `LorenzBenchmark` (true `x0=[0,0,25]`) this put estimators 25 units off at t=0; on `LinearBenchmark` it caused KF(numba) and EKF to start from *different* priors on the same data. Fixed by threading the benchmark's real prior through every estimator ([kf.py:47-48](estimators/classical/kf.py#L47), [ekf.py:51-52](estimators/classical/ekf.py#L51), [ukf.py:102-103](estimators/classical/ukf.py#L102), [pf.py:52-53](estimators/classical/pf.py#L52)).

---

## 2. Benchmark levels

### 2.1 linear — `LinearBenchmark` ([linear.py](benchmark_levels/linear.py))

> `x`: state vector `[position, velocity]`. `F`/`H`: linear state-transition/observation matrices. `Q`/`R`: process/observation noise covariances. `x_0`: initial state.

```
x = [position, velocity]ᵀ                     nx=2, ny=1
F = [[1, dt], [0, 1]]      H = [1, 0]
x_{t+1} = F x_t + w_t,   w_t ~ N(0, Q),  Q = I·process_noise_var
y_t     = H x_t + v_t,   v_t ~ N(0, R),  R = I·observation_noise_var
x_0 = standard_normal(2) · sqrt(initial_state_var)
```

- Noise injection: `LinearSimulator.step`/`observe` draw `rng.multivariate_normal(0, Q)` / `(0, R)` per call — [linear.py:35](benchmark_levels/linear.py#L35), [linear.py:39](benchmark_levels/linear.py#L39).
- Defaults: `process_noise_var=0.01, observation_noise_var=0.1, initial_state_var=1.0, dt=0.1` ([linear.py:51-53](benchmark_levels/linear.py#L51)).
- Fully linear ⇒ **KF is the Bayes-optimal estimator here** — this level exists as the sanity-check / upper-bound case other filters should match.


### 2.2 nonlinear — `NonlinearBenchmark` ([nonlinear.py](benchmark_levels/nonlinear.py))

Gordon, Salmond & Smith (1993) scalar benchmark.

> `x_t`: scalar state at time `t`. `y_t`: scalar observation at time `t`. `t`: discrete timestep, also enters as a forcing term. `w_t`/`v_t`: process/observation noise.

```
x_{t+1} = 0.5 x_t + 25 x_t/(1+x_t²) + 8 cos(1.2 t) + w_t,  w_t ~ N(0, Q)
y_t     = x_t² / 20 + v_t,                                  v_t ~ N(0, R)
nx=1, ny=1,  Q=R=1.0,  x_0 ~ N(0,1)
```

- `f` at [nonlinear.py:114-116](benchmark_levels/nonlinear.py#L114), `h` at [nonlinear.py:118-119](benchmark_levels/nonlinear.py#L118).
- `F_jac = 0.5 + 25(1-x²)/(1+x²)²` ([nonlinear.py:121-124](benchmark_levels/nonlinear.py#L121)); `H_jac = x/10` ([nonlinear.py:126-127](benchmark_levels/nonlinear.py#L126)) — the squaring in `h` makes the observation **bimodal in sign** (±x give the same `y`).
- **Design rationale — why this level exists:** it is the standard adversarial test for Gaussian-posterior filters. EKF/UKF assume unimodal local-Gaussian posteriors; squaring the state breaks that assumption, so this level is expected to favor the particle filter and expose EKF/UKF weakness by design, not by bug.
- Time-forcing term `t` is threaded into every estimator's call to `f(x, t)` — see EKF/UKF/PF/KalmanNet `estimate()` calls — otherwise the `8 cos(1.2 t)` term silently defaults `t=0` and the process model becomes time-invariant (was previously the case; fixed by passing `timestamps[t]` everywhere).

### 2.3 pendulum — `PendulumBenchmark` ([pendulum.py](benchmark_levels/pendulum.py))

Euler-integrated nonlinear pendulum, angle-only sensor.

> `x`: state vector `[θ, ω]` (angle, angular velocity). `α(θ)`: angular acceleration. `g`: gravitational constant. `length`: pendulum length. `θ_0`/`ω_0`: initial angle/angular velocity.

```
x = [θ, ω]ᵀ,  nx=2, ny=1
α(θ) = -(g/length) sin(θ)
x_{t+1} = [θ + ω·dt,  ω + α(θ)·dt] + w_t,  w_t ~ N(0, Q)
y_t     = θ_t + v_t,                        v_t ~ N(0, R)
θ_0 ~ U(-initial_angle_range, +initial_angle_range),  ω_0 = 0
```

- Simulator step/observe: [pendulum.py:29-43](benchmark_levels/pendulum.py#L29).
- Exact Jacobian of the Euler step: `F_jac = [[1, dt], [-(g/length)cos(θ)·dt, 1]]` ([pendulum.py:140-145](benchmark_levels/pendulum.py#L140)).
- Defaults: `process_noise_var=0.001, observation_noise_var=0.01, dt=0.05, g=9.81, length=1.0, initial_angle_range=π/4` ([pendulum.py:56-58](benchmark_levels/pendulum.py#L56)).
- **Prior moment-matching:** θ uniform on `[-r, r]` has variance `r²/3`; `x0_cov = diag([r²/3, 1e-6])` ([pendulum.py:150-151](benchmark_levels/pendulum.py#L150)) — exact analytic variance of a uniform distribution, not a guess. `ω_0=0` deterministically ⇒ near-zero (not exactly zero, to keep covariance PD) variance on that diagonal entry.
- **Design rationale:** default `π/4` keeps the system near the small-angle (linear) regime — intentional, so this level is a *mild* nonlinearity stress test; widening `initial_angle_range` is the documented way to push EKF/UKF toward divergence.

### 2.4 lorenz — `LorenzBenchmark` ([lorenz.py](benchmark_levels/lorenz.py))

Lorenz-63 chaotic attractor, **RK4**-integrated for both data generation and the filter's process model.

> `x,y,z`: the three Lorenz state components (note: unrelated to the generic `x_t`/`y_t` state/observation notation used elsewhere — here `x,y,z` are the Lorenz state's own coordinates, and only `x,y` are observed). `σ,ρ,β`: Lorenz system parameters.

```
ẋ = σ(y-x),  ẏ = x(ρ-z)-y,  ż = xy-βz         nx=3, ny=2 (z unobserved)
x_{t+1} = RK4_step(x_t, dt) + w_t,   w_t ~ N(0, Q),  Q = I·0.001
y_t     = [x_t, y_t] + v_t,           v_t ~ N(0, R),  R = I·1.0
x_0 = standard_normal(3) + [0, 0, 25]
```

- Derivative: [lorenz.py:31-37](benchmark_levels/lorenz.py#L31). RK4 step (both simulator and filter model use the identical 4-stage integrator): [lorenz.py:39-49](benchmark_levels/lorenz.py#L39) (simulator), [lorenz.py:158-164](benchmark_levels/lorenz.py#L158) (`get_filter_model().f`).
- Classic chaotic parameters `σ=10, ρ=28, β=8/3` ([lorenz.py:66-68](benchmark_levels/lorenz.py#L66)) — positive Lyapunov exponent, so RMSE on this level is expected to be trajectory-length- and seed-sensitive (errors compound exponentially); this is a property of the dynamics, not a bug.
- **Design rationale — RK4 in the filter, not Euler:** the filter's `f` must exactly match the simulator's integrator, or "process noise" silently absorbs an *integration-scheme* mismatch rather than genuine stochastic disturbance, which would bias every filter's apparent performance. Previously the filter used forward Euler while the simulator used RK4; fixed so both use RK4 ([lorenz.py:158-164](benchmark_levels/lorenz.py#L158)).
- Analytic Jacobian (linearized RK4-Euler-equivalent, first-order in `dt`): [lorenz.py:169-175](benchmark_levels/lorenz.py#L169).
- `x0_mean=[0,0,25], x0_cov=I` ([lorenz.py:186](benchmark_levels/lorenz.py#L186)) — matches the data generator's actual initialization.

**Overflow guard (state clipping, not noise):** `f` clips its input to `±1e3` before integrating ([lorenz.py:148,159](benchmark_levels/lorenz.py#L148)):
```python
state_bound = 1.0e3
def f(x, t=0.0):
    x = np.clip(x, -state_bound, state_bound)
    ...RK4...
```
Mirrored in the numba dynamics ([_numba_dynamics.py:141-147](benchmark_levels/_numba_dynamics.py#L141)). **Why:** the true attractor lives within roughly `[-20,20]×[-25,25]×[0,50]`; `1e3` is ~20–50× that extent, so this bound never touches legitimate trajectories. It exists purely to stop a *diverging filter estimate* (e.g. linear KF linearized at the origin, fed Lorenz data) from producing a state so large that `xy`/`xz` terms in the RK4 derivative overflow float64 to `inf`/`NaN`, which would otherwise poison every subsequent step of that trajectory irrecoverably. This is a guard on the **filter's internal estimate**, not on the data-generating simulator (the simulator has no such clip — ground truth is never altered).

---

## 3. Classical estimators

All consume a `FilterModel`; benchmark-agnostic by construction (swap the model, not the filter code). Source of truth for math: pure-NumPy path in each `estimators/classical/*.py`; numba paths in `_numba_kernels.py` are required to match bit-for-bit modulo `fastmath=True` reordering (tested in `tests/test_classical_filters.py`.

### 3.1 KF — `KalmanFilterEstimator` ([kf.py](estimators/classical/kf.py))

> `x`: state estimate, `x⁻`: predicted (prior) state estimate. `P`: state covariance, `P⁻`: predicted (prior) covariance. `K`: Kalman gain. `S`: innovation covariance. `y`: observation.

```
predict:  x⁻ = F x,        P⁻ = F P Fᵀ + Q
update:   S  = H P⁻ Hᵀ + R
          K  = P⁻ Hᵀ S⁻¹
          x  = x⁻ + K(y - H x⁻)
          P  = (I - K H) P⁻
```
[kf.py:79-87](estimators/classical/kf.py#L79) (NumPy path), [_numba_kernels.py:120-129](estimators/classical/_numba_kernels.py#L120) (numba path).

- Optimal only when `f(x)=Fx, h(x)=Hx` exactly (true for `LinearBenchmark`).
- `F = self._model.F(np.zeros(nx))` ([kf.py:44](estimators/classical/kf.py#L44)) — evaluates the Jacobian at the origin; correct only because a linear model's `F_jac` ignores its input and returns the same constant matrix everywhere.
- **Guardrail:** `use_numba=True` calls `assert_linear_model(f, h, F, H, ...)` ([kf.py:51](estimators/classical/kf.py#L51), implementation [_numba_kernels.py:71-91](estimators/classical/_numba_kernels.py#L71)) which probes `f(x_probe)` against `F@x_probe` at a random point and raises `ValueError` on mismatch. **Why:** the numba fast path hardcodes the `f(x)=Fx` assumption for speed; without this guard, running `KalmanFilterEstimator(use_numba=True)` on `pendulum`/`nonlinear`/`lorenz` would silently linearize at the origin and report plausible-looking but wrong numbers instead of erroring.

**Overflow/convergence handling (numerical-stability design choice):**
```python
cov_ceiling = 1.0e12
def _bound_cov(P):
    P = 0.5 * (P + P.T)              # force symmetry
    return np.clip(P, -cov_ceiling, cov_ceiling)
```
Applied after both predict and update covariance updates ([kf.py:70-86](estimators/classical/kf.py#L70), numba: [_numba_kernels.py:32-47](estimators/classical/_numba_kernels.py#L32)). **Why:** on a model mismatch (e.g. this *linear* KF run against chaotic Lorenz data, where `F` is the dynamics linearized at the origin and wrong almost everywhere) `P` grows roughly geometrically through the `F@P@Fᵀ` predict each step. Left unbounded it eventually overflows float64 to `inf`, then `inf - inf = NaN` poisons every subsequent estimate, turning one bad trajectory into a `NaN` RMSE for the whole run. The cap is set at `1e12` — far above any well-behaved covariance, so it never engages on a correctly-matched model, but finite, so a diverging filter still produces a **legitimately bad but numeric** estimate. Design intent: the benchmark should measure "how wrong is the bad filter," not crash with `NaN`.

### 3.2 EKF — `EKFEstimator` ([ekf.py](estimators/classical/ekf.py))

> `x`: state estimate, `x⁻`: predicted state estimate. `F`/`H`: Jacobians of `f`/`h`, evaluated at the current estimate (not constant, unlike KF). `P`/`P⁻`: covariance/predicted covariance. `K`: Kalman gain.

```
predict:  x⁻ = f(x, t),     F = ∂f/∂x|_x        P⁻ = F P Fᵀ + Q
update:   H  = ∂h/∂x|_x⁻
          S  = H P⁻ Hᵀ + R,  K = P⁻ Hᵀ S⁻¹
          x  = x⁻ + K(y - h(x⁻))
          P  = (I - K H) P⁻
```
[ekf.py:78-93](estimators/classical/ekf.py#L78).

- General — works on any `FilterModel`; `f`/`h` are arbitrary Python (or numba) callables, no linearity assumption baked into the dispatch.
- Time threaded explicitly: `x_pred = self._model.f(x, float(timestamps[t]))` ([ekf.py:82](estimators/classical/ekf.py#L82)) — required for `nonlinear`'s `8 cos(1.2 t)` forcing term to take effect.
- Same `_bound_cov` covariance-ceiling guard as KF, same `1e12` constant, same rationale ([ekf.py:72-76](estimators/classical/ekf.py#L72)): EKF on Lorenz can have its estimate wander far from the true trajectory, at which point the local Jacobian `F=F(x)` itself is evaluated far outside the region where the linearization is valid and can be large, compounding the same overflow risk as KF.

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
[ukf.py:126-150](estimators/classical/ukf.py#L126).

- Defaults `alpha=1e-3, beta=2.0, kappa=0.0` ([ukf.py:21-22](estimators/classical/ukf.py#L21)) — standard scaled-unscented-transform choice for Gaussian posteriors (Wan & van der Merwe 2000); not tuned per-level.
- **Linear fast path guard:** `ukf_linear_loop` hardcodes `f(x)=Fx, h(x)=Hx`; dispatch on `use_numba=True` runs `assert_linear_model` first ([ukf.py, numba dispatch mirrors kf.py's pattern]; kernel at [_numba_kernels.py:362-463](estimators/classical/_numba_kernels.py#L362)) so misuse on a nonlinear `FilterModel` raises `ValueError` instead of silently returning a linear-UKF result mislabeled as the real thing.
- General numba path `ukf_loop` ([_numba_kernels.py:218-309](estimators/classical/_numba_kernels.py#L218)) propagates sigma points through the level's actual `@njit` `f`/`h` — valid on every level, not just linear ones.

**Convergence/PD-matrix handling (two layered fixes, both for the same root cause — `P` losing positive-definiteness from accumulated floating-point error over many timesteps):**

1. *Symmetrization before Cholesky* — `M = (nx+λ)·0.5·(P+Pᵀ)` ([ukf.py:57](estimators/classical/ukf.py#L57)). **Why:** the Kalman update `P⁻ - K S Kᵀ` is symmetric only in exact arithmetic; repeated float64 rounding makes `P` drift slightly asymmetric, and `np.linalg.cholesky` requires (numerically) symmetric PD input.
2. *Escalating-jitter retry, scaled to the matrix's own magnitude* — on `LinAlgError`:
```python
scale = max(1.0, trace(M)/nx)
jitter = 1e-9 * scale
for _ in range(10):
    try: L = cholesky(M + jitter·I); break
    except LinAlgError: jitter *= 10.0
else:
    w, V = eigh(M); w = clip(w, jitter, None); L = cholesky((V*w)@Vᵀ)   # last resort
```
[ukf.py:58-73](estimators/classical/ukf.py#L58), numba mirror `_robust_chol` [_numba_kernels.py:50-68](estimators/classical/_numba_kernels.py#L50). **Why scaled, not a fixed `1e-6` floor (the original implementation):** a fixed jitter is too small relative to a large-magnitude `P` (e.g. Lorenz, where state/covariance values are O(10²)) to restore positive-definiteness, and unnecessarily large relative to a small one (e.g. `nonlinear`, O(1)) — distorting the sigma-point spread more than needed. Scaling jitter to `trace(M)/nx` (the matrix's own average eigenvalue) keeps the correction proportionate. Eigenvalue-clipping is the final fallback because it's guaranteed to produce a valid PD matrix by construction (any other retry could in principle keep failing).
3. Post-update symmetrization: `P = 0.5*(P+P.T)` after every covariance update ([ukf.py:149](estimators/classical/ukf.py#L149), kernel: [_numba_kernels.py:306,460](estimators/classical/_numba_kernels.py#L306)) — same drift-correction rationale as (1), applied proactively every step rather than only on Cholesky failure.

### 3.4 PF — `ParticleFilterEstimator` ([pf.py](estimators/classical/pf.py))

Sequential Importance Resampling (bootstrap particle filter) — only estimator here making **no Gaussian-posterior assumption**, asymptotically correct (`M→∞`) even on `nonlinear`'s bimodal observation model.

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
[pf.py:59-86](estimators/classical/pf.py#L59).

- **Numerical-stability design choice — log-sum-exp weighting:** `log_w -= log_w.max()` before exponentiating ([pf.py:74](estimators/classical/pf.py#L74)). **Why:** the raw Gaussian log-likelihood can be a large negative number (e.g. -10⁴ for a far-off particle); `exp(-10⁴)` underflows to exactly `0.0` in float64 for *every* particle if computed directly when likelihoods are all very small, making the weight sum `0/0 = NaN`. Subtracting the max shifts the largest log-weight to `0` before exponentiating, so at least one particle's weight is exactly `1.0` pre-normalization — numerically stable regardless of how small the absolute likelihoods are; produces the identical normalized weights as the naive formula in exact arithmetic.
- **Resampling trigger — effective sample size:** `N_eff = 1/Σw²` ([pf.py:80](estimators/classical/pf.py#L80)), threshold `resample_threshold·M` (default `0.5·M`, [pf.py:20](estimators/classical/pf.py#L20)). **Why:** without resampling, particle weights degenerate over time (one particle dominates, rest →0) — `N_eff` is the standard estimate of "how many particles are effectively contributing"; resampling only when `N_eff` drops below half of `M` avoids resampling (and its added variance) every single step while still preventing weight collapse.
- **Reproducibility:** `rng = np.random.default_rng(self._random_seed)` ([pf.py:57](estimators/classical/pf.py#L57)), constructor default `random_seed=0` ([pf.py:21](estimators/classical/pf.py#L21)). Previously unseeded (`np.random.default_rng()` with no argument) — every other estimator/level in the repo seeds deterministically; PF was the one exception, fixed for run-to-run reproducibility.
- No covariance-ceiling guard needed/present — PF has no `P` matrix to diverge; an individual particle that wanders off only loses weight (its likelihood `→0`), it doesn't poison the run numerically the way an unbounded covariance does in KF/EKF/UKF.

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
[kalmannet.py:184-204](estimators/neural/kalmannet.py#L184) (`_run_sequence`).

- Architecture: 1-layer GRU → LayerNorm → 2-layer MLP head → flattened `K` ([kalmannet.py:43-77](estimators/neural/kalmannet.py#L43)).
- **Zero-initialized gain head:** `nn.init.zeros_(fc_gain[-1].weight/bias)` ([kalmannet.py:61-62](estimators/neural/kalmannet.py#L61)). **Why:** an untrained network then outputs `K=0` exactly, so `estimate()` on an unfit model reduces to pure process-model rollout (`x_t = f(x_{t-1}, t)`, no correction at all) — a deliberate, checkable invariant: if an unfit `KalmanNetEstimator` does *not* reproduce iterated `f`, the GRU/gain wiring is broken.
- **Uncertainty variant** (`KalmanNetUncertaintyEstimator`, `_predict_log_var=True`): adds a log-variance head, trained with Gaussian NLL instead of MSE — [kalmannet.py:234-238](estimators/neural/kalmannet.py#L234):
```python
if predict_log_var: return gaussian_nll_loss(pred, target, exp(log_var), eps=1e-6)
else:                return mse(pred, target)
```
  `eps=1e-6` floors the NLL's implicit variance term to avoid division by (near-)zero variance — same overflow-prevention spirit as the classical filters' covariance ceiling, applied to a learned variance instead of a propagated one.
- **Training-time stability control:** `clip_grad_norm_(network.parameters(), 1.0)` before every optimizer step ([kalmannet.py:251](estimators/neural/kalmannet.py#L251)) — standard RNN exploding-gradient guard; bounds the gradient L2 norm to 1.0 regardless of sequence length or loss magnitude.
- **Why the analytic predict step is NumPy, not torch:** `_process_model_step` round-trips every state vector through NumPy each timestep (`x_batch.cpu().numpy() → Python loop calling f → back to tensor`, [kalmannet.py:99-108](estimators/neural/kalmannet.py#L99)) because `FilterModel.f` is a plain NumPy callable shared with the classical filters — reusing the exact same `f` (rather than a duplicated torch reimplementation) guarantees KalmanNet's process model is bit-identical to KF/EKF/UKF's, at the cost of being the dominant runtime cost of this estimator.
- `estimate()` always runs on CPU regardless of training device ([kalmannet.py:118-121](estimators/neural/kalmannet.py#L118), enforced at [kalmannet.py:274](estimators/neural/kalmannet.py#L274)) — design choice so inference-time runtime/latency metrics are measured under the same hardware condition as the classical (CPU-only) filters, making cross-estimator runtime comparisons fair.
- `neural_ode.py`, `transformer.py` are stubs — every method raises `NotImplementedError`; excluded from default `ESTIMATORS`, present only in `EXPERIMENTAL_ESTIMATORS`. No math to report; not benchmarkable yet.

---

## 5. Metrics ([metrics/](metrics/))

> `x̂`: estimated state, `x`: true state. `N`: number of trajectories, `T`: number of timesteps, `nx`: state dimension. `s`: elapsed seconds, `n`: number of steps.

```
compute_rmse(x̂, x)         = sqrt(mean((x̂-x)²))                  over ALL of [N,T,nx] pooled — one scalar
compute_rmse_per_dim(x̂, x) = sqrt(mean((x̂-x)², axis=(0,1)))       → array [nx], per-dimension
runtime_per_step_ms(s, n)  = (s/n) * 1000   if n>0 else 0
latency_ms_per_step(s,N,T) = runtime_per_step_ms(s, N*T)
measure_memory()           = psutil RSS in MB, or None if psutil absent
```
[rmse.py:6-19](metrics/rmse.py#L6), [rmse.py:22-40](metrics/rmse.py#L22), [runtime.py:19-22](metrics/runtime.py#L19), [latency.py:6-11](metrics/latency.py#L6), [memory.py:7-16](metrics/memory.py#L7).

- **Why `compute_rmse` pools dims/timesteps into one scalar:** fine for ranking estimators *within* one benchmark, where all state dimensions share physical units consistently (e.g. linear's position+velocity are both in the same length/time-derived scale used throughout that level). **Explicitly not valid across benchmarks** — averaging linear's RMSE with Lorenz's RMSE mixes incompatible units (position vs. chaotic xyz scale ~O(10) vs ~O(1)); `compute_rmse_per_dim` exists for that case, plus ranking-within-benchmark rather than raw-averaging-across-benchmarks for any cross-benchmark summary table.
- One formula for runtime, used in exactly one place (`runner.py` imports `metrics.runtime.runtime_per_step_ms` rather than inlining) — previously the same division was written three times across `runtime.py`/`latency.py`/`runner.py`; consolidated to remove drift risk between copies.

---

## 6. Cross-cutting numerical-stability pattern (summary)

Three independent overflow/convergence problems, three matched fixes, same underlying logic — **bound the quantity that can blow up, but never replace a finite-but-bad estimate with a crash**:

| Failure mode | Where | Guard | Bound |
|---|---|---|---|
| `P` grows unbounded under model mismatch → `inf`/`NaN` | KF, EKF (`_bound_cov`) | symmetrize + clip | `±1e12` |
| `P` loses PD-ness from float rounding → Cholesky fails | UKF (`_sigma_points`/`_robust_chol`) | symmetrize, then escalating jitter scaled to `trace(P)/nx`, eigval-clip as last resort | starts `1e-9·scale`, ×10 per retry, 10 retries |
| Diverging state estimate → RK4 derivative overflows | Lorenz `f` (filter only, not the simulator) | `np.clip(x, -1e3, 1e3)` before integrating | `±1e3` (≈20–50× attractor extent) |
| Particle weights underflow to 0/0 = NaN | PF weighting | log-sum-exp shift before `exp` | exact, not a tunable bound |

All four are **inference-time-only** safeguards on the filter's *own internal state estimate* — none alter the data-generating simulators, none alter `Q`/`R`, and none change the math when the filter is well-behaved (every bound is set far outside any value a correctly-converging filter would ever reach). The intent in every case: a badly-matched filter (e.g. linear KF on chaotic data) should report a numeric, legitimately-bad RMSE — the actual signal the benchmark is trying to measure — rather than `NaN`, which would silently drop that estimator/trajectory from the comparison.
