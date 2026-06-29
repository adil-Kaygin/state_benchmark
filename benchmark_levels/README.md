# benchmark_levels

State-space data generators. Each level defines a discrete-time stochastic
system

```
x_{t+1} = f(x_t, t) + w_t,      w_t ~ N(0, Q)      [STATE, latent, ground truth]
y_t     = h(x_t)     + v_t,      v_t ~ N(0, R)      [OBSERVATION, noisy sensor data]
```

`x_t` is written to `states` in the HDF5 dataset (never seen by an
estimator at inference time — used only for RMSE scoring).
`y_t` is written to `observations` (the only thing an estimator may consume).

`f`/`h` in `FilterModel` are **noise-free** deterministic maps; `Q`/`R` carry
all stochasticity, so the same `f`/`h` is reused for data generation and filter
prediction without double-counting noise.

`BenchmarkLevel.get_filter_model()` returns the matching
`FilterModel(f, h, F, H, Q, R, x0_mean, x0_cov, numba)` used by classical
filters:

- `F`/`H` are the Jacobians of `f`/`h` (constant matrices when the system is
  linear).
- `x0_mean` / `x0_cov` expose the generative prior so estimators initialise from
  the true prior instead of a hardcoded default.
- `numba` is a `NumbaDynamics` mirror of `f/h/F_jac/H_jac` as `@njit` closures
  (built in `_numba_dynamics.py`). It is **required** by the classical filters —
  they run exclusively on it (there is no pure-NumPy fallback); EKF/UKF raise if
  it is `None`. The Python callables remain the source of truth and must match
  the numba math one-for-one.
- `torch` is an optional `TorchDynamics` mirror of `f/h` as **batched** torch
  closures (`[B,nx]→[B,nx]/[B,ny]`, built in `_torch_dynamics.py`), used only by
  KalmanNet for vectorized GPU training. torch is imported lazily inside the
  closures, so levels still import without torch installed.

Each `BenchmarkLevel` also exposes `state_names` — the physical name of every
state dimension (`("x","y","z")` for Lorenz, `("theta","omega")` for pendulum,
`("position","velocity")` for linear, `("x",)` for nonlinear). Metrics and plots
use these to label per-dimension RMSE with the real variable instead of an index.

`f` has signature `f(x, t=0.0)`; the `t` argument lets a level carry
time-varying forcing (see `nonlinear.py`).

## linear.py — `LinearBenchmark`

Constant-velocity tracking model.

```
x = [position, velocity]ᵀ                      nx = 2, ny = 1
F = [[1, dt], [0, 1]]                           h(x) = H x,  H = [1, 0]
x_{t+1} = F x_t + w_t,  w_t ~ N(0, Q),  Q = I·process_noise_var
y_t     = H x_t + v_t,  v_t ~ N(0, R),  R = I·observation_noise_var
x_0 ~ N(0, I·initial_state_var)
```

- Constructor params `process_noise_var`, `observation_noise_var`,
  `initial_state_var` are used directly as variances (`Q = I·process_noise_var`).
  Defaults `0.01 / 0.1 / 1.0`, `dt = 0.1`.
- Fully linear ⇒ `F`, `H` are constant matrices; the Kalman filter is the
  Bayes-optimal estimator here. This level is the sanity-check / upper-bound case
  other filters should match.
- `get_filter_model()` returns `x0_mean = zeros(2)`,
  `x0_cov = I·initial_state_var`.

## nonlinear.py — `NonlinearSimulator` / `NonlinearBenchmark`

Gordon, Salmond & Smith (1993) scalar benchmark, the standard non-Gaussian
particle-filter stress test.

```
x_{t+1} = 0.5 x_t + 25 x_t / (1 + x_t²) + 8 cos(1.2 t) + w_t,  w_t ~ N(0, Q)
y_t     = x_t² / 20 + v_t,                                      v_t ~ N(0, R)
nx = 1, ny = 1,  Q = R = 1.0,  x_0 ~ N(0, 1)
```

- `F_jac(x) = 0.5 + 25(1 - x²)/(1 + x²)²`; `H_jac(x) = x/10`. The squaring in `h`
  makes the observation **bimodal in sign** (±x give the same `y`), which breaks
  filters (EKF/UKF) that assume local linearity/unimodality and rewards the
  particle filter — this is why the level exists.
- `f` carries an explicit time-forcing term `8 cos(1.2 t)`. Estimators thread the
  dataset timestamp into `f(x, t)`; the Jacobian `F_jac` does not include the
  forcing term because `cos(1.2 t)` has zero state-derivative.
- `x0_mean = zeros(1)`, `x0_cov = eye(1)` (matches the data generator's
  `x ~ standard_normal(1)` init).
- Implementation note: `NonlinearSimulator.step(state, control, dt)` reuses its
  `dt` argument *as* the timestep index `t` (`generate_dataset` passes
  `float(t)`), which works only because this level has no real `dt`.

## pendulum.py — `PendulumSimulator` / `PendulumBenchmark`

Nonlinear pendulum, Euler-integrated, angle-only sensor.

```
x = [θ, ω]ᵀ                                     nx = 2, ny = 1
α(θ) = -(g/length) sin(θ)                       angular acceleration
x_{t+1} = [θ + ω·dt, ω + α(θ)·dt] + w_t,  w_t ~ N(0, Q)
y_t     = θ_t + v_t,                       v_t ~ N(0, R)   (angle-only sensor)
θ_0 ~ U(-initial_angle_range, +initial_angle_range),  ω_0 = 0
```

- `F_jac(x) = [[1, dt], [-(g/length) cos(θ) dt, 1]]` — exact Jacobian of the
  Euler step.
- Defaults: `process_noise_var = 0.001`, `observation_noise_var = 0.01`,
  `dt = 0.05`, `g = 9.81`, `length = 1.0`, `initial_angle_range = π/4`. The
  small-angle default keeps the system close to the linear regime; widen
  `initial_angle_range` to stress-test EKF/UKF divergence at large swing angles.
- Prior is moment-matched from the uniform angle distribution: θ uniform on
  `[-r, r]` has variance `r²/3`, so `x0_mean = zeros(2)`,
  `x0_cov = diag([r²/3, 1e-6])` (the `1e-6` keeps the deterministic `ω_0 = 0`
  diagonal entry positive-definite).

## lorenz.py — `LorenzSimulator` / `LorenzBenchmark`

Lorenz-63 chaotic attractor, RK4-integrated for both data generation and the
filter's process model.

```
ẋ = σ(y - x)
ẏ = x(ρ - z) - y                              nx = 3, ny = 2 (z unobserved)
ż = xy - βz
x_{t+1} = RK4_step(x_t, dt)  + w_t,   w_t ~ N(0, Q),  Q = I·0.001
y_t     = [x_t, y_t] + v_t,            v_t ~ N(0, R),  R = I·1.0
x_0 ~ N([0, 0, 25], I)
```

(Here `x, y, z` are the Lorenz state's own coordinates, distinct from the generic
`x_t`/`y_t` state/observation notation; only `x, y` are observed.)

- Classic chaotic parameters `σ = 10, ρ = 28, β = 8/3` (positive Lyapunov
  exponent ⇒ small state errors compound exponentially, so RMSE on this level is
  expected to be trajectory-length- and seed-sensitive — a property of the
  dynamics, not a bug).
- The simulator step and `get_filter_model().f` use the **identical** 4-stage
  RK4 integrator, so the filter's process model matches the data generator.
- Analytic Jacobian (standard `LorenzBenchmark`): the **exact Jacobian of the
  RK4 step**, computed by the chain rule through the four stages, so the EKF/KF
  covariance is propagated with the same `O(dt⁴)` accuracy as the mean `f`.
  `H_jac` selects `[x, y]`.
- `x0_mean = [0, 0, 25]`, `x0_cov = I` (matches the data generator's init).

### `lorenz_fea` — `LorenzFEABenchmark` (Forward-Euler-Approximation baseline)

Identical data generation and `f`/`h` to `LorenzBenchmark`, but the EKF/KF
Jacobian is the **first-order** `F_jac(x) = I + dt·J(x)` (the forward-Euler
linearization of the flow, *not* the Jacobian of the RK4 map). Because `f` is
RK4 (`O(dt⁴)`) while this `F` is `O(dt)`, mean and covariance are propagated at
inconsistent orders. Registered as `lorenz_fea` and retained **only as a
baseline** to quantify the cost of that inconsistency against the standard
`lorenz`. Prefer `lorenz` for any real comparison.
- **State clip (filter only):** `f` clips its input to `±1e3` before integrating
  (mirrored in the numba dynamics). The true attractor lives within roughly
  `[-20,20]×[-25,25]×[0,50]`; `1e3` is ~20–50× that extent, so the bound never
  touches legitimate trajectories. It exists only to stop a *diverging filter
  estimate* from producing a state so large that the `xy`/`xz` terms overflow
  float64 to `inf`/`NaN`. The data-generating simulator has no such clip — ground
  truth is never altered.

## Extending with a new level

1. Subclass `BenchmarkLevel` (`__init__.py` registers it in `BENCHMARK_LEVELS`)
   and implement `state_names` (one physical name per state dimension).
2. Implement a `BaseSimulator` with `step(state, control, dt) -> state` (adds
   process noise) and `observe(state) -> obs` (adds observation noise).
3. `get_filter_model()` must return `f`/`h` as *noise-free* deterministic maps
   (noise lives in `Q`/`R`, not inside `f`/`h`) — classical filters add `Q`/`R`
   themselves; baking noise into `f`/`h` will double-count it.
4. If `f`/`h` are nonlinear, supply exact Jacobians `F`/`H` — EKF accuracy is only
   as good as these derivatives.
5. Expose the generative prior via `x0_mean` / `x0_cov` in `get_filter_model()`.
6. Supply a `NumbaDynamics` (see `_numba_dynamics.py`) — it is **required** for
   the classical filters (no NumPy fallback); keep it bit-equivalent to the
   Python `f/h/F_jac/H_jac`.
7. To train KalmanNet on the level, also supply a `TorchDynamics` (batched torch
   `f`/`h`, see `_torch_dynamics.py`), mirroring the same math.
