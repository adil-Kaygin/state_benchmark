# Issue 6: Multi-Sensor Vehicle Tracking Benchmark (range + bearing, multiple noise levels)

Add a new benchmark level `VehicleTrackingBenchmark` that tracks a ground vehicle
with **several fixed sensor stations**, each reporting **range and bearing**
(distance + angle) at a **different noise level**. This is the canonical
radar/sonar multi-sensor fusion problem and is the first level built on
[Issue 5 — complex / multi-sensor shared contract](5_complex_sensor_benchmarks_shared_contract.md).

> Read [Issue 5](5_complex_sensor_benchmarks_shared_contract.md) first, then the
> three existing levels [`linear.py`](../benchmark_levels/linear.py),
> [`pendulum.py`](../benchmark_levels/pendulum.py),
> [`lorenz.py`](../benchmark_levels/lorenz.py). This issue only states the deltas.

## Why this level

The user's framing: *"a simulation where we are tracking a vehicle, our state
vector is angle and distance, and there are multiple sensors with multiple noise
levels."* The literature calls this **range-bearing multi-sensor target
tracking**. It is the standard step up from our current levels because:

- The measurement is **nonlinear** (polar: range/bearing of a Cartesian state) —
  but only *mildly* nonlinear compared to bearings-only, which makes it the right
  first complex level: hard enough that UKF should beat EKF, not so hard
  everything diverges. (Bearings-only is "high nonlinearity"; range+bearing is
  "low nonlinearity" — see refs.)
- It is a genuine **fusion** problem: `K` sensors, **heterogeneous `R`**, so the
  filter must weight a precise sensor over a noisy one. None of the current levels
  test this.
- It maps cleanly onto our stacked-`h` / block-`R` contract with no schema change.

## State, dynamics

Track the vehicle in **Cartesian** state (the standard choice — Cartesian
dynamics are linear and stable; the nonlinearity is pushed entirely into `h`,
which is the well-conditioned formulation):

```
x = [px, py, vx, vy]              nx = 4        state_names = ("px","py","vx","vy")
```

Constant-velocity (CV) process model with a discrete white-noise-acceleration
`Q` (the textbook CV model):

```
f(x) = [px + vx·dt,  py + vy·dt,  vx,  vy]          (linear → F is constant)
Q = q · [[dt^4/4, 0, dt^3/2, 0],
         [0, dt^4/4, 0, dt^3/2],
         [dt^3/2, 0, dt^2, 0],
         [0, dt^3/2, 0, dt^2]]                       (per-axis DWNA, scalar intensity q)
```

> A coordinated-turn variant (state `[px,py,vx,vy,ω]`, nonlinear `f`) is a good
> *follow-up* level but out of scope here — keep `f` linear so the only new
> difficulty is the sensors. Note the option in the docstring.

## Measurement: K sensors, range + bearing, heterogeneous noise

`K` sensor stations at fixed positions `(sx_k, sy_k)`. Each emits
`[range_k, bearing_k]`; stack them:

```
ny = 2·K
h(x)[2k]   = sqrt((px - sx_k)^2 + (py - sy_k)^2)            # range
h(x)[2k+1] = atan2(py - sy_k, px - sx_k)                    # bearing  (wrap to (-π,π])
R = blkdiag( diag(σ_r,k^2, σ_b,k^2)  for k in 0..K-1 )      # block-diagonal [ny,ny]
```

**Multiple noise levels** = the per-sensor `(σ_r,k, σ_b,k)` differ (a precise
station and a cheap one), *and* a global scalar `noise_scale` multiplies all of
them for sweeps. Default to **K = 3** sensors with, e.g.,
`σ_r = (0.5, 2.0, 5.0) m`, `σ_b = (0.5°, 2°, 5°)` so the filter has to fuse one
good + two poor sensors. Place sensors around the scene (e.g. three corners) so
the geometry (GDOP) varies along the trajectory.

### Analytic Jacobian `H` (per sensor block, `[2,4]`)

With `dx = px - sx_k`, `dy = py - sy_k`, `r = sqrt(dx^2 + dy^2)` (guard `r > eps`,
raise/`ε`-floor for degenerate r=0 over a sensor):

```
∂range/∂(px,py)   = [ dx/r ,  dy/r ]            ∂range/∂(vx,vy)   = 0
∂bearing/∂(px,py) = [ -dy/r^2 ,  dx/r^2 ]       ∂bearing/∂(vx,vy) = 0
```

Stack the `K` blocks → `H(x) ∈ R^{2K×4}`. `F` is the constant CV matrix.

### Bearing angle-wrap (mandatory footgun)

Every innovation `y - h(x)` involving a bearing MUST wrap the bearing component to
`(-π, π]` (`atan2(sin(d), cos(d))`). This applies to EKF/UKF innovations **and**
to any neural innovation feature (KalmanNet / Transformer / Mamba). Document it in
the level docstring; it is the #1 way this level silently mis-behaves near the
±π branch cut.

## Generation

- `VehicleTrackingSimulator(BaseSimulator)`: `step` = CV + process noise from `Q`;
  `observe` builds the stacked `[2K]` reading and adds per-sensor noise from the
  `R` blocks. Wrap emitted bearings to `(-π,π]`.
- Init each trajectory with a random start pose/velocity in a scene box (expose
  `scene_size`, `initial_speed_range`, `initial_state_var`); `x0_mean`/`x0_cov`
  from those ranges (copy the pendulum's analytic-prior pattern).
- Same 70/15/15 HDF5 split via `HDF5Writer` + `DatasetMetadata`; copy
  [`lorenz.py:130-169`](../benchmark_levels/lorenz.py#L130-L169). Ground truth
  (Cartesian states) is never noised.
- Suggested defaults: `trajectory_length=200`, `num_trajectories=2000`, `dt=0.1`.

## Optional: sensor dropout (gated, OFF by default)

Add a `dropout_prob: float = 0.0` knob. When > 0, with that per-step,
per-sensor probability emit **`NaN`** in that sensor's range+bearing slot (never
`0.0`). If used, set metadata/docstring loudly per Issue 5. Leave it **0.0** by
default so EKF/UKF and the current estimators run unmodified; handling NaN gating
is a separate follow-up.

## FilterModel completeness (per Issue 5)

`f, h, F, H, Q, R, x0_mean, x0_cov` **plus**:

- `numba`: `build_vehicle_tracking_numba_dynamics(sensors, dt)` in
  [`_numba_dynamics.py`](../benchmark_levels/_numba_dynamics.py) — `@njit` `f`/`h`
  (stacked, `atan2`/`sqrt` are njit-OK) and `F_jac`/`H_jac`, identical math to the
  NumPy versions. Required: EKF/UKF have no NumPy fallback.
- `torch`: `build_vehicle_tracking_torch_dynamics(sensors, dt)` in
  [`_torch_dynamics.py`](../benchmark_levels/_torch_dynamics.py) — batched
  `[B,4]→[B,2K]`, `torch.atan2`/`torch.sqrt`, identical math. Required for
  KalmanNet/neural GPU training; absent ⇒ Issue-0 `ValueError`, no silent
  fallback.

All three (`numpy` / `njit` / `torch`) MUST agree to float tolerance — same
invariant the other levels hold.

## Wiring

1. `benchmark_levels/vehicle_tracking.py` (Simulator + Benchmark, `name =
   "vehicle_tracking"`).
2. Numba + torch dynamics builders (above).
3. Register in [`__init__.py`](../benchmark_levels/__init__.py): import,
   `BENCHMARK_LEVELS["vehicle_tracking"]`, `__all__`.
4. Add to the experiment/notebook config dicts (`BENCHMARK_CONFIGS`,
   `BENCHMARK_CLASSES`, and the per-level estimator-config blocks). Suggested
   `BENCHMARK_CONFIGS` entry:

   ```python
   "vehicle_tracking": dict(
       trajectory_length=200,
       num_trajectories=2000,
       num_sensors=3,
       sensor_range_noise=(0.5, 2.0, 5.0),     # metres
       sensor_bearing_noise_deg=(0.5, 2.0, 5.0),
       process_noise_intensity=0.1,
       dropout_prob=0.0,
   ),
   ```

## Acceptance criteria

- [ ] `VehicleTrackingBenchmark` mirrors the existing levels; `nx=4`, `ny=2K`,
      `state_names=("px","py","vx","vy")`; 70/15/15 HDF5 split; ground truth
      uncorrupted.
- [ ] Stacked range+bearing `h` → `[2K]`; block-diagonal `R`; analytic `H
      [2K,4]`; constant CV `F`; DWNA `Q`.
- [ ] **Bearing residuals angle-wrapped** at every innovation site; documented.
- [ ] Heterogeneous per-sensor noise + global `noise_scale`; sensors placed for
      varying geometry.
- [ ] `FilterModel` complete incl. `numba` and `torch` builders with math
      identical to the NumPy path (numpy/njit/torch agree to tolerance).
- [ ] `dropout_prob` knob emits `NaN` (not `0.0`) when on; OFF by default + loud
      metadata when on.
- [ ] Registered in `__init__.py` + experiment/notebook config dicts; runs in the
      standard sweep against existing EKF/UKF/KalmanNet (+ neural) estimators.
- [ ] No `dataset.states` leak in any estimator path; no `tests/`, no `pip
      install`; lazy `numba`/`torch` imports.

## References (multi-sensor range-bearing tracking)

- Kalman-filter sensor fusion for road-object detection/tracking, autonomous
  vehicles — https://www.researchgate.net/publication/347816613
- Adaptive UKF for target tracking with time-varying noise covariance, multi-sensor
  fusion — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8434080/
- Range/bearing vs bearings-only nonlinearity (range+bearing is low-nonlinearity);
  bearings-only underwater multi-target tracking — https://pmc.ncbi.nlm.nih.gov/articles/PMC9370893/
- Pseudolinear filter for bearings-only tracking (stability of EKF/UKF on polar
  measurements) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8399602/
- Joint Adaptive Kalman Filter (JAKF) for vehicle motion state estimation —
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4970148/
