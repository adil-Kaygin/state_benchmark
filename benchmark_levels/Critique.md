# Critique — benchmark_levels

Methodology review of the four data-generating levels. Tags: **[bug]** /
**[design-limitation]** / **[debatable-choice]** / **[valid-as-is]**. See the
root [Critique.md](../Critique.md) for cross-cutting issues.

---

## 1. Lorenz EKF Jacobian one integration-order below `f` — [resolved]

*Was a design-limitation; fixed.* The standard `LorenzBenchmark` now hands EKF/KF
the **exact Jacobian of the 4-stage RK4 map** (chain rule through the four stages,
[lorenz.py](lorenz.py)), so the covariance is propagated with the same `O(dt⁴)`
accuracy as the RK4 mean `f` — mean and covariance are integration-order
consistent, and the verification matches a finite-difference Jacobian of `f` to
~1e-9. The old first-order `F = I + dt·J` linearization is retained only as the
explicitly-labelled `LorenzFEABenchmark` (`lorenz_fea`) baseline, so the cost of
the inconsistency can still be measured rather than silently shipped.

## 2. Pendulum ground truth is forward-Euler — [design-limitation]

The pendulum **simulator** integrates with a single forward-Euler step
(`new_state = [θ + ω·dt, ω + α(θ)·dt]`, [pendulum.py:37](pendulum.py#L37)). The
"true" trajectory the filters are scored against therefore carries Euler
integration error and does not conserve energy — at the default `π/4` swing and
`dt = 0.05` this is small, but it grows with `initial_angle_range` and `dt`.

This is consistent in the sense that the filter's `f` is the *same* Euler map
(so there's no model mismatch), but it means "ground truth" is a low-order
discretization of the real pendulum, not the real pendulum. Any claim about
absolute accuracy on a physical pendulum is therefore limited by the integrator,
not the filter. (Contrast Lorenz, where the simulator is RK4.)

*Recommendation:* document this as a deliberate "mild nonlinearity" choice, or
upgrade the pendulum simulator to RK4 to match Lorenz and remove the integration
bias from ground truth.

## 3. `nonlinear.py` overloads the `dt` argument as the time index `t` — [debatable-choice]

`NonlinearSimulator.step(state, control, dt)` computes
`8.0 * np.cos(1.2 * dt)` ([nonlinear.py:34](nonlinear.py#L34)), and
`generate_dataset` calls `simulator.step(x, None, float(t))`
([nonlinear.py:97](nonlinear.py#L97)) — i.e. it passes the **timestep index** in
the slot named `dt`. This works only because the Gordon-Salmond-Smith model has
no real time-step (the recursion is already discrete), so `dt` is free to be
repurposed as `t`. It is fragile: anyone unifying `BaseSimulator.step`'s
signature across levels, or reading `dt` as an actual time increment, will
silently break the `cos(1.2 t)` forcing.

*Recommendation:* give `step` an explicit `t` parameter distinct from `dt`, or
rename the slot, so the time-forcing is not smuggled through a misnamed argument.

## 4. Lorenz `±1e3` state clip changes the RMSE of a diverging filter — [valid-as-is, with caveat]

`f` clips its input to `±1e3` before integrating
([lorenz.py:148,159](lorenz.py#L148)), and this is correctly applied to the
**filter's estimate only**, never to the data-generating simulator — so ground
truth is untouched. The bound (~20–50× the attractor's extent) never engages on
a well-matched filter.

The caveat to state honestly: when a *badly*-matched filter (e.g. a linear KF on
Lorenz, see `estimators/Critique.md`) diverges, the clip stops it from producing
`inf`/`NaN` and instead yields a large-but-finite estimate. That changes the
**reported RMSE** of the bad filter from "NaN / dropped" to "numerically huge."
That is the stated intent (measure how-wrong, not crash), and it is defensible —
but it means the RMSE of a diverging filter is partly an artifact of where the
clip sits, not a pure measure of the filter. Readers comparing a diverged
filter's RMSE should treat the magnitude as "diverged," not as a meaningful
distance.

## 5. Single fixed parameterization per level — [debatable-choice]

Each level ships one default parameterization (noise levels, `dt`, trajectory
length). There is no sweep over signal-to-noise ratio or nonlinearity strength.
The pendulum README notes that `initial_angle_range` controls how nonlinear the
problem is, but nothing in the benchmark exercises that axis by default — so the
results characterize the estimators at one operating point, not across the
regimes where their relative ranking is known to flip (e.g. EKF vs UKF vs PF as
nonlinearity increases).

*Recommendation:* parameterize at least one level over a nonlinearity/SNR knob
and report RMSE as a curve, not a single point.

---

## Summary table

| # | Issue | Tag | Where |
|---|-------|-----|-------|
| 1 | Lorenz `F_jac` Euler-order while `f` is RK4 | resolved (exact RK4 Jacobian; FEA kept as `lorenz_fea` baseline) | lorenz.py |
| 2 | Pendulum ground truth is forward-Euler | design-limitation | pendulum.py:37 |
| 3 | `dt` argument overloaded as time index `t` | debatable-choice | nonlinear.py:34,97 |
| 4 | `±1e3` clip shapes a diverging filter's RMSE | valid-as-is (caveat) | lorenz.py |
| 5 | One fixed operating point per level | debatable-choice | all levels |
