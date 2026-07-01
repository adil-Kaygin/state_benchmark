# Issue 11: PINN and Neural-ODE GPU forward-pass efficiency (non-cacheable loops)

Unlike KalmanNet/Transformer/Mamba (Issue 9), PINN and Neural-ODE cannot cache
their dynamics: their per-step `f`/RK4 depends on the **network's own previous
output**, so it is recomputed every gradient step by necessity. But each has a
specific, real inefficiency in its GPU training path that IS removable without
changing the math.

## PINN — fuse the two physics-loss sweeps into the forward loop

`PINNFilterEstimator._loss` runs the forward recursion once, then runs **two more
full-trajectory Python loops over T** to build the physics residuals:

- forward — [`pinn.py:139-147`](../estimators/neural/pinn.py#L139-L147): computes
  `x_pred = f(x)` and `y_pred = h(x_pred)` for every step, then **discards them**.
- `r_dyn` — [`pinn.py:165-167`](../estimators/neural/pinn.py#L165-L167): a second
  T-loop of `f` over `x_hat[:, :-1]`.
- `r_meas` — [`pinn.py:173-175`](../estimators/neural/pinn.py#L173-L175): a third
  T-loop of `h` over `x_hat`.

On Lorenz each of those `f` calls is a 4-stage RK4, so the loss does **three**
200-step RK4/h sweeps per batch when it could do far less. The residuals are
defined on the corrected estimate `x_hat`, so they are not literally the forward's
discarded `x_pred` — but they CAN be folded into the single existing forward
T-loop:

- Inside the forward loop, after computing `x = x_pred + dx` (the new `x_hat_t`),
  accumulate `r_meas_t = y_t - h(x_hat_t)` immediately (you already have `h`
  applied once per step there as `y_pred` on `x_pred`; add one `h(x_hat_t)`).
- For `r_dyn_t = x_hat_t - f(x_hat_{t-1})`: carry the previous step's `x_hat`
  and evaluate `f` on it once per step inside the same loop, instead of a separate
  post-hoc loop over `x_hat[:, :-1]`.

This replaces three independent T-loops with one, removing two full RK4/h sweeps
per batch on Lorenz. **Numerically identical** — same residuals, just computed in
one pass. Keep the `lambda_dyn == 0` / `lambda_meas == 0` short-circuits so the
ablation baseline still skips the unused terms.

Minor, do-while-here: `r_meas`/`r_dyn` use `(r**2).mean()`
([`pinn.py:169`](../estimators/neural/pinn.py#L169), [`pinn.py:177`](../estimators/neural/pinn.py#L177))
while `r_data` uses `F.mse_loss` — fine to leave, but if you touch these lines use
`F.mse_loss`/`(...).pow(2).mean()` consistently. Cosmetic only.

NOTE: PINN's forward T-loop itself
([`pinn.py:139`](../estimators/neural/pinn.py#L139)) is an **irreducible causal
recurrence** (each step uses the net's own previous `x`). It cannot be parallelized
over T or cached — leave it sequential. Only the redundant *extra* sweeps are the
target here.

## Neural-ODE — hoist the time-feature column out of the inner drift evals

`_NeuralODENet.drift_fn` rebuilds the scalar time-feature column on **every** drift
evaluation — [`neural_ode.py:63`](../estimators/neural/neural_ode.py#L63):

```python
t_col = torch.full((x.shape[0], 1), float(t_val), dtype=x.dtype, device=x.device)
return self.drift(torch.cat([x, t_col], dim=1))
```

`drift_fn` is called **4 stages × n_substeps × T** times per batch (default
`n_substeps=4`, T=200 ⇒ 3200 allocations/batch) at
[`neural_ode.py:171-174`](../estimators/neural/neural_ode.py#L171-L174), and within
a single RK4 step `t_val` is held constant across all stages and substeps (the
integrator does not advance `t` within the step —
[`_rk4_step_torch`](../estimators/neural/neural_ode.py#L167-L176) uses the same
`t_val` throughout). So the `t_col` is identical for all 16 evals of a step and can
be built **once per timestep** and passed down, eliminating ~3000 tiny allocations
per batch.

Implementation: build `t_col` once in `_rk4_step_torch` (or in `_forward_train`'s
per-`t` body) and thread it into a `drift_fn`/`_torch_drift` variant that takes the
precomputed column, instead of reconstructing it inside `drift_fn`. **Numerically
identical** — same constant column, fewer allocations.

NOTE: do NOT also try to cache `f(x)` across RK4 stages — the model-drift residual
(`use_model_drift=True`, [`neural_ode.py:164`](../estimators/neural/neural_ode.py#L164))
correctly needs `f` at each distinct stage state, and the default
`use_model_drift=False` doesn't call `f` at all. Neural-ODE's T-loop and RK4 stages
are also irreducibly network-dependent (the drift is the learned field) — leave the
loop structure alone.

## Why these are real-life safe

Both changes only reorganize WHERE existing tensor ops run (fuse loops / hoist a
constant alloc). They touch no weights, no gradients, no loss values, no random
state. The residual definitions and the RK4 stages are preserved exactly. The
`lambda_*` short-circuits and the `use_model_drift` branch keep their current
behavior. Verify by asserting the loss value matches `main` for one batch/seed
before removing the assert.

## Out-of-scope levers (mentioned, not part of this issue)

PINN and Neural-ODE share the same structural cost as KalmanNet Phase 2: a
sequential recurrence over T with small batches, so on GPU the per-step kernel
launch overhead dominates and the GPU sits mostly idle. The mitigations
(device choice, batch size) are estimator-config / device questions handled in
Issue 12 — not here. This issue is only the two numerically-neutral forward-pass
cleanups above.

## Acceptance criteria

- [ ] PINN `_loss` computes `r_data`, `r_dyn`, `r_meas` in a SINGLE T-pass (the
      forward loop), removing the two standalone residual loops; loss value
      bit-matches `main` for the same seed/batch; `lambda_dyn=0`/`lambda_meas=0`
      still skip their terms.
- [ ] Neural-ODE builds the time-feature column once per timestep, not once per
      drift eval; forward output bit-matches `main`.
- [ ] Measurable Lorenz training wall-clock drop for PINN (bigger) and Neural-ODE
      (smaller); report before/after.
- [ ] No change to either `_estimate_sequential_cpu` (inference path is the
      benchmark's measured embedded latency — must NOT be altered).
- [ ] No `tests/`, no `pip install`; lazy `torch` imports preserved.
