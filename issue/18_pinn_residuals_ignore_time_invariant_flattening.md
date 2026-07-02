# Issue 18: PINN physics residuals launch f/h per timestep inside the training loop — the one remaining site that ignores `TorchDynamics.time_invariant` flattening

**Severity: high (speed only, math unchanged).** On time-invariant levels the
physics-residual dynamics evaluations — roughly a third to half of the PINN
training forward's kernel launches, and the f-side is a full RK4 on Lorenz —
collapse to two flattened calls per batch. Every other consumer of
teacher-forced dynamics already does this (Issue 10); PINN's loss path was
fused (Issue 11) but never flattened.

## Problem Description

The fused PINN forward
([`pinn.py:139-193`](../estimators/neural/pinn.py#L139-L193)) folds the
r_dyn/r_meas dynamics sweeps into the free-running T-loop: at each step it
evaluates the process model on the previous estimate and the observation model
on the current estimate ([`pinn.py:180-188`](../estimators/neural/pinn.py#L180-L188)).
That was an improvement over the three separate loops it replaced, but it bakes
2T per-step dynamics calls into the loop — per batch, on Lorenz (T=400, RK4
process model ≈ a dozen kernels per call), that is on the order of 5,000+ extra
kernel launches whose inputs are **already fully materialized by the loop
itself**.

The residual inputs are the network's own estimates, so unlike the recursion's
innovation features they cannot be *pre*computed — but they do not need to be
computed *inside* the loop either. Once the estimate sequence is stacked, the
r_dyn term is the process model applied independently to every (trajectory,
timestep) pair of the estimates excluding the last step, and the r_meas term is
the observation model applied to every pair. For a level whose dynamics ignore
the scalar timestep — exactly what `TorchDynamics.time_invariant` certifies
([`base.py:85-92`](../benchmark_levels/base.py#L85-L92)), true for linear,
pendulum, Lorenz, and vehicle-tracking — both are single flattened batch calls
on a [B·T]-row tensor, the same collapse `precompute_teacher_forced` already
performs for the caching estimators
([`_neural_base.py:64-70`](../estimators/neural/_neural_base.py#L64-L70)).
Gradients are identical: the flattened call is the same function applied to the
same tensors, still differentiable through the stacked estimates.

## Impact Analysis

- **Training speed on the dominant levels.** With physics terms active (the
  benchmark's defaults set both lambdas nonzero on every level), the in-loop
  residual evaluations roughly double the per-step dynamics work of the PINN
  forward and add the same again in backward. Flattening removes ~30–40% of
  the launch count per batch on GPU-resident levels and, on CPU, removes 2T
  Python-level dispatches per batch. Expected fit-time reduction for PINN on
  Lorenz-class configs: on the order of 1.3–1.5× — modest next to Issues 15–17
  but obtained with a small, provably math-identical change to one method.
- **Architectural consistency.** `time_invariant` was introduced (Issue 10)
  precisely so that no [B, T] dynamics sweep pays a per-step Python loop when
  the level permits collapsing it. PINN's loss is now the only surviving hot
  path that ignores the flag; leaving it contradicts the optimization contract
  the flag documents and hides a known-cheap win behind a resolved-looking
  issue trail.

## Refactoring Strategy

1. **Move the residual evaluations out of the recursion loop.** The loop's only
   irreducible content is the recursion itself (predict from the previous
   estimate, form the innovation, step the GRU, feed the estimate back). After
   the loop stacks the estimates, compute the r_dyn inputs from the stacked
   estimates shifted by one step and the r_meas inputs from the full stack.
2. **Branch on the level's `time_invariant` flag,** exactly as
   `precompute_teacher_forced` does: when true, flatten [B, T] to one [B·T]
   batch call per residual (the representative scalar timestep is ignored by
   construction); when false (the nonlinear level), keep a per-step stacked
   evaluation over the materialized estimates — still outside the recursion
   loop, preserving current cost with cleaner structure.
3. **Preserve the ablation short-circuit.** The lambda-zero gates must keep
   skipping the corresponding residual evaluation entirely, matching today's
   behavior.
4. **Verify bit-identity, in keeping with the repo's standard.** The stacked
   residual tensors must match the current in-loop accumulations exactly on a
   fixed-seed batch for both a time-invariant level (Lorenz) and the
   time-varying one (nonlinear), and the loss scalar must be unchanged.

## Out of scope

- The recursion loop itself (irreducible; device policy covered by Issue 12).
- Neural-ODE (its RK4 substeps consume the network's own intermediate states —
  nothing to flatten).
- The dead per-step fallback branches in the Transformer/Mamba training
  forwards (unreached when the Issue-9 cache is active; cosmetic).
