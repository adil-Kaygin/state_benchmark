# Issue 10: Vectorize the teacher-forced f/h over B*T for time-independent levels

This is a follow-up sharpening of Issue 9. Even after the teacher-forced
predictions are cached once per fit (Issue 9), the **one-time** build still runs a
Python `for t in range(T)` loop that stacks per-step `f`/`h` calls (see the helper
in Issue 9 and [`kalmannet.py:401-406`](../estimators/neural/kalmannet.py#L401-L406)).
The per-step loop exists for ONE reason: `f`/`h` take a scalar `t`, and the
nonlinear level's `f` actually uses it (`8*cos(1.2*t)` at
[`_torch_dynamics.py:62`](../benchmark_levels/_torch_dynamics.py#L62)). For every
OTHER level the dynamics are **time-independent** and the scalar `t` is ignored:

- linear — [`_torch_dynamics.py:29-37`](../benchmark_levels/_torch_dynamics.py#L29-L37): `x @ F.T`, no `t`.
- pendulum — [`_torch_dynamics.py:45-53`](../benchmark_levels/_torch_dynamics.py#L45-L53): no `t`.
- lorenz — [`_torch_dynamics.py:86-97`](../benchmark_levels/_torch_dynamics.py#L86-L97): RK4 uses only `dt`, never `t`.

For those, the whole T-loop collapses: reshape `x_prev` `[B,T,nx] -> [B*T,nx]`,
call `torch_f` **once**, reshape back. One kernel launch set instead of T. On
Lorenz this turns the 200-step RK4 build into a single batched RK4 over `B*T`
rows.

## Why this is safe (real-life guarantee)

For a time-independent `f`, `f(x, t)` returns the same value for every `t`, so
flattening B and T into one batch dimension and calling `f` once is **exactly
equal** to looping and calling it per step — the calls are independent and
`t`-invariant. The level's `dt` (Lorenz/pendulum) is baked into the closure at
build time, not passed as `t`, so it is preserved automatically.

The danger is doing this blindly on a `t`-dependent level: flattening would feed a
single (wrong/zero) `t` to all `B*T` rows and silently corrupt the nonlinear
level's drift. So this MUST be gated, never blanket-applied.

## The gating problem and the recommended fix

There is currently no flag on `TorchDynamics` saying whether `f`/`h` depend on `t`.
Two acceptable approaches:

1. **Declarative flag (preferred).** Add `time_invariant: bool` (or
   `f_uses_time`/`h_uses_time`) to the `TorchDynamics` dataclass in
   [`benchmark_levels/base.py`](../benchmark_levels/base.py), set `True` for
   linear/pendulum/lorenz builders and `False` for nonlinear in
   [`_torch_dynamics.py`](../benchmark_levels/_torch_dynamics.py). The helper
   branches on it: flatten-and-call-once when invariant, per-step loop otherwise.
   This is explicit, self-documenting, and fails safe (default to `False` /
   per-step loop if a new level forgets to set it).

2. **Pass a `[B*T]` time vector** so even the nonlinear level vectorizes. This
   needs the closures to accept a tensor `t` and broadcast it
   (`torch.cos(1.2 * t_vec)`), i.e. an API change to every `f`/`h` in
   `_torch_dynamics.py`. More work and more risk; only do it if you specifically
   want the nonlinear level vectorized too.

Recommend (1): it gets the win for Lorenz (the level that hurts) with the smallest,
safest change, and leaves the single weird `t`-dependent level on the existing,
known-correct per-step path.

## Scope

- Add the `time_invariant` capability to `TorchDynamics` and set it per level.
- In the Issue-9 `precompute_teacher_forced` helper, branch:
  - invariant → `x_pred = torch_f(x_prev.reshape(B*T, nx), t0).reshape(B, T, nx)`
    then likewise for `y_pred` from `x_pred`.
  - not invariant → existing per-step stack (unchanged).
- Use any representative scalar `t` (e.g. `ts[0]`) for the invariant call, since
  it is ignored; document that it is ignored so a future reader does not "fix" it.

## Real-life correctness checklist

- [ ] Flatten path is **bit-identical** to the per-step path on linear/pendulum/
      lorenz for one batch (assert during dev, then remove).
- [ ] Nonlinear level still uses the per-step loop and is unchanged.
- [ ] `reshape` (not `view`) or a guaranteed-contiguous tensor, so a non-contiguous
      `x_prev` slice cannot raise.
- [ ] Default for any new/forgotten level is the safe per-step path, never a silent
      flatten.

## Acceptance criteria

- [ ] `TorchDynamics` carries an explicit time-invariance capability, set
      correctly for all four current levels.
- [ ] The precompute helper flattens B*T for time-invariant levels and falls back
      to the per-step loop otherwise; results identical to Issue 9 within float
      noise.
- [ ] Extra wall-clock drop on Lorenz precompute vs. Issue 9 alone (report the
      number).
- [ ] No `tests/`, no `pip install`.

## Depends on

- Issue 9 (this optimizes the helper that issue introduces). Land 9 first.
