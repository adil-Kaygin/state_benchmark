# Issue 9: Precompute the teacher-forced f/h dynamics once (KalmanNet Phase 1, Transformer, Mamba)

Three neural estimators build a **teacher-forced input** from the ground-truth
states and the benchmark's TRUE `filter_model.torch.f`/`.h`, then feed that input
through their network. The input is **independent of the network weights**, yet it
is **recomputed from scratch every epoch, every batch**. On the Lorenz level the
true `f` is a 4-stage RK4 over a 3-dim field, so each rebuild is a 200-step Python
loop of ~12 tiny GPU kernels — repeated `num_epochs`/`curriculum_epochs` times for
no reason. This is the single biggest avoidable cost on Lorenz for these three
models.

The identical offending pattern appears in all three:

- KalmanNet Phase 1 — [`kalmannet.py:401-406`](../estimators/neural/kalmannet.py#L401-L406):
  ```python
  x_pred = torch.stack([_torch_batch_step(torch_f, x_prev[:, t, :], ts[t]) for t in range(T)], dim=1)
  y_pred = torch.stack([_torch_batch_step(torch_h, x_pred[:, t, :], ts[t]) for t in range(T)], dim=1)
  ```
- Transformer — [`transformer.py:196-197`](../estimators/neural/transformer.py#L196-L197) (only when `_use_innovation_features=True`):
  ```python
  x_pred = torch.stack([torch_f(x_prev[:, t, :], ts[t]) for t in range(T)], dim=1)
  y_pred = torch.stack([torch_h(x_pred[:, t, :], ts[t]) for t in range(T)], dim=1)
  ```
- Mamba — [`mamba.py:325-326`](../estimators/neural/mamba.py#L325-L326) (byte-for-byte the same two lines).

In every case `x_prev` is the ground-truth state shifted right by one
(`x_prev[:, 1:] = states[:, :-1]`, zeros at t=0) and `torch_f`/`torch_h` are the
level's fixed dynamics. **Nothing here touches the network.** So `x_pred`,
`y_pred`, and the derived `innovation`/`dx_prev` are constants for the entire fit.

## Why this is correct to cache (the real-life guarantee)

The cached tensors are a pure function of `(states, timestamps, filter_model)` —
all three are frozen for the duration of a `fit()`. The network's weights, the
optimizer, and the random seed never enter the computation. Therefore caching is
**numerically identical** to recomputing: same bits, every epoch. This is not an
approximation and changes no training dynamics, gradients, or results — it only
removes redundant work. (Contrast with PINN / Neural-ODE, whose per-step `f`/RK4
depends on the *network's own* previous output and so is genuinely not cacheable —
see Issue 11.)

The one subtlety: the cache must be computed **on the training device (GPU)**, once,
before the epoch loop — not on CPU (that would just move the RK4 cost, not remove
it) and not per batch. Memory cost is `[N, T, ny+nx]` + `[N, T, nx]`, the same
order as the `(obs, states)` tensors already resident, so there is no memory
concern at the benchmark's sizes (N≈350, T=200).

## Scope

### 1. A shared precompute helper (single source of truth)

Because the three call sites are identical, add ONE helper rather than three
copies. Suggested home: [`estimators/neural/_neural_base.py`](../estimators/neural/_neural_base.py)
(or a small `_teacher_forcing.py` next to it), imported by all three.

```python
def precompute_teacher_forced(torch_f, torch_h, states, timestamps):
    """Build the weight-independent teacher-forced predictions for a whole
    batch of trajectories in one shot.

      x_prev[:, t]  = states[:, t-1]  (zeros at t=0)        # GT shifted right
      x_pred[:, t]  = f(x_prev[:, t], t)
      y_pred[:, t]  = h(x_pred[:, t], t)

    Returns (x_pred [B,T,nx], y_pred [B,T,ny]). Runs on states.device.

    `f`/`h` take a SCALAR t per step (the nonlinear level's f uses cos(1.2*t)),
    so t cannot be folded into one [B*T] call in general -- keep the per-step
    stack here. The point of this helper is not to remove the T-loop but to run
    it ONCE per fit instead of once per epoch (the caller caches the result).
    """
    import torch
    B, T, nx = states.shape
    x_prev = torch.zeros_like(states)
    x_prev[:, 1:, :] = states[:, :-1, :]
    ts = timestamps.tolist()
    x_pred = torch.stack([torch_f(x_prev[:, t, :], ts[t]) for t in range(T)], dim=1)
    y_pred = torch.stack([torch_h(x_pred[:, t, :], ts[t]) for t in range(T)], dim=1)
    return x_pred, y_pred
```

NOTE: keep the per-step scalar-`t` loop. A blanket `B*T` flatten would break the
nonlinear level (its `f`/`h` genuinely depend on scalar `t`). If you also want the
flatten optimization for the t-independent levels, do it as a separate, opt-in
follow-up (Issue 10) — do NOT couple it to this change.

### 2. KalmanNet Phase 1 consumes a cached input

Phase 1 (`_run_sequence_teacher_forced`, [`kalmannet.py:346`](../estimators/neural/kalmannet.py#L346))
must be split so the **cacheable prefix** (everything up to and including
`inp = cat([innovation, dx_prev])` and `x_pred`) is computed **once in `fit()`**
and reused across all `curriculum_epochs`. The **non-cacheable tail** stays per
batch/epoch: `gru(inp) -> out_norm -> fc_gain -> einsum correction -> x_pred + correction`.

Cleanest implementation: build a dedicated Phase-1 `DataLoader` over a
`TensorDataset(inp_cached, x_pred_cached, states)` so each batch already carries
its precomputed `inp`/`x_pred`. The Phase-1 `forward_fn` then reads those tensors
instead of recomputing them. Phase 2 (`_run_sequence_vectorized`) is untouched —
it is the free-running recurrence and has nothing to cache.

Make sure the cache is built on `device` (the training device) right after the
network is moved there, and BEFORE the `_run_phase(1, ...)` call. The
`torch.compile` Phase-1 wrappers ([`kalmannet.py:536-544`](../estimators/neural/kalmannet.py#L536-L544))
should now wrap the cheap tail; that is fine and still valid.

### 3. Transformer & Mamba consume a cached input

Both compute the teacher-forced features inside `_forward_train`
([`transformer.py:196-197`](../estimators/neural/transformer.py#L196-L197),
[`mamba.py:325-326`](../estimators/neural/mamba.py#L325-L326)). Cache `x_pred`/
`y_pred` (hence `innovation`, and the full `feats` tensor) once per fit and slice
per batch. For the Transformer this is gated on `_use_innovation_features=True`
(the `else` branch at [`transformer.py:180-184`](../estimators/neural/transformer.py#L180-L184)
uses a zero `x_pred` and needs no caching). Because both inherit
`SequentialNeuralFilter.fit` ([`_neural_base.py:174`](../estimators/neural/_neural_base.py#L174)),
the cleanest route is a small hook the base `fit()` can call to let a subclass
precompute per-split tensors and swap in an augmented `TensorDataset` /
`forward_fn`, OR each subclass overriding `fit` minimally. Pick whichever keeps
the base loop readable; do not duplicate the whole `fit()` body.

## Real-life correctness checklist (must hold)

- [ ] Cached `x_pred`/`y_pred`/`inp` are **bit-identical** to the current per-epoch
      recompute for the same `(states, timestamps, filter_model)` — verify by
      printing/asserting equality on one batch during a dev run (then remove the
      assert). No tolerance fudging.
- [ ] Cache is built on the **training device**, once, before the epoch loop —
      never per batch, never on CPU.
- [ ] Shuffling still works: the cached tensors must be indexed by the SAME
      sample order as `states`/`obs` (put them in the same `TensorDataset` so the
      `DataLoader`'s shuffle permutes them together — do NOT cache by a separate
      index that can desync from a shuffled loader).
- [ ] Validation split gets its own cache (different `states`/`timestamps`);
      don't accidentally reuse the train cache for val.
- [ ] The nonlinear level (scalar-`t` dependent `f`/`h`) still trains correctly —
      the per-step loop in the helper preserves the `ts[t]` argument.
- [ ] No change to Phase 2 (KalmanNet), to the non-innovation Transformer branch,
      or to ANY `_estimate_sequential_cpu` (inference path is out of scope).

## Acceptance criteria

- [ ] One shared `precompute_teacher_forced` helper; KalmanNet Phase 1,
      Transformer (innovation mode), and Mamba all call it and **cache the result
      once per fit**, slicing per batch instead of recomputing per epoch.
- [ ] RMSE/loss curves on every level are unchanged within float noise vs.
      `main` for the same seed (numerically a no-op).
- [ ] Measurable wall-clock drop in Lorenz training for KalmanNet (Phase 1),
      Transformer, and Mamba (report before/after fit time on Lorenz).
- [ ] No `tests/`, no `pip install`; lazy `torch` import inside the helper
      (matches the rest of `_neural_base.py`).

## Out of scope

- Flattening the t-independent levels' f/h to a single `B*T` call (Issue 10).
- Phase-2 device split / batch tuning for KalmanNet (Issue 12).
- PINN / Neural-ODE forward-loop fusion — their loops are network-dependent and
  cannot use this cache (Issue 11).
