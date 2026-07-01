# Issue 13: Transformer & Mamba are teacher-forced-only — a real train/inference (exposure-bias) mismatch

The user's concern is correct **for two of the five neural filters**. The
Transformer and Mamba build their innovation-feature INPUT from the
**ground-truth previous state** at every timestep during training, but at
inference they build the same input from the **model's own previous estimate**.
Training never once sees its own error fed back in, so at deployment the errors
compound (error accumulation) on a distribution the model was never trained on —
the classic *exposure bias* of teacher forcing. The other three filters (PINN,
Neural-ODE, KalmanNet) do **not** have this problem, or already mitigate it.

## Where the mismatch lives

**Transformer** — [`transformer.py:194-200`](../estimators/neural/transformer.py#L194-L200), training:
```python
x_prev = torch.zeros_like(states)
x_prev[:, 1:, :] = states[:, :-1, :]          # GROUND-TRUTH previous state
x_pred = torch.stack([torch_f(x_prev[:, t, :], ts[t]) for t in range(T)], dim=1)
y_pred = torch.stack([torch_h(x_pred[:, t, :], ts[t]) for t in range(T)], dim=1)
innovation = observations - y_pred
feats = torch.cat([observations, innovation, x_pred, dt_col], dim=-1)
```
vs. inference — [`transformer.py:222-235`](../estimators/neural/transformer.py#L222-L235):
```python
for t in range(T):
    x_pred = np.asarray(f(x_prev, t_val))     # x_prev = model's OWN x_hat_{t-1}
    y_pred = np.asarray(h(x_pred, t_val))
    innovation = y_i[t] - y_pred
    ...
    x_prev = x_hat                            # feeds its own output back
```

**Mamba** — byte-for-byte the same split:
[`mamba.py:323-328`](../estimators/neural/mamba.py#L323-L328) (train, GT-prev) vs.
[`mamba.py:345-364`](../estimators/neural/mamba.py#L345-L364) (infer, `x_prev = x_hat`).

In both, the network conditions on `[y, innovation, x_pred, dt]` where
`x_pred = f(x_prev)`. During training `x_prev` is the true state; during inference
`x_prev` is the running estimate. The two distributions of `x_pred`/`innovation`
diverge exactly as the filter accumulates error, and the model has never been
optimized against that regime.

Note this is only active with `use_innovation_features=True` (the default). With
`use_innovation_features=False` the network sees raw `[y, dt]` only — no fed-back
state, so **no exposure bias** (but also no process-model conditioning).

## Why the other three are fine

- **PINN** — `_forward_train` is already **free-running**:
  [`pinn.py:139-146`](../estimators/neural/pinn.py#L139-L146) loops `t`, sets
  `x = x_pred + dx` and feeds that `x` into the next `f(x)`. Training input =
  inference input. No mismatch.
- **Neural-ODE** — same story:
  [`neural_ode.py:209-217`](../estimators/neural/neural_ode.py#L209-L217) carries
  `x = x_post` forward through the learned RK4. Free-running at train time. No
  mismatch.
- **KalmanNet** — has an explicit **two-phase curriculum**
  ([`kalmannet.py:136-150`](../estimators/neural/kalmannet.py#L136-L150)): Phase 1
  teacher-forced (fast, parallel warm-start), then **Phase 2 free-running**
  (`_run_sequence_vectorized`), which trains on the self-propagated state and is
  the "true" objective. With `curriculum_epochs=0` it is free-running only. So it
  already trains the way it is deployed.

So the fleet splits cleanly: **PINN / Neural-ODE / KalmanNet train (at least
partly) free-running; Transformer / Mamba train teacher-forced-only.** That is the
gap.

## Why this matters for the benchmark's claim

The benchmark measures deployment behavior (strictly-sequential CPU inference,
Issue 0). A teacher-forced-only model's reported RMSE is optimistic relative to
what any *architecturally equivalent, honestly-trained* model would achieve,
because it was never asked to recover from its own drift. On stable/observable
levels the two-distribution gap is small; on chaotic (Lorenz) or
weakly-observable levels it can be large and it specifically penalizes the two
models the benchmark might otherwise showcase as the strongest sequence models.
Reporting them alongside free-running KalmanNet/PINN/Neural-ODE without noting the
training asymmetry is not apples-to-apples.

## What "teacher forcing is fine for a Transformer" misses here

In NLP, teacher-forced training of a decoder is standard because at inference you
still feed **ground-truth-shaped tokens you actually observed** (the generated
prefix is in-distribution once the model is good). Here the fed-back quantity is a
**latent state estimate** with no ground truth ever available at deployment, and
`f`/`h` are nonlinear — small state errors map to out-of-distribution
`x_pred`/`innovation`. This is the recursive-estimation version of exposure bias,
where it genuinely bites; it is not the benign NLP case.

## Scope (options — pick per estimator, or make it configurable)

The Transformer's parallel attention forward and Mamba's parallel scan are the
reason they were made teacher-forced (both parallelize over T only when the input
is weight-independent). Fixing the mismatch means giving up some of that
parallelism or approximating it. Options, cheapest first:

### Option A — Free-running fine-tune phase (mirror KalmanNet's curriculum)
Add a second training phase that runs the network on its **own** previous
estimate. For the Transformer/Mamba this is a sequential T-loop at train time
(like KalmanNet Phase 2 / PINN / Neural-ODE already are). Keep the current
teacher-forced pass as a Phase-1 warm-start (it is cheap and parallel), then
anneal into free-running for the last `curriculum_epochs`. This directly reuses
the KalmanNet pattern and the base `fit()` two-phase machinery.
- Pro: closes the gap, keeps the fast warm-start, consistent with KalmanNet.
- Con: the free-running phase loses the parallel-over-T speedup (this is the
  irreducible sequential cost already accepted for PINN/Neural-ODE/KalmanNet
  Phase 2 in Issue 12).

### Option B — Scheduled sampling
With probability `p_t` (annealed from 0 → high over training) replace the
ground-truth `x_prev` with the model's own estimate when building the feature.
For the parallel models this still needs a sequential pass to get "own estimate",
so in practice it reduces to Option A with a per-step mix. More knobs, similar
cost; only worth it if Option A overfits to full free-running too fast.

### Option C — Document-and-measure (minimum honest bar)
If the sequential fine-tune is deemed too expensive for these two models, at
minimum: (1) state in each estimator's docstring that it is **teacher-forced-only
and subject to exposure bias**, and (2) add a benchmark diagnostic that reports,
per level, the RMSE gap between teacher-forced eval and free-running eval for
these models so the mismatch is visible rather than hidden. This does not fix the
model but stops the results from being silently optimistic.

## Real-life correctness checklist

- [ ] Any free-running training phase must feed the **model's own** `x_hat_{t-1}`
      into `f`/`h` to build `x_pred`/`innovation` — identical construction to
      `_estimate_sequential_cpu`, so train and infer match bit-for-bit in method.
- [ ] The teacher-forced Phase-1 warm-start (if kept) must remain gated on
      `use_innovation_features=True`; the raw-`[y,dt]` branch needs no change.
- [ ] `use_innovation_features=False` path is unaffected (already no exposure
      bias).
- [ ] No change to PINN / Neural-ODE / KalmanNet training — they are already
      free-running (or curriculum'd). Do not "fix" a non-problem.
- [ ] Interacts with Issue 9's caching: the cached teacher-forced features are
      only valid for the Phase-1 warm-start, NOT for a free-running phase (whose
      input depends on the network's output and cannot be cached — see Issue 11).

## Acceptance criteria

- [ ] Transformer and Mamba either (A/B) train with a free-running phase whose
      input construction matches `_estimate_sequential_cpu`, or (C) are explicitly
      documented as teacher-forced-only with a reported teacher-forced-vs-free-
      running RMSE gap per level.
- [ ] If A/B: report before/after deployment RMSE on each level (expect the
      largest improvement on Lorenz / weakly-observable levels; ~no change on
      easy linear levels), confirming the gap was real and is closed.
- [ ] No `tests/`, no `pip install`; lazy `torch` imports as elsewhere.

## Out of scope

- KalmanNet, PINN, Neural-ODE training changes (already free-running/curriculum).
- The parallel-scan / attention math itself (Issues 9-12) — unchanged.
- `use_innovation_features=False` behavior.
