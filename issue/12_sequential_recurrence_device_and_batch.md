# Issue 12: Fix the sequential-recurrence GPU stall (device split + batch sizing)

The remaining slowness — after the precompute (Issue 9/10) and forward-fusion
(Issue 11) wins — is structural and shared by every estimator that has an
**irreducible sequential T-loop** at train time:

- KalmanNet Phase 2 (`_run_sequence_vectorized`, [`kalmannet.py:317-335`](../estimators/neural/kalmannet.py#L317-L335)) — free-running recurrence.
- PINN forward ([`pinn.py:139`](../estimators/neural/pinn.py#L139)).
- Neural-ODE forward ([`neural_ode.py:209`](../estimators/neural/neural_ode.py#L209)).

Each iterates T=200 timesteps, and every step depends on the network's own
previous output, so the loop **cannot** be parallelized over T or cached. With the
Lorenz config (`batch_size=32`, hidden=64, nx=3) the per-step compute is tiny, so
on GPU each step pays full kernel-launch latency (~10-20µs) while the math finishes
in nanoseconds — **the GPU is idle >95% of the time and a CPU is often faster.**
Lorenz is the worst case because (a) its `batch_size=32` is the smallest of all
levels (linear uses 512) → 4-16x more loop iterations per epoch, and (b) its `f`
is a 4-stage RK4 → more tiny kernels per step.

This issue is about choosing WHERE these loops run and HOW big the batch is. It
changes no math — only execution location and batching.

## Lever 1 (recommended, lowest risk): run the sequential phase on CPU

For a 200-step recurrence over a `[32, 3]` state with a 64-wide GRU, the CPU has
no launch overhead and the tensors fit in cache; it frequently **beats** the GPU.
This also matches the benchmark's deployment story — `estimate()` already runs
strictly on CPU ([`kalmannet.py:680`](../estimators/neural/kalmannet.py#L680),
[`_neural_base.py:127`](../estimators/neural/_neural_base.py#L127)).

KalmanNet is the clean case because it ALREADY has two phases on (potentially)
different needs: Phase 1 (one big parallel `[B,T,*]` GRU matmul) genuinely wants
the GPU; Phase 2 (the sequential recurrence) often wants the CPU. The current
single `device` ([`kalmannet.py:261-265`](../estimators/neural/kalmannet.py#L261-L265))
forces both onto one device and can't express this.

### Design: add `phase2_device` to KalmanNet

- `device` keeps its meaning: the primary / Phase-1 device (auto → cuda).
- `phase2_device: Optional[str] = None`:
  - `None` → default to `"cpu"` (the automatic win for the sequential recurrence).
  - explicit (`"cuda"`/`"cpu"`) → honored exactly (escape hatch for large-batch
    runs where GPU Phase 2 wins).
- Resolution: `phase1 = _training_device()` as today; `phase2 = phase2_device or "cpu"`.

Correctness requirements (real-life, must get right):
- Move the network to `phase2_device` **between** phases (`network.to(phase2_dev)`),
  and build Phase 2's optimizer **after** the move — Adam's state tensors must live
  on the params' device. `_run_phase` already creates a fresh optimizer per phase
  ([`kalmannet.py:598`](../estimators/neural/kalmannet.py#L598)), so this slots in.
- Clear the compiled Phase-1 graphs before the move (they are device-specialized);
  `fit()` already nulls them at the end ([`kalmannet.py:676-677`](../estimators/neural/kalmannet.py#L676-L677)).
- Phase-2 batches must `.to(phase2_device)`; `_run_epoch` already takes a `device`
  arg — pass `phase2_device` for the Phase-2 call.
- Phase-1 best-checkpoint tensors are cloned to CPU already
  ([`kalmannet.py:622`](../estimators/neural/kalmannet.py#L622)), so loading them
  before the move is device-safe.
- Do NOT silently override an explicit `device="cuda"` for Phase 2 — only default
  to CPU when `phase2_device is None`. (Surprising silent overrides are worse than
  a slow default the user can see and change.)

PINN / Neural-ODE are single-phase (they inherit `SequentialNeuralFilter.fit`,
[`_neural_base.py:174`](../estimators/neural/_neural_base.py#L174)), so "phase 2"
doesn't apply. For them the lever is simply the existing `device` kwarg: allow
running the whole fit on CPU when that's faster for the level. No code change
needed beyond letting the notebook config pass `device="cpu"` per (level,
estimator) — verify `_training_device` honors it (it does). Consider documenting
"CPU may be faster for short/narrow sequential levels" near the configs.

### Small free win while in KalmanNet Phase 2

[`kalmannet.py:318`](../estimators/neural/kalmannet.py#L318) calls
`float(timestamps[t])` inside the T-loop. Phase 1 already hoists this
(`ts = timestamps.tolist()`, [`kalmannet.py:400`](../estimators/neural/kalmannet.py#L400)).
Hoist it in Phase 2 too — one `tolist()` before the loop, index `ts[t]` inside.
Avoids a per-step scalar extraction (and a host sync if `timestamps` is ever on
GPU). Numerically identical.

## Lever 2 (use with care — has a real learning caveat): batch size

Bigger batches amortize the per-step launch overhead (the overhead is per
batch-per-step, so a 2x batch nearly halves the launch count per epoch). BUT for
**Lorenz specifically** the train split is only ~350 trajectories (num_trajectories
=500, 70/15/15 — [`lorenz.py:112`](../benchmark_levels/lorenz.py#L112),
[`base.py:11-24`](../benchmark_levels/base.py#L11-L24)):

- `batch_size=32` → ~11 batches/epoch (≈ healthy update count).
- `batch_size=256` → ~2 batches/epoch ⇒ ~2 optimizer steps/epoch ⇒ ~40 total steps
  over 20 epochs. **The model won't train**, regardless of LR. This is an
  update-COUNT problem, not the generalization-gap problem.

So:
- A **modest bump (32 → 64, maybe 96)** is the sweet spot for Lorenz: ~2-3x fewer
  launches/epoch, still ~4-6 batches/epoch, low risk. Pair with a small LR bump
  (linear scaling rule: ×k batch ⇒ ~×k LR, with the existing cosine schedule
  annealing it) — e.g. `lr 5e-4 → ~1e-3` for a 2x batch.
- **Do NOT crank to 256+ on Lorenz.** Too few optimizer steps on 350 train
  trajectories. (Levels with more trajectories / a bigger train split can take a
  larger batch — judge per level.)
- Treat batch/LR as **empirical tuning in the notebook config**
  (`KALMANNET_CONFIGS` etc.), not a code change, and confirm RMSE doesn't regress.

Lever 1 (CPU for the sequential phase) is preferred precisely because it has
**zero** learning-dynamics risk — it changes only where the math runs, not the
optimization. Reach for Lever 2 only for incremental gains, and validate RMSE.

## Acceptance criteria

- [ ] KalmanNet gains `phase2_device` (default `None` → CPU); Phase 1 runs on
      `device`, Phase 2 on `phase2_device`; explicit values honored, no silent
      override of an explicit `device`.
- [ ] Network + per-phase optimizer correctly placed across the inter-phase device
      move; training runs end-to-end with `device="cuda", phase2_device=None`
      (Phase 1 GPU, Phase 2 CPU) and produces RMSE within noise of an all-CPU and
      an all-GPU run for the same seed.
- [ ] KalmanNet Phase 2 hoists `timestamps.tolist()` out of the T-loop.
- [ ] PINN / Neural-ODE confirmed to honor `device="cpu"` for a full-CPU fit when
      the notebook config requests it; doc note added near the configs.
- [ ] Notebook config: Lorenz batch/LR tuned conservatively (e.g. 32→64 + LR bump)
      IF it helps, with RMSE confirmed non-regressed; explicitly NOT set to a batch
      so large it starves the optimizer.
- [ ] A short timing comparison (Lorenz, per estimator) GPU-only vs. the new
      default, recorded in the PR description.
- [ ] No `tests/`, no `pip install`.

## Depends on / relates to

- Independent of Issues 9-11 but most meaningful after them (those remove the
  cacheable/fusable cost, leaving this irreducible-recurrence cost as the
  dominant remainder). Land 9-11 first, then measure, then apply this.
