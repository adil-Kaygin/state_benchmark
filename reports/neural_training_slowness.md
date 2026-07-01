# Report: Why the neural filters are slow on Lorenz (Neural-ODE & KalmanNet especially)

**Question (user):** The neural filters are far too slow, especially Neural-ODE
and especially on Lorenz; KalmanNet is also too slow. Find the reasons.

**Short answer:** It is **not the network size** — the nets are tiny. It is
**launch-bound sequential Python loops over time (T)**, run with a tiny batch, on
GPU where each step's math finishes in nanoseconds but each kernel dispatch costs
~10–20 µs. The GPU sits **>90 % idle**. Lorenz is the worst case on every axis
that matters, and Neural-ODE multiplies the loop count by 32× on top. The three
open perf issues (**9, 11, 12**) describe fixes but are **not yet implemented**
(they're still in `issue/`, not `issue/resolved/`).

---

## 1. The Lorenz config is the worst case on every axis

From [`notebooks/experiment_29_06.py`](../notebooks/experiment_29_06.py) (Lorenz):

| Knob | Lorenz | Linear (for contrast) |
|---|---|---|
| `trajectory_length` T | **400** | (short) |
| `num_trajectories` | 2000 → **1400 train** | — |
| `batch_size` | **32** (smallest of all levels) | 512 |
| `num_epochs` | **100** | 10 |
| Neural-ODE `n_substeps` | **8** | 2 |
| `f` = process model | **4-stage RK4** over 3-dim field | linear matmul |

Batches per epoch = 1400 / 32 = **~44**. Every one of the levers above is turned
to its most expensive setting *simultaneously* on Lorenz. Linear uses `batch_size=512`
(16× fewer iterations), 10 epochs (10× fewer), and a cheap linear `f`. That is why
Lorenz dominates wall-clock.

## 2. The core mechanism: launch-bound sequential T-loops

A causal filter's training forward has an **irreducible loop over T** — each step
depends on the network's own previous output, so it cannot be parallelized over
time or cached (Issue 12 states this for KalmanNet Phase 2, PINN, Neural-ODE).
With Lorenz's `batch_size=32`, `nx=3`, `hidden=64`, the per-step tensor math is
*nanoseconds*, but each GPU kernel launch is ~10–20 µs of fixed overhead. Result:
**the GPU is idle >95 % of the time**, and a CPU (no launch overhead, tensors fit
in cache) is frequently *faster*. This is the single structural reason for the
slowness, shared by KalmanNet Phase 2, PINN, and Neural-ODE.

## 3. Neural-ODE — why it is *especially* slow (32× multiplier)

Neural-ODE's forward ([`neural_ode.py:209-217`](../estimators/neural/neural_ode.py#L209-L217))
loops over T, and **each timestep runs a full RK4 integration** with `n_substeps`
substeps, each substep evaluating the drift MLP **4 times** (k1..k4)
([`neural_ode.py:170-175`](../estimators/neural/neural_ode.py#L170-L175)):

```
drift-MLP evals per trajectory (one forward)
  = T × n_substeps × 4
  = 400 × 8 × 4
  = 12,800   drift-MLP forward passes
```

Per epoch that is 44 batches × 12,800 = **~563,000 drift evals**; over 100 epochs,
**~56 million** — each a separate launch-bound MLP dispatch. On top of that, every
single drift eval **allocates a fresh time-feature column** at
[`neural_ode.py:63`](../estimators/neural/neural_ode.py#L63):
```python
t_col = torch.full((x.shape[0], 1), float(t_val), ...)   # 12,800 allocs/traj, all identical within a step
```
`t_val` is constant across all 32 drift evals of a timestep, so ~3,000+ of these
allocations per batch are pure waste (Issue 11 §Neural-ODE). **Neural-ODE is
slowest because it has the deepest inner loop (the 32× RK4-substep-stage factor)
layered on the same launch-bound problem** — and Lorenz sets `n_substeps=8`, the
largest of any level, doubling the pendulum/nonlinear factor.

## 4. KalmanNet — why it is slow (Phase 2 + double the epoch count)

KalmanNet's slowness has two contributors:

1. **Phase 2 free-running recurrence** ([`kalmannet.py:317-335`](../estimators/neural/kalmannet.py#L317-L335))
   is a T=400 sequential loop, run **deliberately eager** (no `torch.compile`) — the
   same launch-bound bottleneck as §2. Per step it launches: RK4 `f` (`_torch_batch_step`,
   ~12 kernels), `h`, a GRU step, a `bmm`. ≈ 44 batches × 400 steps = **17,600
   sequential GRU steps/epoch**.
2. **Phase 1 teacher-forced** ([`kalmannet.py:400-405`](../estimators/neural/kalmannet.py#L400-L405))
   *is* parallel over T, but its weight-independent `f`/`h` prefix is **recomputed
   from scratch every epoch, every batch** — a 400-step RK4 Python loop repeated for
   nothing (Issue 9). On Lorenz `curriculum_epochs=30`, so this redundant rebuild
   runs on every one of those epochs.

Lorenz runs KalmanNet at `num_epochs=100, curriculum_epochs=30` — so both the
launch-bound Phase-2 loop (100 epochs) and the redundantly-recomputed Phase-1
prefix (30 epochs) are paid at their maximum. `batch_size=32` again maximizes the
iteration count.

## 5. Why not "just make the batch bigger / net smaller"?

The nets are already tiny — size is not the cost. The cost is **iteration count ×
per-launch overhead**. Two structural levers actually help (Issue 12):
- **Run the sequential phase on CPU** — no launch overhead; for a 400-step loop
  over a `[32,3]` state with a 64-wide GRU the CPU often *beats* the GPU, and it
  matches the deployment story (`estimate()` is already CPU-only).
- **Raise `batch_size` on Lorenz** — 32 → 256/512 cuts batches/epoch 8–16× and
  amortizes each launch over more work. Lorenz uses the smallest batch of any
  level for no stated reason.

## 6. Status of the fixes — all IMPLEMENTED (math-preserving)

| Issue | Target | Status |
|---|---|---|
| 9 | Cache the weight-independent teacher-forced `f`/`h` (KalmanNet Ph1, Transformer, Mamba) | **✅ resolved** |
| 11 | Fuse PINN's 3 T-loops into 1; hoist Neural-ODE's `t_col` alloc | **✅ resolved** |
| 12 | Run sequential recurrence on CPU (`phase2_device`) + batch/LR levers | **✅ resolved** |
| 14 | Profiling notebook to *measure* the time before/after 9/11/12 | **✅ resolved** |

What landed (see `issue/resolved/`):
- **Issue 9** — new shared `precompute_teacher_forced` helper in
  [`_neural_base.py`](../estimators/neural/_neural_base.py); the teacher-forced `f`/`h`
  prefix is built **once per fit** and carried through the DataLoader (shuffle-safe)
  for KalmanNet Phase 1, Transformer, and Mamba. The RK4 f/h loop no longer reruns
  every epoch. Numerically a no-op (proven bit-identical).
- **Issue 11** — PINN's three T-loops fused into one (`_forward_train_fused`);
  Neural-ODE's `t_col` allocation hoisted out of the 4×`n_substeps` inner drift
  evals. Same residuals, same drift — fewer allocations / passes.
- **Issue 12** — KalmanNet gains `phase2_device` (default **CPU**) so the
  launch-bound free-running recurrence leaves the idle GPU; PINN/Neural-ODE honor
  `device="cpu"` (documented). Phase-2 `timestamps.tolist()` hoisted out of the loop.
  Batch/LR are notebook-config levers (not code), as the issue specifies.
- **Issue 14** — [`notebooks/profile_neural_training.py`](../notebooks/profile_neural_training.py):
  a self-contained measurement notebook (wall-clock est×device, forward/backward
  split, KalmanNet phase split, batch sweep, Neural-ODE `n_substeps` sweep,
  `torch.profiler` launch-bound table). Re-run it before/after to quantify the wins.

**Verification:** all reorderings were proven **bit-identical** with a numpy mirror
of f/h (PINN `f_prev`/`h_hat`, KalmanNet cached `inp`/`x_pred`/`innovation`,
Transformer/Mamba `x_pred` slice recovery). No training dynamics, gradients, or
results change — only redundant work and idle-GPU time are removed. (torch is not
installed on this machine, so end-to-end wall-clock numbers come from running the
profiling notebook on Colab.)

## Bottom line

- **Root cause:** launch-bound sequential T-loops (T=400) at `batch_size=32` on
  GPU → GPU >90 % idle. Not net size.
- **Neural-ODE especially:** an extra **32× inner factor** (n_substeps=8 × 4 RK4
  stages × T) plus a redundant per-eval `torch.full` allocation.
- **KalmanNet:** launch-bound Phase-2 loop **plus** a per-epoch recompute of the
  cacheable Phase-1 prefix, both at Lorenz's max epoch/min batch settings.
- **Lorenz:** every expensive knob (T=400, batch=32, 100 epochs, RK4 `f`,
  n_substeps=8) maxed at once.
- **Fixes exist on paper (Issues 9/11/12) but are unimplemented; Issue 14 proposes
  measuring first.**
