# Issue 14: Add a small profiling notebook to pinpoint neural-filter training slowness (Lorenz)

Issues 9, 11, and 12 diagnose *why* neural training is slow (launch-bound
sequential T-loops at tiny batch on GPU; redundant teacher-forced recompute;
Neural-ODE's 32× RK4-substep inner loop; per-eval `torch.full` allocs) and
propose fixes. Those diagnoses are currently **reasoned, not measured**. Before
spending effort on 9/11/12 — and to *prove* each one actually helped — we need a
small, self-contained **profiling notebook** that turns the hand-wavy "it's too
slow" into numbers.

See the companion report:
[`reports/neural_training_slowness.md`](../reports/neural_training_slowness.md).

## Goal

One notebook, `notebooks/profile_neural_training.py` (percent-cell `# %%` format,
matching the existing notebooks), that answers, with hard numbers, for each neural
estimator on **Lorenz**:

1. Total `fit()` wall-clock, and per-epoch time.
2. Forward vs. backward vs. optimizer-step split.
3. For KalmanNet: Phase-1 (teacher-forced) vs. Phase-2 (free-running) time.
4. GPU vs. CPU wall-clock for the **same** `fit()` (the Issue-12 lever).
5. Effect of `batch_size` (32 → 128 → 512) on per-epoch time (the other Issue-12 lever).
6. Neural-ODE: effect of `n_substeps` (8 → 4 → 2) on forward time (the Issue-11 multiplier).
7. Approximate kernel-launch count / % GPU-idle, via `torch.profiler`.

The notebook is a **measurement tool**, not a fix. It must not modify any
estimator or benchmark code.

## Why a notebook and not tests

Per project policy there is **no `tests/`** and **no `pip install`** on this
machine ([memory: no-tests-no-pip]). Profiling is exploratory and needs a GPU
session (Colab), so a runnable notebook is the right artifact — it fits the
existing `notebooks/` workflow (`experiment_29_06.py`, `kalmannet_tuning.py`).

## Scope / suggested structure

Keep it small and dependency-free (only what training already imports: `torch`,
`numpy`, the repo's benchmark/estimator modules). Use a **reduced Lorenz config**
so a full sweep runs in minutes, not hours (e.g. `num_trajectories=200`,
`trajectory_length=200`, `num_epochs=3`) — the *relative* costs and the launch
pattern are what matter, not converged RMSE.

### Cell 1 — config & data
- Build a small `LorenzBenchmark`, generate (or load) a train/val split.
- A `DEVICES = ["cuda", "cpu"]` list (guard `cuda` availability).
- A dict of estimator factories: KalmanNet, Neural-ODE, PINN, Transformer, Mamba,
  each constructed from `benchmark.get_filter_model()` with the Lorenz-shaped
  hyperparameters (small epoch count).

### Cell 2 — coarse wall-clock table
- For each estimator × device: time `fit()` end-to-end (`time.perf_counter`,
  `torch.cuda.synchronize()` before/after on GPU). Print a table:
  `estimator | device | total_s | s/epoch`.
- This alone answers "is CPU faster than GPU for the sequential loops?" (Issue 12
  Lever 1) and "which estimator is worst?" (expect Neural-ODE).

### Cell 3 — forward/backward/step breakdown
- Wrap one epoch with manual timers around `_loss(...)` (forward), `loss.backward()`
  (backward), `optimizer.step()`. Use `torch.cuda.synchronize()` to get true GPU
  times. Report the split per estimator.

### Cell 4 — KalmanNet phase split
- Run KalmanNet with `curriculum_epochs>0` and time Phase 1 vs. Phase 2 epochs
  separately (the `history_["phase"]` tag already distinguishes them —
  [`kalmannet.py:214`](../estimators/neural/kalmannet.py#L214)). Confirms the
  Phase-1 recompute cost (Issue 9) and the Phase-2 launch-bound cost (Issue 12).

### Cell 5 — batch-size sweep
- For KalmanNet (or all), sweep `batch_size ∈ {32,128,512}`, report s/epoch on
  GPU. Demonstrates the Issue-12 Lever-2 amortization.

### Cell 6 — Neural-ODE n_substeps sweep
- Sweep `n_substeps ∈ {2,4,8}`, report forward s/epoch. Demonstrates the linear
  blow-up (Issue 11 §Neural-ODE) — expect ~4× from 2→8.

### Cell 7 — torch.profiler launch/idle
- Run one `fit()` step under `torch.profiler.profile(activities=[CPU,CUDA],
  record_shapes=False)`; print `key_averages().table(sort_by="cuda_time_total")`
  and the CPU-vs-CUDA total. Surfaces the **kernel-launch-bound** signature
  (huge `cudaLaunchKernel` count, tiny per-kernel CUDA time, GPU mostly idle) that
  Issues 11/12 assert. This is the definitive evidence.

## Real-life correctness / hygiene

- [ ] Notebook **only measures** — imports and calls existing `fit()`/`estimate()`;
      changes no estimator, benchmark, or base-class code.
- [ ] Reduced Lorenz config so the whole notebook runs in a few minutes; a comment
      states it profiles *relative* cost, not accuracy.
- [ ] GPU timings use `torch.cuda.synchronize()` around the timed region (else the
      async queue makes GPU times meaningless).
- [ ] `cuda` paths guarded by `torch.cuda.is_available()`; notebook still runs
      CPU-only (which is itself a data point).
- [ ] No `tests/`, no `pip install`; lazy `torch` import inside cells, matching the
      rest of `notebooks/`.
- [ ] Percent-cell (`# %%`) format so it opens as a notebook in the existing
      workflow.

## Acceptance criteria

- [ ] `notebooks/profile_neural_training.py` exists and runs top-to-bottom on both
      CPU and (if present) GPU without editing any non-notebook file.
- [ ] Produces: (a) the wall-clock table (est × device), (b) forward/backward/step
      split, (c) KalmanNet phase split, (d) batch-size sweep, (e) Neural-ODE
      n_substeps sweep, (f) a `torch.profiler` table showing the launch-bound /
      GPU-idle signature.
- [ ] The output confirms or refutes each hypothesis in Issues 9/11/12 with a
      number (esp.: Neural-ODE dominated by drift-eval count; KalmanNet Phase-2
      launch-bound; CPU-vs-GPU crossover for the sequential loops).
- [ ] Becomes the before/after harness: re-running it after 9/11/12 land shows the
      wall-clock drop.

## Out of scope

- Implementing the fixes themselves (Issues 9, 11, 12) — this issue only measures.
- Changing any estimator, benchmark, or `_neural_base` code.
- Accuracy/RMSE tuning (this notebook profiles speed only).
