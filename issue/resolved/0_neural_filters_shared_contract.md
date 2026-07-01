# Issue 0: Shared Contract for Neural Filter Estimators (read first)

This is the common reference for the four new neural estimators
([Neural-ODE](1_neural_ode_filter.md), [PINN](2_pinn_filter.md),
[Transformer](3_transformer_filter.md), [Mamba](4_mamba_filter.md)).
Each of those issues assumes everything below and only states its own deltas.

## Why this exists

The benchmark already ships one fully-worked neural estimator,
[`KalmanNetEstimator`](../estimators/neural/kalmannet.py). It is the canonical
template: every new neural filter must satisfy the **same interfaces**, the
**same hardware-split deployment contract**, and the **same fail-fast rules** so
that `experiments/runner.py` and the notebooks can swap them in without special
casing. Read `kalmannet.py` end to end before implementing any of the four.

## The estimator interface (hard requirement)

Every estimator subclasses [`BaseEstimator`](../estimators/base.py) and
implements exactly these:

```python
estimator_name -> str          # stable id, e.g. "neural_ode"
estimator_type -> str          # "neural"
fit(train_dataset, val_dataset) -> None
estimate(dataset) -> np.ndarray   # shape [N, T, nx]
save(path: Path) -> None
load(path: Path) -> BaseEstimator  # classmethod
```

`fit`/`estimate` receive a [`TrajectoryDataset`](../datasets/schema.py):

```python
dataset.states        # [N, T, nx]  GROUND TRUTH. fit() may read it; estimate() MUST NOT.
dataset.observations  # [N, T, ny]  the only input estimate() is allowed to consume.
dataset.timestamps    # [T]         shared across trajectories; thread into time-varying f/h.
dataset.metadata      # DatasetMetadata (nx, ny, seed, ...)
```

`estimate()` returns `np.ndarray [N, T, nx]` of state estimates. It must never
read `dataset.states` (the README's core rule: estimators consume `y_t`, never
`x_t`).

## The process model: `FilterModel` (how to use f / h / Q / R)

Construct every estimator with a `filter_model = benchmark.get_filter_model()`
([`FilterModel`](../benchmark_levels/base.py)). Same object EKF/UKF/KalmanNet
get. Relevant fields:

```python
filter_model.f(x, t) -> [nx]     # NumPy process model, single state vector
filter_model.h(x, t) -> [ny]     # NumPy observation model, single state vector
filter_model.F, filter_model.H   # Jacobian callables (may be None on some levels)
filter_model.Q, filter_model.R   # [nx,nx], [ny,ny] process/obs noise covariances
filter_model.x0_mean, x0_cov     # optional initial-state prior
filter_model.torch               # Optional[TorchDynamics]: BATCHED torch f/h, [B,nx]->[B,nx]/[B,ny]
```

- `self._nx = filter_model.Q.shape[0]`, `self._ny = filter_model.R.shape[0]`.
- These four filters are given the **true** model `f, h` (no model mismatch),
  exactly like EKF/UKF. Whether a given architecture *uses* `f`/`h` is part of
  each issue (see per-issue "Process-model usage").
- If an estimator needs `filter_model.torch` for GPU training and it is `None`,
  raise a descriptive `ValueError` naming the missing `TorchDynamics` (copy the
  message style at [`kalmannet.py:295-300`](../estimators/neural/kalmannet.py#L295-L300)).

## The hardware-split deployment contract (NON-NEGOTIABLE)

This is the project's defining methodology and is enforced in code, not just
convention (see [README.md](../README.md) "KalmanNet hardware split"):

- **`fit()` / validation**: fully **batched on the GPU** when available
  (`torch.cuda.is_available()`), one `[B, T, *]` tensor pass per batch, using
  `filter_model.torch` for any process-model evals. No per-row Python loop, no
  NumPy round-trip on the GPU path.
- **`estimate()` / inference**: **strictly sequential on the CPU** — one
  trajectory at a time, one timestep at a time, using the NumPy `filter_model.f`
  / `filter_model.h` on a single state vector. This deliberately simulates
  microprocessor / embedded deployment and is what the latency metric measures.
  Move the network to CPU and `.eval()` inside `estimate()`.

  Mirror [`_run_sequence_sequential_cpu`](../estimators/neural/kalmannet.py#L679-L738)
  and [`estimate`](../estimators/neural/kalmannet.py#L740-L758). **Architectures
  that are not naturally autoregressive (Transformer, Mamba) still run causally
  on CPU at test time** — see those issues for exactly what "sequential" means
  for them (no peeking at future observations; the latency you report is the
  per-step deployment latency).

If a model is mathematically non-recurrent but you batch its `estimate()` on
GPU "because it's faster", you have broken the benchmark. The whole point is to
compare *deployment* latency under identical embedded conditions.

### On the teacher-forcing curriculum (KalmanNet only — DO NOT copy blindly)

KalmanNet's two-phase curriculum
([`_run_sequence_teacher_forced`](../estimators/neural/kalmannet.py#L345-L432))
exists for **one reason**: KalmanNet is an inherently sequential GRU recurrence,
so its free-running training loop cannot be parallelized over T on the GPU.
Teacher forcing makes the inputs independent of the network's own output so the
whole `[B, T, *]` sequence can run in a single parallel GRU call as a warm-start.

**It is a speed hack for a recurrent model, not part of the contract.** For the
four new estimators, prefer the architecture's *native* training-time
parallelism and skip the curriculum entirely:

- **Transformer** — already fully parallel over T (one masked self-attention
  forward pass trains all timesteps at once). No curriculum.
- **Mamba** — train with the **parallel selective-scan / Mamba-2 SSD kernels**
  (the associative scan is parallel over T). No curriculum.
- **Neural-ODE / PINN** — parallel over the batch / collocation dimension; the
  ODE integration is the only sequential part and is handled by the integrator,
  not a teacher-forcing trick.

Only reach for teacher forcing if you deliberately build a sequential recurrence
with no parallel training form — and justify it in the issue if so.

## Fail-fast rules (enforced, not optional)

From [README.md](../README.md) "Architectural rules":

- No silent fallbacks, no dummy returns (`0.0`/`NaN`/`None` for an undefined
  result), no implicit coercions. Bad input / mismatched shapes / scientifically
  unsound config raise a descriptive `ValueError`/`RuntimeError`/`ImportError`
  **immediately**.
- `estimate()` before `fit()` raises `RuntimeError` (see
  [`kalmannet.py:743-744`](../estimators/neural/kalmannet.py#L743-L744)).
- `torch` is imported **lazily inside methods**, never at module top level, so
  importing the estimators package never requires torch
  (`if TYPE_CHECKING: import torch`). Same for any extra dep (e.g. `torchdiffeq`,
  `mamba_ssm`): import inside `fit`/`estimate`, and on `ImportError` raise a
  clear message telling the user what to install — but see the next section.

## Environment constraints (this machine)

- **Do NOT add a `tests/` directory and do NOT run `pip install`** on this
  machine (standing user rule). Issues may *list* an optional dependency, but the
  implementation must degrade to a clear `ImportError` message when it is absent
  rather than assuming it can be installed here. Where an architecture can be
  written in plain PyTorch without a third-party package (e.g. a from-scratch
  selective-scan for Mamba, an explicit RK4 integrator for the ODE), **prefer the
  dependency-free path** and note the trade-off.

## save / load

- `save(path)`: `path.parent.mkdir(parents=True, exist_ok=True)`, then
  `torch.save({...})` a dict containing `state_dict`, `nx`, `ny`,
  `estimator_name`, and every hyperparameter needed to rebuild the network shape
  (mirror [`kalmannet.py:760-775`](../estimators/neural/kalmannet.py#L760-L775)).
- `load(path)`: KalmanNet raises `NotImplementedError` from `load` because it
  needs a `FilterModel` to reconstruct
  ([`kalmannet.py:777-783`](../estimators/neural/kalmannet.py#L777-L783)). You
  may follow that exact pattern (document the rebuild recipe in the message), or
  implement a real `load(cls, path, filter_model)` helper — but keep the
  `BaseEstimator.load(path)` signature satisfied (raise the descriptive
  `NotImplementedError` if you need the extra arg).

## Training niceties to reuse from KalmanNet

These are already solved in `kalmannet.py`; copy the patterns rather than
reinventing: per-epoch `history_` dict (`train_loss`/`val_loss`/`lr`/`epoch`),
best-checkpoint-in-memory by val loss, gradient clipping (`grad_clip_norm`),
optional LR scheduler (`plateau`/`cosine`/`step`), optional early stopping,
NaN/Inf-loss skip, `random_seed` via `torch.manual_seed`, `verbose` printing.
A constructor with `hidden_size`, `learning_rate`, `num_epochs`, `batch_size`,
`device`, `random_seed` is the minimum; add architecture-specific knobs per issue.

## Registration / wiring (do this for each new estimator)

1. Replace the stub class (where one exists: `neural_ode.py`, `transformer.py`)
   or add a new module under `estimators/neural/`.
2. Export it from [`estimators/neural/__init__.py`](../estimators/neural/__init__.py)
   and update `__all__`.
3. The notebooks build estimators in a `KALMANNET_CONFIGS`-style per-level dict
   (see `notebooks/experiment_29_06.py:53-56`, `:126-144`). Each issue gives a
   suggested per-level hyperparameter starting point; add the estimator to that
   construction block so it runs in the standard sweep.

## Definition of done (every issue)

- [ ] Class implements the full `BaseEstimator` interface; `estimate()` returns
      `[N, T, nx]` and never touches `dataset.states`.
- [ ] `fit()`/val run batched on GPU; `estimate()` runs strictly sequential on
      CPU per the hardware-split contract.
- [ ] Fail-fast everywhere (no dummy returns); missing optional dep → clear
      `ImportError`; missing `FilterModel.torch` when required → clear `ValueError`.
- [ ] Exported from `__init__.py`; added to the notebook estimator block with a
      per-level config.
- [ ] No `tests/`, no `pip install` run on this machine.
- [ ] Docstrings explain the math and the GPU-train / CPU-infer split, in the
      style of `kalmannet.py`.
