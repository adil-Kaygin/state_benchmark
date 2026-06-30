# Issue 8: Checkpoint + skip/resume for neural training (survive Colab disconnects)

Cell 5 of [`notebooks/experiment_30_06.py`](../notebooks/experiment_30_06.py)
trains **every neural estimator in-process, back to back** — 4 levels ×
(KalmanNet, NeuralODE, PINN, Transformer, Mamba) ≈ 20 fits, ~100 min of GPU time
on Colab. When the Colab runtime disconnects (idle timeout, GPU eviction, browser
drop) the whole cell dies and **all of that work is lost**: nothing is written to
disk during training, so a rerun starts from zero.

```python
# Cell 5 today — one disconnect throws away everything trained so far.
for benchmark_name, estimator_set in estimators.items():
    for estimator_name, estimator in estimator_set.items():
        if estimator.estimator_type == "neural":
            estimator.fit(train_ds, val_ds)   # best weights live ONLY in RAM
```

## Why this hurts so much right now

The frustrating part is that the best weights **already exist in memory** and are
**already thrown away on disconnect**. Both training loops track the best
checkpoint by val loss:

- [`estimators/neural/_neural_base.py:239`](../estimators/neural/_neural_base.py#L239)
  — `SequentialNeuralFilter.fit` keeps `self._best_state_dict` (best-by-val-loss),
  reloads it at the end, but never writes it to disk.
- [`estimators/neural/kalmannet.py:620`](../estimators/neural/kalmannet.py#L620)
  — `KalmanNetEstimator.fit` does the same per curriculum phase.

So we have *best weights in RAM* and a *durable filesystem already mounted*
(`DATA_ROOT`/`FIGURES_ROOT` are `./drive/MyDrive/...`, see
[`notebooks/experiment_30_06.py:71-72`](../notebooks/experiment_30_06.py#L71-L72))
— we just never connect the two. `save()` exists on every neural estimator and
writes `state_dict` + hyperparams; the missing piece is a **load-weights path**
and a **Cell-5 skip loop**.

## Scope (agreed): per-estimator skip + resume, checkpoints on Google Drive

Granularity is **per estimator**, not per epoch. After each estimator finishes
training we persist its best weights to Drive; on a rerun, any estimator whose
checkpoint already exists is **loaded from disk and skipped**, so training
resumes at the first not-yet-completed estimator. A disconnect loses at most the
single estimator that was mid-fit. This is ~90% of the benefit of full per-epoch
resume for a fraction of the code, and it touches `fit()` not at all.

Storage: a new `CKPT_ROOT = Path("./drive/MyDrive/checkpoints_30_06")`, one file
per `(level, estimator)` — survives runtime death and full browser disconnect
because it's on the already-mounted Drive.

### 1. A `load_weights(path)` method on the neural estimators

`SequentialNeuralFilter.load()` and `KalmanNetEstimator.load()` deliberately
raise (they need a `FilterModel` to rebuild the net — see
[`_neural_base.py:296`](../estimators/neural/_neural_base.py#L296)). That's fine:
in Cell 4 the estimator is **already constructed** with its `FilterModel`, so we
don't need a classmethod `load`. We need an **instance** method that rebuilds the
network and loads a saved `state_dict` into it:

```python
# on SequentialNeuralFilter (covers NeuralODE / PINN / Transformer / Mamba)
def load_weights(self, path: Path) -> bool:
    """Load a saved best-weights checkpoint into a fresh network on CPU and
    mark the estimator as fit(). Returns False if the file is absent."""
    import torch
    if not Path(path).exists():
        return False
    payload = torch.load(path, map_location="cpu")
    network = self._build_network()
    network.load_state_dict(payload["state_dict"])
    self._network = network.to("cpu")
    self._best_val_loss = float(payload.get("best_val_loss", self._best_val_loss))
    return True
```

Add the matching method to `KalmanNetEstimator` (it has its own `_build_network`
at [`kalmannet.py:267`](../estimators/neural/kalmannet.py#L267) and its own
`save()` at [`kalmannet.py:760`](../estimators/neural/kalmannet.py#L760)). Keep
the existing `save()`; optionally include `best_val_loss` in the `save()` payload
so the loaded estimator reports a meaningful `best_val_loss` (used only for
logging — fail-fast still applies, a corrupt/partial file should raise, not be
silently treated as "no checkpoint").

After `load_weights`, `estimate()` works unchanged: it only needs
`self._network` set (see
[`_neural_base.py:267-280`](../estimators/neural/_neural_base.py#L267-L280)).

### 2. Cell 5 becomes skip-or-train-then-save

```python
# Cell 5: fit each neural estimator ONCE, with skip/resume from Drive.
CKPT_ROOT = Path("./drive/MyDrive/checkpoints_30_06")

for benchmark_name, estimator_set in estimators.items():
    train_ds = datasets[benchmark_name]["train"]
    val_ds = datasets[benchmark_name]["val"]
    for estimator_name, estimator in estimator_set.items():
        if estimator.estimator_type != "neural":
            continue
        ckpt = CKPT_ROOT / benchmark_name / f"{estimator_name}.pt"
        if estimator.load_weights(ckpt):
            print(f"[resume] loaded {estimator_name} on '{benchmark_name}' from {ckpt}")
            continue
        print(f"Training {estimator_name} on '{benchmark_name}'...")
        estimator.fit(train_ds, val_ds)
        estimator.save(ckpt)          # best weights persisted immediately
        print(f"[saved] {estimator_name} on '{benchmark_name}' -> {ckpt}")
```

The `estimator.save(ckpt)` lands **right after each fit**, so the checkpoint is on
Drive before the next (longer) estimator starts. On any rerun the loop fast-skips
everything already on disk and continues from the first missing one.

### 3. Cell 6 must not refit neural estimators

Cell 6 fits the *classical* estimators lazily
([`experiment_30_06.py:231-232`](../notebooks/experiment_30_06.py#L231-L232));
neural ones are already fit (or loaded) by Cell 5, so leave that branch as is. Just
confirm Cell 6 never calls `fit` on a neural estimator — it doesn't today, and the
skip-loaded estimators satisfy `estimate()`'s `self._network is not None` check.

## Edge cases / fail-fast

- **Partial/corrupt checkpoint** (disconnect mid-`torch.save`): `torch.save` is
  not atomic. Write to a temp path and `os.replace()` onto the final name so a
  half-written file never looks complete. A genuinely corrupt file should raise on
  load (do **not** swallow it and silently retrain — that hides a real problem).
- **Changed hyperparameters / config:** a checkpoint is keyed only by
  `(level, estimator)`, so if you change a `*_CONFIGS` entry you must delete the
  stale `.pt` (or bump `CKPT_ROOT`) or you'll load weights for an
  architecture-mismatched network → `load_state_dict` raises, which is the correct
  loud failure. Document this in the cell comment.
- **History for loss curves (Cell 11):** a *loaded* estimator has an empty
  `history_`, so Cell 11 already prints "no training history" and skips it
  ([`experiment_30_06.py:409-412`](../notebooks/experiment_30_06.py#L409-L412)).
  Acceptable for resume; if loss curves matter on resume, also persist `history_`
  in the payload and restore it in `load_weights` (optional, low priority).

## Acceptance criteria

- [ ] `SequentialNeuralFilter` and `KalmanNetEstimator` gain
      `load_weights(path) -> bool` that rebuilds the network and loads a saved
      `state_dict` on CPU, setting `self._network` so `estimate()` works; returns
      `False` only when the file is absent (corrupt file raises).
- [ ] `save()` writes atomically (temp file + `os.replace`) so a disconnect
      mid-save can't leave a checkpoint that loads as valid-but-wrong.
- [ ] Cell 5 skips any `(level, estimator)` whose checkpoint exists, otherwise
      trains and immediately `save()`s to `CKPT_ROOT = ./drive/MyDrive/checkpoints_30_06`.
- [ ] A simulated disconnect (interrupt Cell 5 partway, rerun) resumes at the
      first un-checkpointed estimator; all earlier ones load from Drive without
      retraining.
- [ ] Cell 6/7+ run unchanged on the loaded estimators (RMSE/latency identical to
      a from-scratch run for the same weights).
- [ ] No `tests/`, no `pip install`; lazy `torch` import inside `load_weights`
      (matches the rest of `_neural_base.py`).

## Out of scope (possible follow-ups)

- **Per-epoch resume inside `fit()`** (resume a single estimator mid-training
  from its last epoch incl. optimizer/scheduler state). Bigger change to both
  `fit()` loops; not needed once per-estimator skip is in — the largest single
  estimator is the only thing at risk, and that's a bounded loss.
- **Comet model artifacts** as the durable store instead of / in addition to
  Drive (`experiment.log_model`). Drive is simpler and already mounted.
