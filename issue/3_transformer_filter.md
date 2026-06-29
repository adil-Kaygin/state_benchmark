# Issue 3: Transformer-Based Filter Estimator

Implement `TransformerEstimator` (replace the stub in
[`estimators/neural/transformer.py`](../estimators/neural/transformer.py)).

> Read [Issue 0 — shared contract](0_neural_filters_shared_contract.md) first.
> The canonical template is [`kalmannet.py`](../estimators/neural/kalmannet.py).

## Idea

A **causal (decoder-style) Transformer** that maps the observation sequence
`y_{1:t}` to the state estimate `x̂_t`. Self-attention lets each estimate attend
over the whole past window of observations — a different inductive bias from a
recurrent filter (explicit long-range access vs. a compressed hidden state).

```
tokens_t  = embed([y_t, h(f(x̂_{t-1})), Δt])           # per-step input features
z_{1:t}   = CausalTransformer(tokens_{1:t})            # masked self-attention, causal
x̂_t       = head(z_t)            (optionally  x̂_t = x_pred_t + head(z_t)  residual form)
```

Use **causal masking** so `x̂_t` depends only on `y_{≤t}` — a filter, not a
smoother. Add positional encoding (sinusoidal or learned) over the time axis.
Optionally feed the model-based `x_pred = f(x̂_{t-1})` / innovation as input
features (residual/innovation form), mirroring how KalmanNet conditions on the
innovation; this is recommended for the nonlinear levels.

## Process-model usage

- **`h` (and optionally `f`): used to build innovation input features**
  (`y_t - h(x_pred)`), the same conditioning KalmanNet uses. The Transformer can
  also run as a pure black-box on raw `y_{1:t}` (set `use_innovation_features=False`)
  — keep both as a constructor flag for ablation.
- If innovation features are on and the GPU path is used, you need
  `filter_model.torch.f`/`.h` (raise the Issue-0 `ValueError` if `None`). Pure
  black-box mode needs no `filter_model.torch`.

## Why NO curriculum is needed (key point)

A causal Transformer is **fully parallel over T at training time**: one masked
forward pass computes `x̂_t` for all `t` simultaneously (the causal mask enforces
the no-future-leak constraint). This is exactly the parallelism KalmanNet had to
*fake* with teacher forcing. So:

- **Training = one parallel `[B, T, *]` forward pass.** No sequential T-loop, no
  two-phase curriculum, no `_run_sequence_teacher_forced` analog.
- **BUT** if you use innovation features built from `x̂_{t-1}` (the model's own
  previous estimate), that input *is* sequential. Resolve this the standard way:
  build the innovation features from the **ground-truth previous state during
  training** (this is the legitimate, parallel teacher-forced *input* construction
  — analogous to [`_run_sequence_teacher_forced`](../estimators/neural/kalmannet.py#L345-L432),
  but here it is the natural training mode, not a curriculum hack), and from the
  model's own previous estimate at inference. If `use_innovation_features=False`,
  the input is just raw `y` and this issue disappears.

## fit() — GPU, batched, single parallel pass

- DataLoader of `[N, T, *]` tensors as usual.
- One causal-masked forward over the full sequence → `x̂ [B, T, nx]`; MSE vs
  `states`. Reuse KalmanNet's epoch/scheduler/early-stop/best-checkpoint/NaN-skip
  scaffolding.
- Implement the Transformer with `torch.nn.TransformerEncoder` +
  `nn.Transformer.generate_square_subsequent_mask` (causal). No third-party
  dependency — plain PyTorch.

## estimate() — strictly sequential on CPU

**This is the subtle part.** At inference the Transformer must still run causally,
one step at a time, on CPU per the Issue-0 contract — you may NOT do one big
parallel GPU pass over the test set, because that would not measure deployment
latency. Implement it as: for each trajectory, for each `t`, run the model on the
prefix `y_{1:t}` (with innovation features from the model's own `x̂_{t-1}` via
NumPy `f`/`h`) and take the last position's output as `x̂_t`. Recomputing the
prefix each step is the honest deployment cost of an attention model with no KV
cache; if you add a CPU KV-cache to amortize it, that is fine and realistic —
document which you did. Returns `[N, T, nx]`. Mirror the single-trajectory,
single-step loop shape of
[`_run_sequence_sequential_cpu`](../estimators/neural/kalmannet.py#L679-L738).

## Constructor (additions to the Issue-0 minimum)

```python
d_model: int = 64
n_heads: int = 4
n_layers: int = 3
dim_feedforward: int = 256
dropout: float = 0.1
max_len: int = 1024              # positional-encoding capacity ≥ trajectory T
use_innovation_features: bool = True
residual_head: bool = True       # x̂_t = x_pred_t + head(z_t)  vs  x̂_t = head(z_t)
```

## Suggested per-level starting config

```python
"linear":    dict(d_model=32,  n_layers=2, n_heads=2, num_epochs=10),
"pendulum":  dict(d_model=64,  n_layers=2, n_heads=4, num_epochs=20),
"nonlinear": dict(d_model=128, n_layers=4, n_heads=8, num_epochs=80, learning_rate=1e-3,
                  early_stopping_patience=10, weight_decay=1e-4, scheduler="cosine"),
"lorenz":    dict(d_model=128, n_layers=4, n_heads=8, num_epochs=100, learning_rate=5e-4,
                  early_stopping_patience=10, weight_decay=1e-4, scheduler="cosine"),
```

## Acceptance criteria

- [ ] `TransformerEstimator` replaces the stub; full `BaseEstimator` interface.
- [ ] Causal-masked attention (no future leak); positional encoding; plain
      PyTorch, no third-party dep.
- [ ] `fit()` is a single parallel `[B,T,*]` pass — **no curriculum**.
- [ ] `estimate()` runs causally one step at a time on CPU → `[N,T,nx]`; the
      prefix-recompute / KV-cache choice is documented.
- [ ] `use_innovation_features` toggle works in both modes; innovation features
      built from GT-prev in training, model-prev at inference.
- [ ] No `dataset.states` in `estimate()`, no `tests/`, no `pip install`.
- [ ] Exported from `__init__.py`; added to the notebook estimator block.
