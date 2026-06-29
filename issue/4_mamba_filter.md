# Issue 4: Mamba (Selective State-Space) Filter Estimator

Implement `MambaEstimator` in a new module
[`estimators/neural/mamba.py`](../estimators/neural/mamba.py).

> Read [Issue 0 — shared contract](0_neural_filters_shared_contract.md) first.
> The canonical template is [`kalmannet.py`](../estimators/neural/kalmannet.py).

## Idea

A **selective state-space model (Mamba)** filter. Mamba is the natural fit for
this benchmark: it *is* a learned linear state-space recursion with
input-dependent (selective) `A, B, C, Δ` parameters — a learned generalization of
the Kalman recursion. It trains in parallel via an associative **selective scan**
and runs as a cheap **constant-memory linear recurrence at inference**, which is
exactly the embedded-deployment story this benchmark measures.

```
tokens_t = embed([y_t, innovation_features_t])     # per-step input
h_t      = Ā_t ⊙ h_{t-1} + B̄_t x_t                  # selective SSM recurrence (per channel)
z_t      = C_t h_t  + D x_t
x̂_t      = head(z_t)        (optionally residual: x̂_t = x_pred_t + head(z_t))
```

This maps cleanly onto the filtering problem: `h_t` is the learned analog of the
filter's belief state, the selective `Δ_t`/`A_t` give it KalmanNet's
"trust adapts over time" behavior, and inference is a true `O(T)` recurrence with
fixed state — the cheapest learned filter in the suite at deployment.

## Process-model usage

Same options as the Transformer ([Issue 3](3_transformer_filter.md)): feed
innovation features `y_t - h(x_pred)` (recommended for nonlinear levels) or run
as a black-box on raw `y` (`use_innovation_features=False`). When innovation
features + GPU training are used, require `filter_model.torch.f`/`.h` (Issue-0
`ValueError` if `None`).

## Dependency policy (important — prefer dependency-free)

`mamba_ssm` (the official CUDA kernels, incl. Mamba-2 SSD) is the fast path but
**cannot be pip-installed on this machine** (Issue 0 env constraints; also needs
a specific CUDA toolchain). Therefore:

- **Default implementation: a dependency-free Mamba block in plain PyTorch** —
  the discretized selective SSM with a **parallel associative scan** for training
  (a `torch`-only `selective_scan` via cumulative-product segments or a
  log-space parallel scan) and a **plain sequential recurrence** for inference.
  This is the speedup the curriculum gives KalmanNet, but here it is *native*: the
  scan parallelizes training over T with no teacher forcing.
- **Optional fast path:** if `mamba_ssm` is importable, use its
  `selective_scan_fn` / `Mamba2` kernels for the GPU training scan. Lazy import
  inside `fit`; on `ImportError` fall back to the plain-PyTorch scan **with a
  printed notice** (this is the one allowed "fallback" because both paths are
  numerically the same recurrence — document it; do not silently differ in math).

## Why NO teacher-forcing curriculum (key point)

The selective scan is an **associative operation → parallel over T** at training
time (that is Mamba's entire premise). So `fit()` runs the whole `[B, T, *]`
sequence through the parallel scan in one pass — no sequential T-loop over the
network, no two-phase curriculum. The teacher-forcing trick KalmanNet needs is
unnecessary here; the parallelism is built into the SSM. (Same caveat as the
Transformer: if innovation features depend on the model's own `x̂_{t-1}`, build
them from ground-truth-prev during training, model-prev at inference; with
`use_innovation_features=False` this is moot.)

## fit() — GPU, batched, parallel scan

- Standard DataLoader / epoch / scheduler / early-stop / best-checkpoint / NaN-skip
  scaffolding from KalmanNet.
- Forward = stacked Mamba blocks via the parallel selective scan → `x̂ [B,T,nx]`;
  MSE vs `states`.

## estimate() — strictly sequential on CPU

Run the SSM as its **recurrent form**: one trajectory, one timestep at a time,
updating the fixed-size hidden state `h_t` on CPU with NumPy `f`/`h` for the
innovation features. This is where Mamba's `O(T)` constant-state inference cost is
proven — likely the cheapest learned filter at deployment. Mirror
[`_run_sequence_sequential_cpu`](../estimators/neural/kalmannet.py#L679-L738).
Returns `[N, T, nx]`.

## Constructor (additions to the Issue-0 minimum)

```python
d_model: int = 64
d_state: int = 16          # SSM state expansion N
d_conv: int = 4            # depthwise causal conv width
expand: int = 2            # block expansion factor
n_layers: int = 2
use_innovation_features: bool = True
residual_head: bool = True
use_mamba_ssm_kernels: bool = True   # try mamba_ssm if importable, else plain-torch scan
```

## Suggested per-level starting config

```python
"linear":    dict(d_model=32, d_state=8,  n_layers=1, num_epochs=10),
"pendulum":  dict(d_model=64, d_state=16, n_layers=2, num_epochs=20),
"nonlinear": dict(d_model=128, d_state=16, n_layers=3, num_epochs=80, learning_rate=1e-3,
                  early_stopping_patience=10, weight_decay=1e-4, scheduler="cosine"),
"lorenz":    dict(d_model=128, d_state=32, n_layers=4, num_epochs=100, learning_rate=5e-4,
                  early_stopping_patience=10, weight_decay=1e-4, scheduler="cosine"),
```

## Acceptance criteria

- [ ] `MambaEstimator` in `mamba.py`; full `BaseEstimator` interface.
- [ ] Dependency-free plain-PyTorch selective SSM works WITHOUT `mamba_ssm`
      (parallel scan for training, sequential recurrence for inference).
- [ ] Optional `mamba_ssm`/Mamba-2 kernel path behind a lazy import; the
      fall-back is documented and numerically identical.
- [ ] `fit()` parallel scan over T — **no curriculum**.
- [ ] `estimate()` is the `O(T)` constant-state recurrence, strictly sequential on
      CPU → `[N,T,nx]`.
- [ ] No `dataset.states` in `estimate()`, no `tests/`, no `pip install`.
- [ ] Exported from `__init__.py`; added to the notebook estimator block.
```
