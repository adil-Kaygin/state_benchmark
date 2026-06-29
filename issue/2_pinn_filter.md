# Issue 2: Physics-Informed Neural Network (PINN) Filter Estimator

Implement `PINNFilterEstimator` in a new module
[`estimators/neural/pinn.py`](../estimators/neural/pinn.py).

> Read [Issue 0 — shared contract](0_neural_filters_shared_contract.md) first.
> The canonical template is [`kalmannet.py`](../estimators/neural/kalmannet.py).

## Idea

A filter trained with a **physics-informed loss** that penalizes violation of
the *known* dynamics `f`/`h`, not just state-MSE. The network maps the
observation history to a state estimate; the training objective adds residual
terms that force the estimate to obey the benchmark's physics. This directly
tests whether injecting the known model into the loss (rather than the
architecture) improves estimation.

Two residuals, both built from the **true** `filter_model` (no model mismatch,
same `f`/`h` EKF gets):

```
data residual    r_data = x̂_t - x_t                       (supervised state error)
dynamics residual r_dyn  = x̂_{t}   - f(x̂_{t-1}, t-1)       (process-model consistency)
measurement res. r_meas  = y_t      - h(x̂_t, t)            (observation consistency, no GT needed)

loss = ‖r_data‖² + λ_dyn ‖r_dyn‖² + λ_meas ‖r_meas‖²
```

- `r_dyn` enforces that successive estimates respect the known transition `f`.
- `r_meas` is **self-supervised** (uses only `y`, not `x`) — keeps corrections
  consistent in measurement space; this is the physics-informed signal that needs
  no ground truth (see the "measurement-consistency aux term" in
  [`brain_storm/29_06.md`](../brain_storm/29_06.md)).

## Process-model usage

- **Both `f` and `h` are used in the LOSS**, not necessarily inside the forward
  pass. This is what makes it a PINN rather than a black-box regressor. On the
  GPU training path use `filter_model.torch.f` / `.torch.h` so the residuals are
  batched and differentiable; require `filter_model.torch` (raise the Issue-0
  `ValueError` if `None`).
- Expose `lambda_dyn` and `lambda_meas` as constructor weights. Setting both to
  `0.0` reduces it to a plain supervised filter (useful ablation baseline).

## Backbone architecture

Keep the estimation network simple and **causal** — it must run sequentially at
inference (Issue 0 contract). Recommended: a small **GRU** over the innovation
features `[y_t, h(x_pred), x_pred]`, emitting `x̂_t`, structured exactly like
KalmanNet's predict/update but with the physics encoded in the *loss* instead of
the gain form. Reusing KalmanNet's `_KalmanGainGRU`-style recurrence means the
GPU/CPU split and the sequential-inference path are already solved — the only new
thing is the loss. (A non-recurrent MLP-over-window backbone is acceptable as an
alternative, but it must still run causally per Issue 0.)

## fit() — GPU, batched

- Same DataLoader / epoch / scheduler / early-stop / best-checkpoint scaffolding
  as KalmanNet.
- Forward produces `x̂ [B, T, nx]`. Compute the three residuals batched on device
  (`f`/`h` from `filter_model.torch`), combine with `lambda_*`. Skip non-finite
  losses (Issue 0).
- **No teacher-forcing curriculum** — the physics residuals are computed on the
  produced `[B, T, nx]` sequence in parallel over the batch.

## estimate() — strictly sequential on CPU

Identical structure to
[`_run_sequence_sequential_cpu`](../estimators/neural/kalmannet.py#L679-L738):
the physics loss is training-only; at inference the network just runs causally on
CPU with NumPy `f`/`h`, one trajectory / one timestep at a time. Returns `[N,T,nx]`.

## Constructor (additions to the Issue-0 minimum)

```python
hidden_size: int = 64
lambda_dyn: float = 1.0      # weight on f-consistency residual
lambda_meas: float = 0.1     # weight on h-consistency (self-supervised) residual
```

## Suggested per-level starting config

```python
"linear":    dict(hidden_size=32, lambda_dyn=1.0, lambda_meas=0.1, num_epochs=10),
"pendulum":  dict(hidden_size=32, lambda_dyn=1.0, lambda_meas=0.2, num_epochs=20),
"nonlinear": dict(hidden_size=128, lambda_dyn=0.5, lambda_meas=0.2, num_epochs=80,
                  learning_rate=1e-3, early_stopping_patience=10, weight_decay=1e-4, scheduler="cosine"),
"lorenz":    dict(hidden_size=64, lambda_dyn=0.5, lambda_meas=0.1, num_epochs=100,
                  learning_rate=5e-4, early_stopping_patience=10, weight_decay=1e-4, scheduler="cosine"),
```

## Acceptance criteria

- [ ] `PINNFilterEstimator` in `pinn.py`; full `BaseEstimator` interface.
- [ ] Physics-informed loss with `r_data` + `λ_dyn·r_dyn` + `λ_meas·r_meas`,
      residuals built from the true `filter_model.f`/`h` (GPU via `.torch`).
- [ ] `lambda_dyn=lambda_meas=0` recovers the supervised baseline (documented).
- [ ] GPU-batched `fit()`, strictly-sequential CPU `estimate()` → `[N,T,nx]`.
- [ ] No curriculum, no `dataset.states` in `estimate()`, no `tests/`, no `pip install`.
- [ ] Exported from `__init__.py`; added to the notebook estimator block.
