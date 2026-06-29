# Issue 1: Neural-ODE Filter Estimator

Implement `NeuralODEEstimator` (replace the stub in
[`estimators/neural/neural_ode.py`](../estimators/neural/neural_ode.py)).

> Read [Issue 0 — shared contract](0_neural_filters_shared_contract.md) first.
> Everything below is a delta on top of it. The canonical template is
> [`kalmannet.py`](../estimators/neural/kalmannet.py).

## Idea

A continuous-time recursive filter. Between observations the latent state is
propagated by a **learned ODE** `dx/dt = g_θ(x, t)` (a small MLP), integrated
over each `dt = timestamps[t] - timestamps[t-1]`. At each observation we apply a
**learned, innovation-driven correction** in the spirit of a continuous-discrete
EKF, but with the drift and the correction both learned:

```
x_pred(t_k)  = x(t_{k-1}) + ∫_{t_{k-1}}^{t_k} g_θ(x, t) dt     # learned drift (ODE solve)
innov        = y_k - h(x_pred(t_k))                            # h = filter_model.h (true obs model)
x_post(t_k)  = x_pred(t_k) + c_φ([innov, x_pred, ctx])         # learned correction (small MLP)
```

This is the natural "continuous dynamics" counterpart to KalmanNet's
discrete predict/update. It tests whether a learned continuous flow beats the
benchmark's fixed discrete `f`.

## Process-model usage

- **`h` (observation model): USE the true `filter_model.h`** to form the
  innovation — same as EKF/KalmanNet. On GPU use `filter_model.torch.h`.
- **`f` (process model): the learned ODE `g_θ` REPLACES `f`.** This is the
  experiment's whole point — does a learned drift beat the level's `f`? Provide a
  constructor flag `use_model_drift: bool = False`; when `True`, parameterize the
  drift as `f_residual`: `dx/dt ≈ (f(x) - x)/dt_nominal + g_θ(x,t)` (learn the
  residual on top of the known dynamics). Default `False` = pure learned drift.

## Integrator (dependency policy)

- **Default: a dependency-free explicit RK4** written in plain PyTorch
  (fixed-step, `n_substeps` per inter-observation interval, configurable). This
  keeps it installable-free on this machine (see Issue 0 env constraints) and is
  fully batched/differentiable for GPU training via backprop-through-the-solver.
- **Optional:** if `torchdiffeq` is importable, allow `solver="dopri5"` using
  `odeint` / `odeint_adjoint` (adjoint = constant memory, good for long T). Import
  it lazily inside `fit`/`estimate`; on `ImportError` with `solver != "rk4"`,
  raise a clear message ("install torchdiffeq or use solver='rk4'"). Never
  silently fall back.

## fit() — GPU, batched, parallel over the batch

- Tensors `[N, T, *]` → `DataLoader(TensorDataset(obs, states), batch_size, shuffle=True)`.
- Forward: for each trajectory, seed `x_0` (zeros or `x0_mean`), then loop the T
  observation times — **the loop over T is intrinsic** (a filter is causal), but
  each step is a single batched `[B, nx]` ODE solve + batched correction, all on
  device with `filter_model.torch.h`. Parallelism is over the batch `B`, exactly
  like KalmanNet's free-running `_run_sequence_vectorized`
  ([`kalmannet.py:273-343`](../estimators/neural/kalmannet.py#L273-L343)).
- **No teacher-forcing curriculum** (Issue 0): the ODE drift is parallel over B
  and the integrator handles the within-step continuous propagation; there is no
  GRU recurrence to warm-start.
- Loss: MSE of `x_post` vs ground-truth `states` (reuse KalmanNet's loss/epoch/
  scheduler/early-stop/best-checkpoint machinery verbatim).
- Requires `filter_model.torch` for the GPU `h`; if `None`, raise the
  descriptive `ValueError` (Issue 0).

## estimate() — strictly sequential on CPU

Mirror [`_run_sequence_sequential_cpu`](../estimators/neural/kalmannet.py#L679-L738):
one trajectory, one timestep at a time, RK4 stepping with the **NumPy**
`filter_model.h` (and `filter_model.f` if `use_model_drift`), single state
vector. Report this as the embedded deployment latency. Returns `[N, T, nx]`.

## Constructor (additions to the Issue-0 minimum)

```python
ode_hidden: int = 64        # width of g_θ MLP
ode_layers: int = 2
n_substeps: int = 4         # RK4 fixed steps per inter-observation interval
solver: str = "rk4"         # "rk4" | "dopri5" (needs torchdiffeq)
correction_hidden: int = 64 # width of c_φ MLP
use_model_drift: bool = False
```

## Suggested per-level starting config

```python
"linear":    dict(ode_hidden=32,  n_substeps=2, num_epochs=10),
"pendulum":  dict(ode_hidden=64,  n_substeps=4, num_epochs=20),
"nonlinear": dict(ode_hidden=128, n_substeps=4, num_epochs=80, learning_rate=1e-3,
                  early_stopping_patience=10, weight_decay=1e-4, scheduler="cosine"),
"lorenz":    dict(ode_hidden=128, n_substeps=8, num_epochs=100, learning_rate=5e-4,
                  early_stopping_patience=10, weight_decay=1e-4, scheduler="cosine"),
```

(Lorenz is chaotic — more substeps; expect this to be the hardest level.)

## Acceptance criteria

- [ ] `NeuralODEEstimator` replaces the stub; full `BaseEstimator` interface.
- [ ] Dependency-free RK4 default works without `torchdiffeq`; `dopri5` path
      guarded by a lazy import + clear `ImportError`.
- [ ] GPU-batched `fit()`, strictly-sequential CPU `estimate()` returning `[N,T,nx]`.
- [ ] `use_model_drift` toggles residual-vs-pure learned drift; `h` always the
      true `filter_model.h`.
- [ ] No curriculum, no `dataset.states` in `estimate()`, no `tests/`, no `pip install`.
- [ ] Exported from `__init__.py`; added to the notebook estimator block.
