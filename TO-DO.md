## TO-DO

### 1. add gradient clipping better training setup
- [x] Already present (`clip_grad_norm_` in `KalmanNetEstimator.fit`); clip
  threshold is now a `grad_clip_norm` constructor param (default 0.5).
  See estimators/README.md.

### 2. implement kalmannet with uncertanity head (if possible)
- [x] Already present: `KalmanNetUncertaintyEstimator` (log-variance head +
  Gaussian NLL loss), registered in `EXPERIMENTAL_ESTIMATORS`.

### 3. add optional visualization techniques such as state-wise ground-truth vs predictions; step-wise RMSE or similar ; ...
- [x] Added `metrics.rmse.compute_rmse_per_timestep`,
  `visualization.plot_states_all_dims`, `visualization.plot_rmse_per_timestep`.

### 4. sannity check all codebase especially benchmark_levels and estimators
- [x] Audited; no new bugs. Only intentional `NotImplementedError` stubs
  (neural_ode/transformer, classical/neural `load()`). See estimators/README.md
  and benchmark_levels/README.md "Sanity check" sections.

### 5. add filterpy models (kf ekf ukf)
- [x] Added `FilterpyKFEstimator`/`FilterpyEKFEstimator`/`FilterpyUKFEstimator`
  (`estimators/classical/filterpy_filters.py`), optional import, registered in
  `REFERENCE_ESTIMATORS`. `filterpy>=1.4` added to setup.py (install on Colab).

### 6. how KF implemented in non-linear system dynamics
- [x] Answered: linearizes once at the origin, treats F/H as constant;
  `use_numba=True` raises on nonlinear models, pure-NumPy silently
  linearizes. See estimators/README.md.

### 7. add comet logging
- [x] `CometExperimentLogger` made push-only + batched + deduped (no metric
  recomputation, no train-param logging). Optional `comet_logger` param on
  `ExperimentRunner`, `flush_comet_logger()` to push. See utils/logging.py.

### 8. is RMSE evaluation fair?
- [x] Answered: fair within a benchmark, not poolable across benchmarks
  (different units). See metrics/README.md.

### 9. is latency evaluation fair?
- [x] Answered: yes by design — measures CPU inference latency only
  (excludes fit()/JIT warm-up), matching missile-science distinction between
  in-flight latency and total compute cost. No code change (per decision).
  See metrics/README.md.
