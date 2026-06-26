# visualization

Pure rendering — no computation. Every function takes already-computed arrays
(`states`/`estimates` from `estimators/`, or scalar metrics from `metrics/`)
and either shows or saves a matplotlib figure. None of these functions touch
`Q`/`R`/noise; they plot the outputs of the model code documented in
`benchmark_levels/README.md` and `estimators/README.md`.

```
trajectory.py  : plot_trajectory          single state dim, x_t (truth) vs x̂_t (estimate)
                 plot_states_all_dims      same, but one subplot per state dimension
rmse.py        : plot_rmse_comparison      bar chart, one compute_rmse() scalar per estimator
                 plot_rmse_per_timestep    line plot, compute_rmse_per_timestep() per estimator
runtime.py     : plot_runtime_comparison   bar chart, one runtime_per_step_ms scalar per estimator
```

All five are exported from `visualization/__init__.py`'s `__all__`.

## trajectory.py — `plot_trajectory` / `plot_states_all_dims`

`plot_trajectory` plots a single state dimension of a single trajectory over
time: `states[trajectory_index, :, state_index]` (ground truth, never observed
by the estimator) overlaid with `estimates[trajectory_index, :, state_index]`.
It does **not** plot `observations` — to sanity-check the raw sensor signal,
call it with `estimates` set to the raw observations array (only valid when
`observation_dimension == state_dimension`, i.e. not for `pendulum` or `lorenz`
where `ny < nx`).

`plot_states_all_dims` plots every state dimension of one trajectory in its own
subplot (`states` vs `estimates`) — use it to see all dimensions at once (e.g.
Lorenz's unobserved `z`, or pendulum's `θ`/`ω` together).

`timestamps` must come from the same dataset split as `states`/`estimates`
(`HDF5Writer` stores one `timestamps` array per split, shared across all
trajectories in that split — see `datasets/schema.py`).

## rmse.py / runtime.py — bar charts and per-timestep lines

`plot_rmse_comparison` and `plot_runtime_comparison` are thin bar-chart wrappers
over `(estimator_names, values)` pairs — no new math. `rmse_values` /
`runtime_values` are expected to already be the output of
`metrics.rmse.compute_rmse` / `metrics.latency.latency_ms_per_step`. Mismatched
units (e.g. passing raw seconds into the "ms/step" plot) are not caught here —
the caller must use the right metric function first. Call the RMSE comparison
once per benchmark (one figure per benchmark), not on a cross-benchmark average
of raw `compute_rmse` values (see `metrics/Critique.md` for why that average is
not physically meaningful).

`plot_rmse_per_timestep` takes a `{estimator_name: compute_rmse_per_timestep(...)}`
dict and overlays one line per estimator against `timestamps` — use it to see
*when* in a trajectory a filter's error grows (e.g. EKF/UKF divergence on
`lorenz`'s chaotic dynamics), which the single scalar from `plot_rmse_comparison`
can't show.

## Extending with a new plot

- Keep the "pure rendering, no metric computation" boundary: if a new plot
  needs a derived quantity (e.g. per-timestep RMSE instead of the scalar mean),
  add that reduction to `metrics/`, not here.
- Match the existing signature shape: `(...arrays/values, title=..., output_path:
  Optional[Path] = None)`, saving to `output_path` if given else `plt.show()`,
  always `plt.close(fig)` after.
- Register the new function in `visualization/__init__.py`'s `__all__`.
