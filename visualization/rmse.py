from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
  
  
def plot_rmse_comparison_per_dim(
    rmse_per_dim_by_estimator: Dict[str, Dict[str, float]],
    state_names: Sequence[str],
    title: str = "Per-variable RMSE Comparison",
    output_path: Optional[Path] = None,
) -> None:
    """Grouped bar chart of RMSE per named state variable, one bar per estimator.

    Replaces the old single-scalar `plot_rmse_comparison`: pooling state
    dimensions of different physical units/scales into one bar was scientifically
    unsound (it is dominated by the largest-magnitude dimension), so RMSE is now
    always shown per physical variable (e.g. x/y/z for Lorenz, theta/omega for
    the pendulum).

    Parameters
    ----------
    rmse_per_dim_by_estimator : dict mapping estimator_name -> {state_var: rmse}
        (each inner dict is the output of metrics.rmse.compute_rmse_per_dim).
    state_names : ordered names of the state variables (the x-axis groups).

    Raises
    ------
    ValueError if any estimator is missing an RMSE for a declared state variable
        (fail fast: a silently-dropped dimension would mislabel the chart).
    """
    import matplotlib.pyplot as plt

    estimator_names = list(rmse_per_dim_by_estimator.keys())
    state_names = list(state_names)

    for est_name, per_dim in rmse_per_dim_by_estimator.items():
        missing = [s for s in state_names if s not in per_dim]
        if missing:
            raise ValueError(
                f"estimator '{est_name}' is missing RMSE for state variable(s) "
                f"{missing}; expected one value per name in {state_names}."
            )

    n_groups = len(state_names)
    n_est = len(estimator_names)
    x = np.arange(n_groups)
    width = 0.8 / max(n_est, 1)

    fig, ax = plt.subplots(figsize=(max(8, 1.5 * n_groups * n_est), 5))
    for j, est_name in enumerate(estimator_names):
        offsets = x + (j - (n_est - 1) / 2.0) * width
        values = [rmse_per_dim_by_estimator[est_name][s] for s in state_names]
        bars = ax.bar(offsets, values, width, label=est_name)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.3g}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xlabel("State variable")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(state_names)
    ax.legend()
    ax.grid(True, axis="y")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)


def plot_rmse_per_timestep(
    timestamps: np.ndarray,
    rmse_per_step: Dict[str, np.ndarray],
    title: str = "Step-wise RMSE",
    output_path: Optional[Path] = None,
) -> None:
    """
    Step-wise RMSE: one line per estimator, showing how error evolves over
    a trajectory (e.g. filter convergence/divergence).

    Parameters
    ----------
    timestamps : np.ndarray, shape [T]
    rmse_per_step : dict mapping estimator_name -> np.ndarray, shape [T]
        (e.g. from metrics.rmse.compute_rmse_per_timestep)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for estimator_name, rmse_values in rmse_per_step.items():
        ax.plot(timestamps, rmse_values, label=estimator_name, linewidth=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)
