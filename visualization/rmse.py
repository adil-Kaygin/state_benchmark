from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
  
  
def plot_rmse_comparison(  
    estimator_names: List[str],  
    rmse_values: List[float],  
    title: str = "RMSE Comparison",  
    output_path: Optional[Path] = None,  
) -> None:  
    import matplotlib.pyplot as plt  
  
    fig, ax = plt.subplots(figsize=(8, 5))  
    bars = ax.bar(estimator_names, rmse_values)  
    ax.set_xlabel("Estimator")  
    ax.set_ylabel("RMSE")  
    ax.set_title(title)  
    ax.grid(True, axis="y")  
  
    for bar, val in zip(bars, rmse_values):  
        ax.text(  
            bar.get_x() + bar.get_width() / 2.0,  
            bar.get_height(),  
            f"{val:.4f}",  
            ha="center",  
            va="bottom",  
            fontsize=9,  
        )  
  
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
