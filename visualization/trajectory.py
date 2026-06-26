from __future__ import annotations  
  
from pathlib import Path  
from typing import Optional  
  
import numpy as np  
  
  
def plot_trajectory(  
    states: np.ndarray,  
    estimates: np.ndarray,  
    timestamps: np.ndarray,  
    trajectory_index: int = 0,  
    state_index: int = 0,  
    title: str = "State Trajectory",  
    output_path: Optional[Path] = None,  
) -> None:  
    import matplotlib.pyplot as plt  
  
    fig, ax = plt.subplots(figsize=(10, 4))  
    ax.plot(timestamps, states[trajectory_index, :, state_index],  
            label="True State", linewidth=1.5)  
    ax.plot(timestamps, estimates[trajectory_index, :, state_index],  
            label="Estimate", linestyle="--", linewidth=1.5)  
    ax.set_xlabel("Time")  
    ax.set_ylabel(f"State[{state_index}]")  
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


def plot_states_all_dims(
    states: np.ndarray,
    estimates: np.ndarray,
    timestamps: np.ndarray,
    trajectory_index: int = 0,
    title: str = "State Trajectory (all dimensions)",
    output_path: Optional[Path] = None,
) -> None:
    """
    State-wise ground-truth vs. prediction: one subplot per state dimension
    for a single trajectory, complementing plot_trajectory (single dim).
    """
    import matplotlib.pyplot as plt

    nx = states.shape[2]
    fig, axes = plt.subplots(nx, 1, figsize=(10, 3 * nx), sharex=True, squeeze=False)

    for i in range(nx):
        ax = axes[i, 0]
        ax.plot(timestamps, states[trajectory_index, :, i],
                label="True State", linewidth=1.5)
        ax.plot(timestamps, estimates[trajectory_index, :, i],
                label="Estimate", linestyle="--", linewidth=1.5)
        ax.set_ylabel(f"State[{i}]")
        ax.grid(True)
        if i == 0:
            ax.legend()

    axes[-1, 0].set_xlabel("Time")
    fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)
