from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def plot_loss_curves(
    history: Dict[str, list],
    title: str = "Training / Validation Loss",
    output_path: Optional[Path] = None,
    log_scale: bool = True,
) -> None:
    """
    Plot per-epoch train and validation loss (and the learning-rate schedule)
    from a KalmanNetEstimator.history_ dict.

    Parameters
    ----------
    history : dict with keys "epoch", "train_loss", "val_loss", and optionally
        "lr" -- as produced by KalmanNetEstimator.fit().
    log_scale : use a log y-axis for the loss panel (loss often spans decades).
    """
    import matplotlib.pyplot as plt

    epochs = history.get("epoch") or list(range(1, len(history["val_loss"]) + 1))
    train_loss = history.get("train_loss", [])
    val_loss = history["val_loss"]
    lrs = history.get("lr")

    ncols = 2 if lrs else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4.5), squeeze=False)

    ax = axes[0, 0]
    if train_loss:
        ax.plot(epochs, train_loss, label="train", marker="o", markersize=3, linewidth=1.5)
    ax.plot(epochs, val_loss, label="val", marker="o", markersize=3, linewidth=1.5)

    finite_val = [v for v in val_loss if np.isfinite(v)]
    if finite_val:
        best_idx = int(np.nanargmin(np.where(np.isfinite(val_loss), val_loss, np.inf)))
        ax.axvline(epochs[best_idx], color="gray", linestyle="--", linewidth=1,
                   label=f"best (epoch {epochs[best_idx]})")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.4)

    if lrs:
        ax_lr = axes[0, 1]
        ax_lr.plot(epochs, lrs, marker="o", markersize=3, color="tab:green", linewidth=1.5)
        ax_lr.set_yscale("log")
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("Learning rate")
        ax_lr.set_title("LR schedule")
        ax_lr.grid(True, which="both", alpha=0.4)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)


def plot_hyperparam_search(
    labels: list,
    val_losses: list,
    rmse_values: Optional[list] = None,
    title: str = "Hyperparameter search",
    output_path: Optional[Path] = None,
) -> None:
    """
    Bar chart comparing configurations from a hyperparameter sweep.

    Parameters
    ----------
    labels : config labels (one per bar)
    val_losses : best validation loss per config
    rmse_values : optional test RMSE per config (plotted on a twin axis)
    """
    import matplotlib.pyplot as plt

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 5))

    width = 0.4 if rmse_values is not None else 0.6
    bars = ax.bar(x - (width / 2 if rmse_values is not None else 0),
                  val_losses, width, label="best val loss", color="tab:blue")
    ax.set_ylabel("Best val loss", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, val_losses):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                f"{val:.3g}", ha="center", va="bottom", fontsize=7)

    if rmse_values is not None:
        ax2 = ax.twinx()
        ax2.bar(x + width / 2, rmse_values, width, label="test RMSE", color="tab:orange")
        ax2.set_ylabel("Test RMSE", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)
