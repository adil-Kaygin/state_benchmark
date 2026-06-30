from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def plot_runtime_comparison(
    estimator_names: List[str],
    runtime_values: List[float],
    title: str = "Runtime Comparison (ms/step)",
    output_path: Optional[Path] = None,
    runtime_errors: Optional[List[float]] = None,
) -> None:
    """Bar chart of per-step runtime, one bar per estimator.

    runtime_values is the per-estimator latency (a single mean ms/step value).
    runtime_errors, if supplied, gives a matching spread (e.g. std / 95% CI
    half-width) and draws symmetric error bars.
    """
    import matplotlib.pyplot as plt

    if runtime_errors is not None and len(runtime_errors) != len(runtime_values):
        raise ValueError(
            f"runtime_errors has length {len(runtime_errors)} but there are "
            f"{len(runtime_values)} estimators; they must match."
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(estimator_names, runtime_values, yerr=runtime_errors,
                  capsize=(3 if runtime_errors is not None else 0))
    ax.set_xlabel("Estimator")
    ax.set_ylabel("Runtime (ms / step)")
    ax.set_title(title)
    ax.grid(True, axis="y")

    for i, (bar, val) in enumerate(zip(bars, runtime_values)):
        label = f"{val:.3f}" if runtime_errors is None else f"{val:.3f}\n±{runtime_errors[i]:.2g}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            label,
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
