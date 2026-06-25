from __future__ import annotations  
  
from pathlib import Path  
from typing import List, Optional  
  
  
def plot_runtime_comparison(  
    estimator_names: List[str],  
    runtime_values: List[float],  
    title: str = "Runtime Comparison (ms/step)",  
    output_path: Optional[Path] = None,  
) -> None:  
    import matplotlib.pyplot as plt  
  
    fig, ax = plt.subplots(figsize=(8, 5))  
    bars = ax.bar(estimator_names, runtime_values)  
    ax.set_xlabel("Estimator")  
    ax.set_ylabel("Runtime (ms / step)")  
    ax.set_title(title)  
    ax.grid(True, axis="y")  
  
    for bar, val in zip(bars, runtime_values):  
        ax.text(  
            bar.get_x() + bar.get_width() / 2.0,  
            bar.get_height(),  
            f"{val:.3f}",  
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
