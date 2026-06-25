from __future__ import annotations

from .runtime import runtime_per_step_ms


def latency_ms_per_step(total_seconds: float, num_trajectories: int, trajectory_length: int) -> float:
    """
    Normalized inference latency in milliseconds per timestep, averaged
    across all trajectories in a batch ([N, T, nx] -> N*T steps).
    """
    return runtime_per_step_ms(total_seconds, num_trajectories * trajectory_length)
