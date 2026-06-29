from __future__ import annotations

"""
Batched, GPU-friendly torch dynamics for each level, consumed only by
KalmanNet's vectorized training/validation (the classical filters use the
@njit NumbaDynamics instead).

Each `build_*_torch_dynamics(...)` returns a `TorchDynamics` whose f/h take a
batched state tensor [B, nx] (plus a scalar timestep t) and return [B, nx] /
[B, ny] using only torch tensor ops on the input's own device -- no per-row
Python loop, no NumPy round-trip. This is what lets KalmanNet's predict step
run fully on the GPU during fit().

torch is imported lazily inside the returned closures, so importing this module
(and the benchmark levels) never requires torch to be installed -- only
actually training a KalmanNet does. The math mirrors the pure-Python /
@njit f/h in each level one-for-one; keep them in sync.
"""

from .base import TorchDynamics


def build_linear_torch_dynamics(F_mat, H_mat) -> TorchDynamics:
    import numpy as np

    F_np = np.ascontiguousarray(F_mat, dtype=np.float64)
    H_np = np.ascontiguousarray(H_mat, dtype=np.float64)

    def f(x, t=0.0):
        import torch
        F = torch.as_tensor(F_np, dtype=x.dtype, device=x.device)
        return x @ F.T  # [B, nx]

    def h(x, t=0.0):
        import torch
        H = torch.as_tensor(H_np, dtype=x.dtype, device=x.device)
        return x @ H.T  # [B, ny]

    return TorchDynamics(f=f, h=h)


def build_pendulum_torch_dynamics(g: float, length: float, dt: float) -> TorchDynamics:
    gl = g / length

    def f(x, t=0.0):
        import torch
        theta = x[:, 0]
        omega = x[:, 1]
        alpha = -gl * torch.sin(theta)
        return torch.stack([theta + omega * dt, omega + alpha * dt], dim=1)

    def h(x, t=0.0):
        return x[:, 0:1]  # [B, 1]

    return TorchDynamics(f=f, h=h)


def build_nonlinear_torch_dynamics() -> TorchDynamics:
    def f(x, t=0.0):
        import torch
        xv = x[:, 0]
        out = 0.5 * xv + 25.0 * xv / (1.0 + xv ** 2) + 8.0 * torch.cos(torch.as_tensor(1.2 * t, dtype=x.dtype, device=x.device))
        return out.unsqueeze(1)  # [B, 1]

    def h(x, t=0.0):
        xv = x[:, 0]
        return (xv ** 2 / 20.0).unsqueeze(1)  # [B, 1]

    return TorchDynamics(f=f, h=h)


def build_lorenz_torch_dynamics(sigma: float, rho: float, beta: float, dt: float) -> TorchDynamics:
    state_bound = 1.0e3

    def _deriv(state):
        import torch
        xv = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        return torch.stack([
            sigma * (y - xv),
            xv * (rho - z) - y,
            xv * y - beta * z,
        ], dim=1)

    def f(x, t=0.0):
        import torch
        x = torch.clamp(x, -state_bound, state_bound)
        k1 = _deriv(x)
        k2 = _deriv(x + 0.5 * dt * k1)
        k3 = _deriv(x + 0.5 * dt * k2)
        k4 = _deriv(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def h(x, t=0.0):
        return x[:, 0:2]  # [B, 2]

    return TorchDynamics(f=f, h=h)
