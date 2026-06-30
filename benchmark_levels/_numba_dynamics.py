from __future__ import annotations

"""
@njit factory functions producing the CPU-optimized dynamics for each level.

Each `build_*_numba_dynamics(...)` returns a `NumbaDynamics` whose f/h/F_jac/H_jac
are `@njit` closures over the level's scalar parameters (dt, g, length, sigma,
...). Numba bakes those captured constants into the compiled code, so the
resulting functions match the generic kernel signature used by KF/EKF/UKF
(f(x, t), h(x, t), F_jac(x), H_jac(x)) without per-call Python overhead.

These are kept in their own module (file-backed, so `@njit(cache=True)` can
persist compiled artifacts) and deliberately mirror the pure-Python f/h/F/H in
linear.py / pendulum.py / nonlinear.py / lorenz.py one-for-one. If you change
the math in a level's `get_filter_model`, change it here too.

numba is a HARD dependency: the classical filters run exclusively on these
@njit dynamics (there is no pure-NumPy filter path). Per the "fail fast and
loud" rule, a missing numba raises ImportError here rather than silently
producing un-jitted closures.
"""

import numpy as np

from .base import NumbaDynamics

from numba import njit


def build_linear_numba_dynamics(F_mat: np.ndarray, H_mat: np.ndarray) -> NumbaDynamics:
    F_c = np.ascontiguousarray(F_mat, dtype=np.float64)
    H_c = np.ascontiguousarray(H_mat, dtype=np.float64)

    @njit(cache=True, fastmath=True)
    def f(x, t):
        return F_c @ x

    @njit(cache=True, fastmath=True)
    def h(x, t):
        return H_c @ x

    @njit(cache=True, fastmath=True)
    def F_jac(x):
        return F_c

    @njit(cache=True, fastmath=True)
    def H_jac(x):
        return H_c

    return NumbaDynamics(f=f, h=h, F_jac=F_jac, H_jac=H_jac)


def build_pendulum_numba_dynamics(g: float, length: float, dt: float) -> NumbaDynamics:
    gl = g / length

    @njit(cache=True, fastmath=True)
    def f(x, t):
        theta = x[0]
        omega = x[1]
        alpha = -gl * np.sin(theta)
        out = np.empty(2)
        out[0] = theta + omega * dt
        out[1] = omega + alpha * dt
        return out

    @njit(cache=True, fastmath=True)
    def h(x, t):
        out = np.empty(1)
        out[0] = x[0]
        return out

    @njit(cache=True, fastmath=True)
    def F_jac(x):
        theta = x[0]
        out = np.empty((2, 2))
        out[0, 0] = 1.0
        out[0, 1] = dt
        out[1, 0] = -gl * np.cos(theta) * dt
        out[1, 1] = 1.0
        return out

    @njit(cache=True, fastmath=True)
    def H_jac(x):
        out = np.zeros((1, 2))
        out[0, 0] = 1.0
        return out

    return NumbaDynamics(f=f, h=h, F_jac=F_jac, H_jac=H_jac)


def build_nonlinear_numba_dynamics() -> NumbaDynamics:
    @njit(cache=True, fastmath=True)
    def f(x, t):
        xv = x[0]
        out = np.empty(1)
        out[0] = 0.5 * xv + 25.0 * xv / (1.0 + xv ** 2) + 8.0 * np.cos(1.2 * t)
        return out

    @njit(cache=True, fastmath=True)
    def h(x, t):
        out = np.empty(1)
        out[0] = x[0] ** 2 / 20.0
        return out

    @njit(cache=True, fastmath=True)
    def F_jac(x):
        xv = x[0]
        out = np.empty((1, 1))
        out[0, 0] = 0.5 + 25.0 * (1.0 - xv ** 2) / (1.0 + xv ** 2) ** 2
        return out

    @njit(cache=True, fastmath=True)
    def H_jac(x):
        out = np.empty((1, 1))
        out[0, 0] = x[0] / 10.0
        return out

    return NumbaDynamics(f=f, h=h, F_jac=F_jac, H_jac=H_jac)


# Mirror lorenz.py: clip a diverging estimate so the chaotic RK4 step cannot
# overflow to inf/NaN. The true attractor is far inside this bound.
_LORENZ_STATE_BOUND = 1.0e3


def _build_lorenz_common(sigma: float, rho: float, beta: float, dt: float):
    """njit f/h plus the continuous-field deriv/Jacobian helpers shared by both
    the standard (RK4-Jacobian) and FEA Lorenz dynamics builders."""
    state_bound = _LORENZ_STATE_BOUND

    @njit(cache=True, fastmath=True)
    def _deriv(state):
        xv = state[0]
        y = state[1]
        z = state[2]
        out = np.empty(3)
        out[0] = sigma * (y - xv)
        out[1] = xv * (rho - z) - y
        out[2] = xv * y - beta * z
        return out

    @njit(cache=True, fastmath=True)
    def _field_jac(state):
        """Jacobian of the continuous Lorenz vector field at `state`."""
        xv = state[0]
        y = state[1]
        z = state[2]
        J = np.empty((3, 3))
        J[0, 0] = -sigma
        J[0, 1] = sigma
        J[0, 2] = 0.0
        J[1, 0] = rho - z
        J[1, 1] = -1.0
        J[1, 2] = -xv
        J[2, 0] = y
        J[2, 1] = xv
        J[2, 2] = -beta
        return J

    @njit(cache=True, fastmath=True)
    def f(x, t):
        x = np.minimum(np.maximum(x, -state_bound), state_bound)
        k1 = _deriv(x)
        k2 = _deriv(x + 0.5 * dt * k1)
        k3 = _deriv(x + 0.5 * dt * k2)
        k4 = _deriv(x + dt * k3)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    @njit(cache=True, fastmath=True)
    def h(x, t):
        out = np.empty(2)
        out[0] = x[0]
        out[1] = x[1]
        return out

    @njit(cache=True, fastmath=True)
    def H_jac(x):
        out = np.zeros((2, 3))
        out[0, 0] = 1.0
        out[1, 1] = 1.0
        return out

    return _deriv, _field_jac, f, h, H_jac


def build_lorenz_numba_dynamics(sigma: float, rho: float, beta: float, dt: float) -> NumbaDynamics:
    """Standard Lorenz dynamics: F_jac is the exact Jacobian of the RK4 step
    (chain rule through the four stages), matching lorenz.LorenzBenchmark."""
    state_bound = _LORENZ_STATE_BOUND
    _deriv, _field_jac, f, h, H_jac = _build_lorenz_common(sigma, rho, beta, dt)

    @njit(cache=True, fastmath=True)
    def F_jac(x):
        x = np.minimum(np.maximum(x, -state_bound), state_bound)
        I = np.eye(3)

        k1 = _deriv(x)
        a2 = x + 0.5 * dt * k1
        k2 = _deriv(a2)
        a3 = x + 0.5 * dt * k2
        k3 = _deriv(a3)
        a4 = x + dt * k3

        dk1 = _field_jac(x)
        dk2 = _field_jac(a2) @ (I + 0.5 * dt * dk1)
        dk3 = _field_jac(a3) @ (I + 0.5 * dt * dk2)
        dk4 = _field_jac(a4) @ (I + dt * dk3)

        return I + (dt / 6.0) * (dk1 + 2.0 * dk2 + 2.0 * dk3 + dk4)

    return NumbaDynamics(f=f, h=h, F_jac=F_jac, H_jac=H_jac)


def build_lorenz_fea_numba_dynamics(sigma: float, rho: float, beta: float, dt: float) -> NumbaDynamics:
    """FEA baseline Lorenz dynamics: F_jac is the first-order forward-Euler
    linearization I + dt*J of the flow, matching lorenz.LorenzFEABenchmark."""
    state_bound = _LORENZ_STATE_BOUND
    _deriv, _field_jac, f, h, H_jac = _build_lorenz_common(sigma, rho, beta, dt)

    @njit(cache=True, fastmath=True)
    def F_jac(x):
        x = np.minimum(np.maximum(x, -state_bound), state_bound)
        return np.eye(3) + dt * _field_jac(x)

    return NumbaDynamics(f=f, h=h, F_jac=F_jac, H_jac=H_jac)
