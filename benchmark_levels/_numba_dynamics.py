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
the math in a level's `get_filter_model`, change it here too -- the regression
tests (tests/test_classical_filters.py) assert the two paths agree.
"""

import numpy as np

from .base import NumbaDynamics

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is an optional accelerator
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _decorator(func):
            return func
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _decorator


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


def build_lorenz_numba_dynamics(sigma: float, rho: float, beta: float, dt: float) -> NumbaDynamics:

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
    def f(x, t):
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
    def F_jac(x):
        xv = x[0]
        y = x[1]
        z = x[2]
        J = np.empty((3, 3))
        J[0, 0] = 1.0 + dt * (-sigma)
        J[0, 1] = dt * sigma
        J[0, 2] = 0.0
        J[1, 0] = dt * (rho - z)
        J[1, 1] = 1.0 + dt * (-1.0)
        J[1, 2] = dt * (-xv)
        J[2, 0] = dt * y
        J[2, 1] = dt * xv
        J[2, 2] = 1.0 + dt * (-beta)
        return J

    @njit(cache=True, fastmath=True)
    def H_jac(x):
        out = np.zeros((2, 3))
        out[0, 0] = 1.0
        out[1, 1] = 1.0
        return out

    return NumbaDynamics(f=f, h=h, F_jac=F_jac, H_jac=H_jac)
