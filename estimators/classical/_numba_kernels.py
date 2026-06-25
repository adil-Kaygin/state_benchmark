from __future__ import annotations

"""
Numba-jitted sequential filter loops shared by KF/EKF/UKF.

Kept in a dedicated module (rather than one monolithic filter file) so each
estimator file stays focused on its own contract implementation while the
hot inner loops live next to each other for easy comparison/maintenance.

Numba is an optional accelerator: every estimator falls back to a pure-NumPy
loop when numba is not installed, so the framework has no hard dependency on it.
"""

from typing import Tuple

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _decorator(func):
            return func
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return _decorator


def assert_linear_model(f, h, F: np.ndarray, H: np.ndarray, nx: int, ny: int) -> None:
    """Raise ValueError unless f(x)=Fx and h(x)=Hx on a probe vector.

    kf_loop_batch/ukf_linear_loop hardcode the f(x)=Fx, h(x)=Hx assumption for
    speed; silently enabling use_numba=True on a nonlinear FilterModel (e.g.
    pendulum/lorenz/nonlinear) would linearize at the origin instead of
    erroring, producing numbers that look like KF/UKF results but aren't.
    """
    rng = np.random.default_rng(0)
    x_probe = rng.standard_normal(nx)
    f_actual = np.asarray(f(x_probe))
    f_linear = F @ x_probe
    h_actual = np.asarray(h(x_probe))
    h_linear = H @ x_probe
    if not (np.allclose(f_actual, f_linear, atol=1e-8) and np.allclose(h_actual, h_linear, atol=1e-8)):
        raise ValueError(
            "use_numba=True requires f(x)=F@x and h(x)=H@x (e.g. LinearBenchmark); "
            "this FilterModel's f/h are nonlinear, so the fast numba path would "
            "silently linearize at the origin instead of running the real filter. "
            "Set use_numba=False for this benchmark."
        )


@njit(cache=True, fastmath=True)
def kf_loop(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    observations: np.ndarray,  # [T, ny]
    x0: np.ndarray,  # [nx]
    P0: np.ndarray,  # [nx, nx]
) -> np.ndarray:
    """Linear Kalman filter over a single trajectory. Returns estimates [T, nx].

    x0/P0 must be the benchmark's generative prior (FilterModel.x0_mean/x0_cov),
    not a hardcoded zeros/identity default -- otherwise this diverges from the
    pure-NumPy/EKF paths, which do use the real prior, and KF/EKF/UKF stop being
    a fair comparison on the same dataset.
    """
    T, ny = observations.shape
    nx = Q.shape[0]
    estimates = np.zeros((T, nx))

    x = x0.copy()
    P = P0.copy()
    I = np.eye(nx)
    Ht = H.T

    for t in range(T):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        y = observations[t]
        S = H @ P_pred @ Ht + R
        K = P_pred @ Ht @ np.linalg.inv(S)
        x = x_pred + K @ (y - H @ x_pred)
        P = (I - K @ H) @ P_pred
        estimates[t] = x

    return estimates


@njit(cache=True, fastmath=True)
def kf_loop_batch(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    observations: np.ndarray,  # [N, T, ny]
    x0: np.ndarray,  # [nx]
    P0: np.ndarray,  # [nx, nx]
) -> np.ndarray:
    N = observations.shape[0]
    T, ny = observations.shape[1], observations.shape[2]
    nx = Q.shape[0]
    estimates = np.zeros((N, T, nx))
    for i in range(N):
        estimates[i] = kf_loop(F, H, Q, R, observations[i], x0, P0)
    return estimates


def ukf_sigma_points(
    x: np.ndarray, P: np.ndarray, nx: int, alpha: float, beta: float, kappa: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plain-NumPy sigma point generator, reused by the (non-jitted) UKF
    estimator since sigma points require calling arbitrary Python f()/h()
    callables from FilterModel, which numba cannot compile generically."""
    lam = alpha ** 2 * (nx + kappa) - nx
    try:
        L = np.linalg.cholesky((nx + lam) * P)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky((nx + lam) * (P + 1e-6 * np.eye(nx)))

    pts = np.zeros((2 * nx + 1, nx))
    pts[0] = x
    for j in range(nx):
        pts[j + 1] = x + L[:, j]
        pts[nx + j + 1] = x - L[:, j]

    Wm = np.full(2 * nx + 1, 1.0 / (2.0 * (nx + lam)))
    Wc = np.full(2 * nx + 1, 1.0 / (2.0 * (nx + lam)))
    Wm[0] = lam / (nx + lam)
    Wc[0] = lam / (nx + lam) + (1.0 - alpha ** 2 + beta)

    return pts, Wm, Wc


@njit(cache=True, fastmath=True)
def ukf_linear_loop(
    F: np.ndarray,
    H: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    observations: np.ndarray,  # [T, ny]
    alpha: float,
    beta: float,
    kappa: float,
    x0: np.ndarray,  # [nx]
    P0: np.ndarray,  # [nx, nx]
) -> np.ndarray:
    """Fast UKF path for benchmarks whose f/h are linear (f(x)=Fx, h(x)=Hx),
    e.g. LinearBenchmark. Runs the full sigma-point machinery under njit
    since F/H are constant matrices rather than arbitrary Python callables.

    x0/P0 must be the benchmark's generative prior -- see kf_loop's docstring
    for why a hardcoded zeros/identity default would make this path diverge
    from the pure-NumPy UKF/EKF on the same dataset.
    """
    T, ny = observations.shape
    nx = Q.shape[0]
    n_sig = 2 * nx + 1

    lam = alpha * alpha * (nx + kappa) - nx
    c = nx + lam
    Wm = np.empty(n_sig)
    Wc = np.empty(n_sig)
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha * alpha + beta)
    for i in range(1, n_sig):
        Wm[i] = 1.0 / (2.0 * c)
        Wc[i] = 1.0 / (2.0 * c)

    estimates = np.zeros((T, nx))
    x = x0.copy()
    P = P0.copy()

    for t in range(T):
        sqrtP = np.linalg.cholesky(c * P)
        sigmas = np.empty((n_sig, nx))
        sigmas[0] = x
        for i in range(nx):
            col = sqrtP[:, i]
            sigmas[i + 1] = x + col
            sigmas[nx + i + 1] = x - col

        sig_f = np.empty((n_sig, nx))
        for i in range(n_sig):
            sig_f[i] = F @ sigmas[i]

        x_pred = np.zeros(nx)
        for i in range(n_sig):
            x_pred += Wm[i] * sig_f[i]
        P_pred = Q.copy()
        for i in range(n_sig):
            d = sig_f[i] - x_pred
            P_pred += Wc[i] * np.outer(d, d)

        sqrtPp = np.linalg.cholesky(c * P_pred)
        sig2 = np.empty((n_sig, nx))
        sig2[0] = x_pred
        for i in range(nx):
            col = sqrtPp[:, i]
            sig2[i + 1] = x_pred + col
            sig2[nx + i + 1] = x_pred - col

        sig_h = np.empty((n_sig, ny))
        for i in range(n_sig):
            sig_h[i] = H @ sig2[i]

        y_pred = np.zeros(ny)
        for i in range(n_sig):
            y_pred += Wm[i] * sig_h[i]

        S = R.copy()
        Pxy = np.zeros((nx, ny))
        for i in range(n_sig):
            dy = sig_h[i] - y_pred
            dx = sig2[i] - x_pred
            S += Wc[i] * np.outer(dy, dy)
            Pxy += Wc[i] * np.outer(dx, dy)

        K = Pxy @ np.linalg.inv(S)
        x = x_pred + K @ (observations[t] - y_pred)
        P = P_pred - K @ S @ K.T
        estimates[t] = x

    return estimates
