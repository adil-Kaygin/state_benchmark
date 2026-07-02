from __future__ import annotations

"""
Numba-jitted sequential filter loops shared by KF/EKF/UKF.

Kept in a dedicated module (rather than one monolithic filter file) so each
estimator file stays focused on its own contract implementation while the
hot inner loops live next to each other for easy comparison/maintenance.

Numba is a HARD dependency of the classical filters. There is no pure-NumPy
fallback: the entire benchmark relies exclusively on these @njit kernels (plus
the third-party torch-kf/torchfilter reference filters). Per the "fail fast and loud"
architectural rule, if numba is missing or fails to import, this module raises
ImportError at import time rather than silently degrading to slow/divergent
NumPy code.
"""

import numpy as np

try:
    from numba import njit
except ImportError as exc:  # fail fast and loud -- no NumPy fallback
    raise ImportError(
        "numba is required for the classical filters (KF/EKF/UKF). The "
        "pure-NumPy fallback has been removed; the benchmark relies exclusively "
        "on the @njit kernels in estimators/classical/_numba_kernels.py. "
        "Install numba (`pip install numba`) to run these estimators."
    ) from exc


# Ceiling on covariance magnitude. A KF/EKF whose model is a poor fit for a
# chaotic level (e.g. the linearized-at-origin KF, or the EKF when its estimate
# wanders far from the true Lorenz trajectory) drives P up exponentially through
# the F @ P @ F.T predict, eventually overflowing to inf/NaN and poisoning every
# later step. Capping P keeps the run finite -- the filter still reports its
# (legitimately bad) estimate instead of NaN -- which is the behaviour the
# benchmark wants to measure.
_COV_CEILING = 1.0e12


@njit(cache=True, fastmath=True)
def _bound_cov(P: np.ndarray) -> np.ndarray:
    """Symmetrize P and clip its entries to +/-_COV_CEILING so a diverging
    filter cannot overflow the predict step to inf/NaN."""
    P = 0.5 * (P + P.T)
    return np.minimum(np.maximum(P, -_COV_CEILING), _COV_CEILING)


@njit(cache=True, fastmath=True)
def _wrap_innovation(innov: np.ndarray, angular_mask: np.ndarray) -> np.ndarray:
    """Wrap the angular components of an innovation vector to (-pi, pi].

    angular_mask is a float [ny] with 1.0 where the observation component is an
    angle (a bearing) and 0.0 otherwise (Issues 5/6). For those components the
    raw residual y - h(x) can be ~2*pi wrong near the +/-pi branch cut; atan2(sin,
    cos) folds it back onto (-pi, pi]. Non-angular components are left untouched.
    A caller with no angular components passes an all-zero mask, which is a no-op.
    """
    ny = innov.shape[0]
    out = innov.copy()
    for j in range(ny):
        if angular_mask[j] != 0.0:
            out[j] = np.arctan2(np.sin(innov[j]), np.cos(innov[j]))
    return out


@njit(cache=True, fastmath=True)
def _robust_chol(M: np.ndarray, nx: int, I: np.ndarray) -> np.ndarray:
    """Cholesky of a symmetric matrix that has drifted to (near-)singular.

    Retries with an escalating diagonal jitter scaled to the magnitude of M
    (via its mean diagonal) so a single fixed floor can't be too small for
    large-magnitude states. Numba-friendly: plain loop + flag, no for/else.
    """
    scale = np.trace(M) / nx
    if scale < 1.0:
        scale = 1.0
    jitter = 1e-9 * scale
    for _ in range(12):
        try:
            return np.linalg.cholesky(M + jitter * I)
        except Exception:
            jitter *= 10.0
    # Final attempt; let it raise if even this fails so the caller sees it.
    return np.linalg.cholesky(M + jitter * I)


def angular_mask_float(filter_model, ny: int) -> np.ndarray:
    """Build the float [ny] angular-observation mask (1.0 where a component is an
    angle to wrap, else 0.0) from FilterModel.angular_obs_mask, defaulting to
    all-zeros (no wrapping) for the scalar-observation levels. Kept here so
    EKF/UKF (custom and the torch-kf/torchfilter references) all derive the mask
    identically.

    Raises ValueError if a provided mask has the wrong length -- a mislabelled
    mask that silently wraps the wrong (or no) component is exactly the footgun
    the fail-fast rule guards against.
    """
    mask = getattr(filter_model, "angular_obs_mask", None)
    if mask is None:
        return np.zeros(ny, dtype=np.float64)
    mask = np.asarray(mask)
    if mask.shape != (ny,):
        raise ValueError(
            f"angular_obs_mask must have shape ({ny},) matching the observation "
            f"dimension; got {mask.shape}."
        )
    return np.ascontiguousarray(mask, dtype=np.float64)


def assert_linear_model(f, h, F: np.ndarray, H: np.ndarray, nx: int, ny: int) -> None:
    """Raise ValueError unless f(x)=F@x and h(x)=H@x exactly (a linear model).

    The Kalman filter is the optimal estimator only for a linear-Gaussian model
    and kf_loop_batch hardcodes the f(x)=F@x, h(x)=H@x assumption. Per the
    "fail fast and loud" rule, running the KF on a nonlinear FilterModel (e.g.
    pendulum/lorenz/nonlinear) must crash here rather than silently linearizing
    at the origin and reporting numbers that look like KF results but aren't.

    Linearity is probed at several points (including non-zero offsets) so a map
    that only coincidentally agrees with F@x at the origin cannot slip through.
    F/H shapes are validated too, since a wrong-shaped Jacobian is just as
    invalid as a nonlinear one.
    """
    F = np.asarray(F, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if F.shape != (nx, nx):
        raise ValueError(
            f"KalmanFilter requires F with shape ({nx}, {nx}); got {F.shape}."
        )
    if H.shape != (ny, nx):
        raise ValueError(
            f"KalmanFilter requires H with shape ({ny}, {nx}); got {H.shape}."
        )

    rng = np.random.default_rng(0)
    # Probe at the origin, the canonical basis directions, and random points.
    probes = [np.zeros(nx)]
    for j in range(nx):
        e = np.zeros(nx)
        e[j] = 3.0
        probes.append(e)
    for _ in range(4):
        probes.append(rng.standard_normal(nx) * 5.0)

    for x_probe in probes:
        f_actual = np.asarray(f(x_probe), dtype=np.float64)
        h_actual = np.asarray(h(x_probe), dtype=np.float64)
        if not (
            np.allclose(f_actual, F @ x_probe, atol=1e-8)
            and np.allclose(h_actual, H @ x_probe, atol=1e-8)
        ):
            raise ValueError(
                "KalmanFilterEstimator requires a LINEAR model (f(x)=F@x and "
                "h(x)=H@x exactly, e.g. LinearBenchmark). This FilterModel's "
                "f/h are nonlinear, so a Kalman filter would silently linearize "
                "at the origin instead of running a valid filter. Use EKF/UKF/PF "
                "for nonlinear systems."
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
    not a hardcoded zeros/identity default -- otherwise KF/EKF/UKF would stop
    being a fair comparison on the same dataset (EKF/UKF also use the real prior).
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
        P_pred = _bound_cov(F @ P @ F.T + Q)

        y = observations[t]
        S = H @ P_pred @ Ht + R
        K = P_pred @ Ht @ np.linalg.inv(S)
        x = x_pred + K @ (y - H @ x_pred)
        P = _bound_cov((I - K @ H) @ P_pred)
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


@njit(cache=True, fastmath=True)
def ekf_loop(
    f,  # njit f(x, t) -> [nx]
    h,  # njit h(x, t) -> [ny]
    Fj,  # njit F_jac(x) -> [nx, nx]
    Hj,  # njit H_jac(x) -> [ny, nx]
    Q: np.ndarray,
    R: np.ndarray,
    observations: np.ndarray,  # [T, ny]
    timestamps: np.ndarray,  # [T]
    x0: np.ndarray,  # [nx]
    P0: np.ndarray,  # [nx, nx]
    angular_mask: np.ndarray,  # [ny] float, 1.0 where the obs is an angle
) -> np.ndarray:
    """General (nonlinear) EKF over one trajectory, JIT-compiled.

    f/h/Fj/Hj are themselves @njit functions supplied by the benchmark
    (FilterModel.numba); numba's first-class-function support lets them be
    passed into and called from this compiled loop, so the EKF runs fully in
    machine code. This is the sole EKF implementation -- there is no pure-Python
    fallback.

    angular_mask (Issues 5/6) wraps the innovation's angular (bearing) components
    to (-pi, pi] before the update; an all-zero mask is a no-op for the existing
    scalar-observation levels.
    """
    T, ny = observations.shape
    nx = Q.shape[0]
    estimates = np.zeros((T, nx))

    x = x0.copy()
    P = P0.copy()
    I = np.eye(nx)

    for t in range(T):
        x_pred = f(x, timestamps[t])
        F = Fj(x)
        P_pred = _bound_cov(F @ P @ F.T + Q)

        H = Hj(x_pred)
        y_pred = h(x_pred, timestamps[t])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        innov = _wrap_innovation(observations[t] - y_pred, angular_mask)
        x = x_pred + K @ innov
        P = _bound_cov((I - K @ H) @ P_pred)
        estimates[t] = x

    return estimates


@njit(cache=True, fastmath=True)
def ekf_loop_batch(
    f, h, Fj, Hj,
    Q: np.ndarray,
    R: np.ndarray,
    observations: np.ndarray,  # [N, T, ny]
    timestamps: np.ndarray,  # [T]
    x0: np.ndarray,
    P0: np.ndarray,
    angular_mask: np.ndarray,  # [ny]
) -> np.ndarray:
    N = observations.shape[0]
    T = observations.shape[1]
    nx = Q.shape[0]
    estimates = np.zeros((N, T, nx))
    for i in range(N):
        estimates[i] = ekf_loop(f, h, Fj, Hj, Q, R, observations[i], timestamps, x0, P0, angular_mask)
    return estimates


@njit(cache=True, fastmath=True)
def ukf_loop(
    f,  # njit f(x, t) -> [nx]
    h,  # njit h(x, t) -> [ny]
    Q: np.ndarray,
    R: np.ndarray,
    observations: np.ndarray,  # [T, ny]
    timestamps: np.ndarray,  # [T]
    alpha: float,
    beta: float,
    kappa: float,
    x0: np.ndarray,
    P0: np.ndarray,
    angular_mask: np.ndarray,  # [ny] float, 1.0 where the obs is an angle
) -> np.ndarray:
    """General (nonlinear) UKF over one trajectory, JIT-compiled.

    Propagates each sigma point through the benchmark's own @njit f/h, so it is
    correct for every level (linear/pendulum/nonlinear/lorenz/vehicle_tracking).
    This is the sole UKF implementation -- there is no pure-NumPy fallback.

    angular_mask (Issues 5/6) wraps every measurement residual with an angular
    (bearing) component to (-pi, pi]: both the sigma-point spreads sig_h - y_pred
    that build S/Pxy and the final innovation y - y_pred. An all-zero mask is a
    no-op for the existing scalar-observation levels.
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
    I = np.eye(nx)

    for t in range(T):
        # Symmetrize and Cholesky; on failure, retry with an escalating jitter
        # scaled to the magnitude of P so a fixed floor can't be too small.
        P = 0.5 * (P + P.T)
        M = c * P
        try:
            sqrtP = np.linalg.cholesky(M)
        except Exception:
            sqrtP = _robust_chol(M, nx, I)

        sigmas = np.empty((n_sig, nx))
        sigmas[0] = x
        for i in range(nx):
            col = sqrtP[:, i]
            sigmas[i + 1] = x + col
            sigmas[nx + i + 1] = x - col

        sig_f = np.empty((n_sig, nx))
        for i in range(n_sig):
            sig_f[i] = f(sigmas[i], timestamps[t])

        x_pred = np.zeros(nx)
        for i in range(n_sig):
            x_pred += Wm[i] * sig_f[i]
        P_pred = Q.copy()
        for i in range(n_sig):
            d = sig_f[i] - x_pred
            P_pred += Wc[i] * np.outer(d, d)

        sig_h = np.empty((n_sig, ny))
        for i in range(n_sig):
            sig_h[i] = h(sig_f[i], timestamps[t])

        y_pred = np.zeros(ny)
        for i in range(n_sig):
            y_pred += Wm[i] * sig_h[i]

        S = R.copy()
        Pxy = np.zeros((nx, ny))
        for i in range(n_sig):
            dy = _wrap_innovation(sig_h[i] - y_pred, angular_mask)
            dx = sig_f[i] - x_pred
            S += Wc[i] * np.outer(dy, dy)
            Pxy += Wc[i] * np.outer(dx, dy)

        K = Pxy @ np.linalg.inv(S)
        innov = _wrap_innovation(observations[t] - y_pred, angular_mask)
        x = x_pred + K @ innov
        P = P_pred - K @ S @ K.T
        P = 0.5 * (P + P.T)  # keep symmetric against numerical drift
        estimates[t] = x

    return estimates


@njit(cache=True, fastmath=True)
def ukf_loop_batch(
    f, h,
    Q: np.ndarray,
    R: np.ndarray,
    observations: np.ndarray,  # [N, T, ny]
    timestamps: np.ndarray,
    alpha: float,
    beta: float,
    kappa: float,
    x0: np.ndarray,
    P0: np.ndarray,
    angular_mask: np.ndarray,  # [ny]
) -> np.ndarray:
    N = observations.shape[0]
    T = observations.shape[1]
    nx = Q.shape[0]
    estimates = np.zeros((N, T, nx))
    for i in range(N):
        estimates[i] = ukf_loop(
            f, h, Q, R, observations[i], timestamps, alpha, beta, kappa, x0, P0, angular_mask
        )
    return estimates


# --- Covariance-returning variants (Issue 7) --------------------------------
#
# These mirror the point-estimate loops above one-for-one but ALSO record the
# posterior covariance P at every step, so estimate_with_covariance() can feed
# NEES/NLL. The filter math is identical; the only addition is `covs[t] = P`.
# Kept as separate kernels (rather than always returning P) so the hot point-
# estimate path stays allocation-light and estimate() is unchanged.


@njit(cache=True, fastmath=True)
def kf_loop_cov(F, H, Q, R, observations, x0, P0):
    """Linear KF over one trajectory, returning (estimates [T,nx], covs [T,nx,nx])
    -- the posterior P after each update. Same recursion as kf_loop."""
    T, ny = observations.shape
    nx = Q.shape[0]
    estimates = np.zeros((T, nx))
    covs = np.zeros((T, nx, nx))

    x = x0.copy()
    P = P0.copy()
    I = np.eye(nx)
    Ht = H.T

    for t in range(T):
        x_pred = F @ x
        P_pred = _bound_cov(F @ P @ F.T + Q)

        y = observations[t]
        S = H @ P_pred @ Ht + R
        K = P_pred @ Ht @ np.linalg.inv(S)
        x = x_pred + K @ (y - H @ x_pred)
        P = _bound_cov((I - K @ H) @ P_pred)
        estimates[t] = x
        covs[t] = P

    return estimates, covs


@njit(cache=True, fastmath=True)
def kf_loop_batch_cov(F, H, Q, R, observations, x0, P0):
    N = observations.shape[0]
    T, ny = observations.shape[1], observations.shape[2]
    nx = Q.shape[0]
    estimates = np.zeros((N, T, nx))
    covs = np.zeros((N, T, nx, nx))
    for i in range(N):
        est_i, cov_i = kf_loop_cov(F, H, Q, R, observations[i], x0, P0)
        estimates[i] = est_i
        covs[i] = cov_i
    return estimates, covs


@njit(cache=True, fastmath=True)
def ekf_loop_cov(f, h, Fj, Hj, Q, R, observations, timestamps, x0, P0, angular_mask):
    """EKF over one trajectory, returning (estimates [T,nx], covs [T,nx,nx]).
    Same recursion as ekf_loop (incl. the angular innovation wrap)."""
    T, ny = observations.shape
    nx = Q.shape[0]
    estimates = np.zeros((T, nx))
    covs = np.zeros((T, nx, nx))

    x = x0.copy()
    P = P0.copy()
    I = np.eye(nx)

    for t in range(T):
        x_pred = f(x, timestamps[t])
        F = Fj(x)
        P_pred = _bound_cov(F @ P @ F.T + Q)

        H = Hj(x_pred)
        y_pred = h(x_pred, timestamps[t])
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        innov = _wrap_innovation(observations[t] - y_pred, angular_mask)
        x = x_pred + K @ innov
        P = _bound_cov((I - K @ H) @ P_pred)
        estimates[t] = x
        covs[t] = P

    return estimates, covs


@njit(cache=True, fastmath=True)
def ekf_loop_batch_cov(f, h, Fj, Hj, Q, R, observations, timestamps, x0, P0, angular_mask):
    N = observations.shape[0]
    T = observations.shape[1]
    nx = Q.shape[0]
    estimates = np.zeros((N, T, nx))
    covs = np.zeros((N, T, nx, nx))
    for i in range(N):
        est_i, cov_i = ekf_loop_cov(
            f, h, Fj, Hj, Q, R, observations[i], timestamps, x0, P0, angular_mask
        )
        estimates[i] = est_i
        covs[i] = cov_i
    return estimates, covs


@njit(cache=True, fastmath=True)
def ukf_loop_cov(f, h, Q, R, observations, timestamps, alpha, beta, kappa, x0, P0, angular_mask):
    """UKF over one trajectory, returning (estimates [T,nx], covs [T,nx,nx]).
    Same recursion as ukf_loop (incl. the angular residual wrap)."""
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
    covs = np.zeros((T, nx, nx))
    x = x0.copy()
    P = P0.copy()
    I = np.eye(nx)

    for t in range(T):
        P = 0.5 * (P + P.T)
        M = c * P
        try:
            sqrtP = np.linalg.cholesky(M)
        except Exception:
            sqrtP = _robust_chol(M, nx, I)

        sigmas = np.empty((n_sig, nx))
        sigmas[0] = x
        for i in range(nx):
            col = sqrtP[:, i]
            sigmas[i + 1] = x + col
            sigmas[nx + i + 1] = x - col

        sig_f = np.empty((n_sig, nx))
        for i in range(n_sig):
            sig_f[i] = f(sigmas[i], timestamps[t])

        x_pred = np.zeros(nx)
        for i in range(n_sig):
            x_pred += Wm[i] * sig_f[i]
        P_pred = Q.copy()
        for i in range(n_sig):
            d = sig_f[i] - x_pred
            P_pred += Wc[i] * np.outer(d, d)

        sig_h = np.empty((n_sig, ny))
        for i in range(n_sig):
            sig_h[i] = h(sig_f[i], timestamps[t])

        y_pred = np.zeros(ny)
        for i in range(n_sig):
            y_pred += Wm[i] * sig_h[i]

        S = R.copy()
        Pxy = np.zeros((nx, ny))
        for i in range(n_sig):
            dy = _wrap_innovation(sig_h[i] - y_pred, angular_mask)
            dx = sig_f[i] - x_pred
            S += Wc[i] * np.outer(dy, dy)
            Pxy += Wc[i] * np.outer(dx, dy)

        K = Pxy @ np.linalg.inv(S)
        innov = _wrap_innovation(observations[t] - y_pred, angular_mask)
        x = x_pred + K @ innov
        P = P_pred - K @ S @ K.T
        P = 0.5 * (P + P.T)
        estimates[t] = x
        covs[t] = P

    return estimates, covs


@njit(cache=True, fastmath=True)
def ukf_loop_batch_cov(f, h, Q, R, observations, timestamps, alpha, beta, kappa, x0, P0, angular_mask):
    N = observations.shape[0]
    T = observations.shape[1]
    nx = Q.shape[0]
    estimates = np.zeros((N, T, nx))
    covs = np.zeros((N, T, nx, nx))
    for i in range(N):
        est_i, cov_i = ukf_loop_cov(
            f, h, Q, R, observations[i], timestamps, alpha, beta, kappa, x0, P0, angular_mask
        )
        estimates[i] = est_i
        covs[i] = cov_i
    return estimates, covs

