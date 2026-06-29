from __future__ import annotations

"""
Uncertainty / posterior-consistency scoring for the filters' reported
covariance `P` -- the part of a Bayesian filter's output that point-RMSE
ignores entirely.

A Kalman-family filter outputs a *distribution* N(x̂, P), not just a point x̂.
The standard ways to score that distribution against the truth are:

  - NEES (Normalized Estimation Error Squared): the consistency check from
    Bar-Shalom et al. e_t = (x - x̂)ᵀ P⁻¹ (x - x̂). For a consistent filter
    E[NEES] = nx, so the time/ensemble-averaged NEES should sit near nx (with
    the chi-square interval as the acceptance band). NEES << nx ⇒ the filter is
    under-confident (P too large); NEES >> nx ⇒ over-confident (P too small).

  - NLL (Gaussian Negative Log-Likelihood): the proper scoring rule
    ½[(x-x̂)ᵀ P⁻¹ (x-x̂) + ln det(2π P)]. Lower is better; unlike NEES it
    penalizes both miscalibrated covariance AND large point error, so it ranks
    estimators on the full posterior quality.

Per the "fail fast and loud" rule these crash on a non-positive-definite P,
mismatched shapes, or any non-finite input rather than returning a dummy 0.0 /
NaN: a covariance that cannot be inverted is a real defect in the filter, not
something to paper over.
"""

import numpy as np


def _validate(estimates: np.ndarray, targets: np.ndarray, covariances: np.ndarray):
    estimates = np.asarray(estimates, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)

    if estimates.shape != targets.shape:
        raise ValueError(
            f"estimates and targets must have the same shape; got "
            f"{estimates.shape} vs {targets.shape}."
        )
    if estimates.ndim != 3:
        raise ValueError(
            f"estimates/targets must be 3-D [N, T, nx]; got shape {estimates.shape}."
        )

    N, T, nx = estimates.shape
    if covariances.shape != (N, T, nx, nx):
        raise ValueError(
            f"covariances must have shape [N, T, nx, nx] = {(N, T, nx, nx)}; "
            f"got {covariances.shape}."
        )

    for arr, label in ((estimates, "estimates"), (targets, "targets"), (covariances, "covariances")):
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"{label} contains non-finite values (inf/NaN); cannot score "
                "uncertainty on a diverged/invalid filter output."
            )

    return estimates, targets, covariances, N, T, nx


def compute_nees(
    estimates: np.ndarray,
    targets: np.ndarray,
    covariances: np.ndarray,
) -> float:
    """Mean Normalized Estimation Error Squared over all [N, T] estimates.

    NEES_t = (x_t - x̂_t)ᵀ P_t⁻¹ (x_t - x̂_t). Returns the mean over the
    N*T estimates. Compare against the state dimension nx: a consistent filter
    gives mean NEES ≈ nx (see compute_nees_chi2_bounds for the acceptance band).

    Parameters
    ----------
    estimates    : [N, T, nx]   filter mean x̂
    targets      : [N, T, nx]   ground-truth state x
    covariances  : [N, T, nx, nx]   filter posterior covariance P (must be PD)

    Raises
    ------
    ValueError on shape mismatch, non-finite input, or non-positive-definite P.
    """
    estimates, targets, covariances, N, T, nx = _validate(estimates, targets, covariances)

    err = targets - estimates  # [N, T, nx]
    flat_err = err.reshape(N * T, nx)
    flat_cov = covariances.reshape(N * T, nx, nx)

    total = 0.0
    for k in range(N * T):
        e = flat_err[k]
        P = flat_cov[k]
        try:
            solved = np.linalg.solve(P, e)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"covariance P at flattened index {k} is singular and cannot be "
                "inverted for NEES; the filter reported an invalid posterior."
            ) from exc
        total += float(e @ solved)

    return total / (N * T)


def compute_nees_chi2_bounds(nx: int, num_samples: int, confidence: float = 0.95):
    """Two-sided chi-square acceptance interval for the *mean* NEES.

    For `num_samples` independent nx-dimensional Gaussian errors, num_samples *
    mean-NEES ~ chi2(num_samples * nx). Returns (lower, upper) bounds on the
    mean NEES at the given confidence; a mean NEES inside [lower, upper] is
    consistent, below it under-confident, above it over-confident.

    Requires scipy; raises ImportError if absent (fail fast -- no silent skip).
    """
    if nx <= 0 or num_samples <= 0:
        raise ValueError(f"nx and num_samples must be positive; got {nx}, {num_samples}.")
    try:
        from scipy.stats import chi2
    except ImportError as exc:
        raise ImportError(
            "scipy is required for the chi-square NEES acceptance bounds "
            "(metrics.uncertainty.compute_nees_chi2_bounds)."
        ) from exc

    dof = num_samples * nx
    alpha = 1.0 - confidence
    lower = chi2.ppf(alpha / 2.0, dof) / num_samples
    upper = chi2.ppf(1.0 - alpha / 2.0, dof) / num_samples
    return float(lower), float(upper)


def compute_nll(
    estimates: np.ndarray,
    targets: np.ndarray,
    covariances: np.ndarray,
) -> float:
    """Mean Gaussian Negative Log-Likelihood of the truth under N(x̂, P).

    NLL_t = ½[(x_t - x̂_t)ᵀ P_t⁻¹ (x_t - x̂_t) + ln det(2π P_t)].
    Returns the mean over the N*T estimates (lower is better). Proper scoring
    rule: penalizes both point error and miscalibrated covariance.

    Parameters mirror compute_nees. Raises ValueError on shape mismatch,
    non-finite input, or non-positive-definite P.
    """
    estimates, targets, covariances, N, T, nx = _validate(estimates, targets, covariances)

    err = targets - estimates
    flat_err = err.reshape(N * T, nx)
    flat_cov = covariances.reshape(N * T, nx, nx)

    log_2pi = np.log(2.0 * np.pi)
    total = 0.0
    for k in range(N * T):
        e = flat_err[k]
        P = flat_cov[k]
        # Cholesky both inverts (via solve) and gives a stable log-det; it also
        # raises on a non-PD P, which is exactly the failure we want surfaced.
        try:
            L = np.linalg.cholesky(P)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"covariance P at flattened index {k} is not positive-definite; "
                "cannot evaluate the Gaussian NLL of an invalid posterior."
            ) from exc
        z = np.linalg.solve(L, e)
        maha = float(z @ z)
        log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
        total += 0.5 * (maha + nx * log_2pi + log_det)

    return total / (N * T)
