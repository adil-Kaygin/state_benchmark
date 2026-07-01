from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseEstimator(ABC):

    # Capability flag (Issue 7): True iff this estimator reports a calibrated
    # posterior covariance P and therefore implements estimate_with_covariance().
    # The classical Kalman-family filters (KF/EKF/UKF) set this True; the neural
    # filters have no calibrated P by default and leave it False, so the NEES/NLL
    # table skips them rather than fabricating a covariance. Overridden as a
    # plain class attribute by the subclasses that opt in.
    returns_covariance: bool = False

    @property
    @abstractmethod
    def estimator_name(self) -> str:
        pass

    @property
    @abstractmethod
    def estimator_type(self) -> str:
        pass

    @abstractmethod
    def fit(self, train_dataset, val_dataset) -> None:
        pass

    @abstractmethod
    def estimate(self, dataset):
        pass

    def estimate_with_covariance(self, dataset):
        """Return (estimates [N, T, nx], covariances [N, T, nx, nx]) for the
        uncertainty metrics (NEES/NLL). Opt-in: only estimators with
        returns_covariance = True implement this. Per the fail-fast rule the
        default raises NotImplementedError rather than returning a dummy P -- a
        fabricated covariance would silently corrupt the consistency scores.
        estimate() is unchanged (point estimates only), so nothing that calls it
        breaks."""
        raise NotImplementedError(
            f"{type(self).__name__} does not report a calibrated covariance "
            "(returns_covariance is False); NEES/NLL cannot be computed for it. "
            "Only the classical Kalman-family filters implement "
            "estimate_with_covariance()."
        )

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> BaseEstimator:
        pass
