from .base import BaseEstimator
from .classical.kf import KalmanFilterEstimator
from .classical.ekf import EKFEstimator
from .classical.ukf import UKFEstimator
from .classical.pf import ParticleFilterEstimator
from .classical.filterpy_filters import FilterpyKFEstimator, FilterpyEKFEstimator, FilterpyUKFEstimator
from .neural.kalmannet import KalmanNetEstimator, KalmanNetUncertaintyEstimator
from .neural.neural_ode import NeuralODEEstimator
from .neural.transformer import TransformerEstimator

# Estimators benchmarked by default (see notebooks/experiment_*.py).
# NeuralODEEstimator/TransformerEstimator are excluded here: they are stubs
# whose fit()/estimate() raise NotImplementedError (see estimators/neural/),
# so including them would make any sweep over ESTIMATORS crash. They live in
# EXPERIMENTAL_ESTIMATORS until implemented.
ESTIMATORS = {
    "kf": KalmanFilterEstimator,
    "ekf": EKFEstimator,
    "ukf": UKFEstimator,
    "pf": ParticleFilterEstimator,
    "kalmannet": KalmanNetEstimator,
}

# Opt-in estimators not run by the default benchmark notebook.
EXPERIMENTAL_ESTIMATORS = {
    "kalmannet_uncertainty": KalmanNetUncertaintyEstimator,
    "neural_ode": NeuralODEEstimator,
    "transformer": TransformerEstimator,
}

# filterpy-backed re-implementations of kf/ekf/ukf, used as an independent
# cross-check against this repo's custom NumPy/Numba filters. Importing these
# classes never requires filterpy to be installed; only instantiating one
# does (see filterpy_filters._require_filterpy).
REFERENCE_ESTIMATORS = {
    "filterpy_kf": FilterpyKFEstimator,
    "filterpy_ekf": FilterpyEKFEstimator,
    "filterpy_ukf": FilterpyUKFEstimator,
}

__all__ = [
    "BaseEstimator",
    "KalmanFilterEstimator",
    "EKFEstimator",
    "UKFEstimator",
    "ParticleFilterEstimator",
    "FilterpyKFEstimator",
    "FilterpyEKFEstimator",
    "FilterpyUKFEstimator",
    "KalmanNetEstimator",
    "KalmanNetUncertaintyEstimator",
    "NeuralODEEstimator",
    "TransformerEstimator",
    "ESTIMATORS",
    "EXPERIMENTAL_ESTIMATORS",
    "REFERENCE_ESTIMATORS",
]
