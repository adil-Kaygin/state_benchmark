from .base import BaseEstimator
from .classical.kf import KalmanFilterEstimator
from .classical.ekf import EKFEstimator
from .classical.ukf import UKFEstimator
from .classical.pf import ParticleFilterEstimator
from .classical.filterpy_filters import FilterpyKFEstimator, FilterpyEKFEstimator, FilterpyUKFEstimator
from .neural.kalmannet import KalmanNetEstimator, KalmanNetUncertaintyEstimator
from .neural.neural_ode import NeuralODEEstimator
from .neural.pinn import PINNFilterEstimator
from .neural.transformer import TransformerEstimator
from .neural.mamba import MambaEstimator

# Estimators benchmarked by default (see notebooks/experiment_*.py).
# The four learned filters (neural_ode/pinn/transformer/mamba) are now fully
# implemented (issues 1-4) and run in the standard sweep alongside KalmanNet.
ESTIMATORS = {
    "kf": KalmanFilterEstimator,
    "ekf": EKFEstimator,
    "ukf": UKFEstimator,
    "pf": ParticleFilterEstimator,
    "kalmannet": KalmanNetEstimator,
    "neural_ode": NeuralODEEstimator,
    "pinn": PINNFilterEstimator,
    "transformer": TransformerEstimator,
    "mamba": MambaEstimator,
}

# Opt-in estimators not run by the default benchmark notebook.
EXPERIMENTAL_ESTIMATORS = {
    "kalmannet_uncertainty": KalmanNetUncertaintyEstimator,
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
    "PINNFilterEstimator",
    "TransformerEstimator",
    "MambaEstimator",
    "ESTIMATORS",
    "EXPERIMENTAL_ESTIMATORS",
    "REFERENCE_ESTIMATORS",
]
