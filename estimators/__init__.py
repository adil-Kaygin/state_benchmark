from .base import BaseEstimator
from .classical.kf import KalmanFilterEstimator
from .classical.ekf import EKFEstimator
from .classical.ukf import UKFEstimator
from .classical.pf import ParticleFilterEstimator
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

__all__ = [
    "BaseEstimator",
    "KalmanFilterEstimator",
    "EKFEstimator",
    "UKFEstimator",
    "ParticleFilterEstimator",
    "KalmanNetEstimator",
    "KalmanNetUncertaintyEstimator",
    "NeuralODEEstimator",
    "TransformerEstimator",
    "ESTIMATORS",
    "EXPERIMENTAL_ESTIMATORS",
]
