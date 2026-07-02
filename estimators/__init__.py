from .base import BaseEstimator
from .classical.kf import KalmanFilterEstimator
from .classical.ekf import EKFEstimator
from .classical.ukf import UKFEstimator
from .classical.pf import ParticleFilterEstimator
from .classical.torchkf_filters import (
    TorchKFKFEstimator,
    TorchKFEKFEstimator,
    TorchKFUKFEstimator,
    TorchKFPFEstimator,
)
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
    "torchkf_pf": TorchKFPFEstimator,  # torchfilter particle filter, for future use
}

# torch-kf / torchfilter-backed re-implementations of kf/ekf/ukf, used as an
# independent cross-check against this repo's custom NumPy/Numba filters. The KF
# is backed by torch-kf (linear-only); the EKF/UKF by torchfilter. Importing
# these classes never requires either package to be installed; only instantiating
# one does (see torchkf_filters._require_torchkf / _require_torchfilter).
# torchkf_pf (torchfilter's particle filter) is added for future use and lives in
# EXPERIMENTAL_ESTIMATORS, not here -- it reports point estimates only.
REFERENCE_ESTIMATORS = {
    "torchkf_kf": TorchKFKFEstimator,
    "torchkf_ekf": TorchKFEKFEstimator,
    "torchkf_ukf": TorchKFUKFEstimator,
}

__all__ = [
    "BaseEstimator",
    "KalmanFilterEstimator",
    "EKFEstimator",
    "UKFEstimator",
    "ParticleFilterEstimator",
    "TorchKFKFEstimator",
    "TorchKFEKFEstimator",
    "TorchKFUKFEstimator",
    "TorchKFPFEstimator",
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
