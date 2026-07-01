from .base import BenchmarkLevel, BaseSimulator, FilterModel, NumbaDynamics, TorchDynamics
from .linear import LinearBenchmark
from .pendulum import PendulumBenchmark
from .lorenz import LorenzBenchmark, LorenzFEABenchmark
from .nonlinear import NonlinearBenchmark
from .vehicle_tracking import VehicleTrackingBenchmark

BENCHMARK_LEVELS = {
    "linear": LinearBenchmark,
    "pendulum": PendulumBenchmark,
    "lorenz": LorenzBenchmark,
    "lorenz_fea": LorenzFEABenchmark,
    "nonlinear": NonlinearBenchmark,
    "vehicle_tracking": VehicleTrackingBenchmark,
}

__all__ = [
    "BenchmarkLevel",
    "BaseSimulator",
    "FilterModel",
    "NumbaDynamics",
    "TorchDynamics",
    "LinearBenchmark",
    "PendulumBenchmark",
    "LorenzBenchmark",
    "LorenzFEABenchmark",
    "NonlinearBenchmark",
    "VehicleTrackingBenchmark",
    "BENCHMARK_LEVELS",
]
