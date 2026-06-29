from .base import BenchmarkLevel, BaseSimulator, FilterModel, NumbaDynamics, TorchDynamics
from .linear import LinearBenchmark
from .pendulum import PendulumBenchmark
from .lorenz import LorenzBenchmark, LorenzFEABenchmark
from .nonlinear import NonlinearBenchmark

BENCHMARK_LEVELS = {
    "linear": LinearBenchmark,
    "pendulum": PendulumBenchmark,
    "lorenz": LorenzBenchmark,
    "lorenz_fea": LorenzFEABenchmark,
    "nonlinear": NonlinearBenchmark,
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
    "BENCHMARK_LEVELS",
]
