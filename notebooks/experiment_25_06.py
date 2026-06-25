# %%
# Cell 1: Imports, seed setup, configuration
# ============================================================
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_levels import LinearBenchmark, PendulumBenchmark
from datasets.dataset import load_split
from estimators.classical.kf import KalmanFilterEstimator
from estimators.classical.ekf import EKFEstimator
from estimators.classical.ukf import UKFEstimator
from estimators.neural.kalmannet import KalmanNetEstimator
from metrics.rmse import compute_rmse
from metrics.latency import latency_ms_per_step
from utils.seeds import set_global_seed
from visualization.rmse import plot_rmse_comparison
from visualization.runtime import plot_runtime_comparison

RANDOM_SEED = 42
set_global_seed(RANDOM_SEED)

DATA_ROOT = Path("./data")
FIGURES_ROOT = Path("./figures")

# KalmanNet trains on GPU when available; benchmark *inference* below is
# forced onto CPU for every estimator so KF/EKF/UKF/KalmanNet are compared
# under identical conditions.
KALMANNET_EPOCHS = 1  # quick default; raise for a real training run
KALMANNET_HIDDEN_SIZE = 64

# Difficulty knobs (configurable per LLD: noise/length/initial-state are
# exposed by LinearBenchmark/PendulumBenchmark, not hardcoded).
BENCHMARK_CONFIGS = {
    "linear": dict(
        trajectory_length=150,
        num_trajectories=500,
        random_seed=RANDOM_SEED,
        process_noise_std=0.02,
        observation_noise_std=0.2,
        initial_state_std=2.0,
    ),
    "pendulum": dict(
        trajectory_length=150,
        num_trajectories=500,
        random_seed=RANDOM_SEED,
        process_noise_std=0.002,
        observation_noise_std=0.02,
        initial_angle_range=np.pi / 3,
    ),
}

print("Cell 1 complete.")

# %%
# Cell 2: Dataset generation (LinearBenchmark, PendulumBenchmark)
# ============================================================
benchmarks = {
    "linear": LinearBenchmark(**BENCHMARK_CONFIGS["linear"]),
    "pendulum": PendulumBenchmark(**BENCHMARK_CONFIGS["pendulum"]),
}

for name, benchmark in benchmarks.items():
    output_dir = DATA_ROOT / name
    if not (output_dir / "test.h5").exists():
        print(f"Generating dataset for '{name}' -> {output_dir}")
        benchmark.generate_dataset(output_dir)
    else:
        print(f"Dataset for '{name}' already exists at {output_dir}, skipping generation.")

print("Cell 2 complete.")

# %%
# Cell 3: Dataset loading
# ============================================================
datasets = {}
for name in benchmarks:
    split_dir = DATA_ROOT / name
    datasets[name] = {
        "train": load_split(split_dir, "train"),
        "val": load_split(split_dir, "val"),
        "test": load_split(split_dir, "test"),
    }

print("Cell 3 complete. Loaded splits:", {k: list(v.keys()) for k, v in datasets.items()})

# %%
# Cell 4: Estimator creation (KF, EKF, UKF, KalmanNet)
# ============================================================


def build_estimators(benchmark) -> dict:
    filter_model = benchmark.get_filter_model()
    return {
        "KF": KalmanFilterEstimator(filter_model, use_numba=True),
        "EKF": EKFEstimator(filter_model),
        "UKF": UKFEstimator(filter_model, use_numba=(benchmark.name == "linear")),
        "KalmanNet": KalmanNetEstimator(
            filter_model,
            hidden_size=KALMANNET_HIDDEN_SIZE,
            num_epochs=KALMANNET_EPOCHS,
            random_seed=RANDOM_SEED,
        ),
    }


estimators = {name: build_estimators(benchmark) for name, benchmark in benchmarks.items()}

print("Cell 4 complete. Estimators:", {k: list(v.keys()) for k, v in estimators.items()})

# %%
# Cell 5: Training (KalmanNet only)
# ============================================================
for benchmark_name, estimator_set in estimators.items():
    train_ds = datasets[benchmark_name]["train"]
    val_ds = datasets[benchmark_name]["val"]
    print(f"Training KalmanNet on '{benchmark_name}' ({KALMANNET_EPOCHS} epoch(s))...")
    estimator_set["KalmanNet"].fit(train_ds, val_ds)

print("Cell 5 complete.")

# %%
# Cell 6: CPU-only evaluation (all estimators)
# ============================================================
import time

# Classical estimators are already CPU-only. KalmanNetEstimator.estimate()
# forces its network onto CPU regardless of training device, so every
# estimator below runs inference under identical CPU conditions.
raw_results = {}  # benchmark_name -> estimator_name -> dict(estimates, runtime_seconds)

for benchmark_name, estimator_set in estimators.items():
    test_ds = datasets[benchmark_name]["test"]
    raw_results[benchmark_name] = {}

    for estimator_name, estimator in estimator_set.items():
        if estimator_name != "KalmanNet":
            estimator.fit(None, None)  # no-op for classical estimators

        t0 = time.perf_counter()
        estimates = estimator.estimate(test_ds)
        runtime_seconds = time.perf_counter() - t0

        raw_results[benchmark_name][estimator_name] = {
            "estimates": np.asarray(estimates),
            "runtime_seconds": runtime_seconds,
        }
        print(f"[{benchmark_name}] {estimator_name}: {runtime_seconds:.4f}s total")

print("Cell 6 complete.")

# %%
# Cell 7: Metrics collection (RMSE, runtime, latency)
# ============================================================
metrics_records = []

for benchmark_name, per_estimator in raw_results.items():
    test_ds = datasets[benchmark_name]["test"]
    targets = np.asarray(test_ds.states)
    N, T, _ = targets.shape

    for estimator_name, result in per_estimator.items():
        rmse = compute_rmse(result["estimates"], targets)
        runtime_seconds = result["runtime_seconds"]
        latency_ms = latency_ms_per_step(runtime_seconds, N, T)

        metrics_records.append({
            "benchmark": benchmark_name,
            "estimator": estimator_name,
            "rmse": rmse,
            "runtime_seconds": runtime_seconds,
            "latency_ms_per_step": latency_ms,
        })

metrics_df = pd.DataFrame(metrics_records)
print("Cell 7 complete.")
print(metrics_df.to_string(index=False))

# %%
# Cell 8: Results table
# ============================================================
results_table = metrics_df.pivot(index="estimator", columns="benchmark",
                                  values=["rmse", "latency_ms_per_step"])
print("Results table (RMSE and latency by estimator x benchmark):")
print(results_table.to_string())

ranking = (
    metrics_df.groupby("estimator")[["rmse", "latency_ms_per_step"]]
    .mean()
    .sort_values("rmse")
)
print("\nFinal ranking (mean RMSE across benchmarks, ascending):")
print(ranking.to_string())

# %%
# Cell 9: Visualization (RMSE comparison, runtime comparison)
# ============================================================
for benchmark_name in benchmarks:
    subset = metrics_df[metrics_df["benchmark"] == benchmark_name]
    plot_rmse_comparison(
        estimator_names=subset["estimator"].tolist(),
        rmse_values=subset["rmse"].tolist(),
        title=f"RMSE Comparison — {benchmark_name}",
        output_path=FIGURES_ROOT / f"rmse_{benchmark_name}.png",
    )
    plot_runtime_comparison(
        estimator_names=subset["estimator"].tolist(),
        runtime_values=subset["latency_ms_per_step"].tolist(),
        title=f"Latency Comparison (ms/step) — {benchmark_name}",
        output_path=FIGURES_ROOT / f"latency_{benchmark_name}.png",
    )

print("Cell 9 complete. Figures written to", FIGURES_ROOT)
