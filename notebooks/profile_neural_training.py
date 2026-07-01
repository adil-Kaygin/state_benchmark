# %% [markdown]
# # Profile neural-filter training slowness (Issue 14)
#
# A small, self-contained profiler that turns "training is too slow" into
# numbers, for each neural estimator on a **reduced Lorenz** config. It only
# MEASURES -- it imports and calls the existing `fit()`/`_loss`, changing no
# estimator or benchmark code.
#
# It answers, with hard numbers:
#   1. total fit() wall-clock + per-epoch, per estimator x device (CPU vs GPU)
#   2. forward / backward / optimizer-step split of one epoch
#   3. KalmanNet Phase-1 (teacher-forced) vs Phase-2 (free-running) time
#   4. batch-size sweep (launch amortization)
#   5. Neural-ODE n_substeps sweep (the RK4 inner-loop multiplier)
#   6. a torch.profiler table exposing the launch-bound / GPU-idle signature
#
# Runs on a reduced config so the whole sweep is minutes, not hours -- the
# RELATIVE costs and the launch pattern are what matter here, not converged RMSE.
# Re-run after Issues 9/11/12 land to see the wall-clock drop.

# %%
# On Colab, mount + clone + install exactly like the experiment notebooks.
# Locally (with the repo importable and torch installed) skip this cell.
#
# IMPORTANT: to profile the speedup changes, clone the BRANCH that has them
# (--branch below), not the default main -- main does not have Issues 9/11/12.
# Change BRANCH to whatever you want to compare (e.g. "main" for the baseline).
# BRANCH = "perf/neural-training-speedups-issues-9-11-12-14"
# from google.colab import drive
# drive.mount('/content/drive')
# !rm -rf ./state_benchmark/
# !git clone --branch {BRANCH} https://github.com/adil-Kaygin/state_benchmark.git
# !pip install ./state_benchmark

# %%
import time
import contextlib

import numpy as np
import torch

from benchmark_levels import LorenzBenchmark
from datasets.schema import TrajectoryDataset
from estimators.neural.kalmannet import KalmanNetEstimator
from estimators.neural.neural_ode import NeuralODEEstimator
from estimators.neural.pinn import PINNFilterEstimator
from estimators.neural.transformer import TransformerEstimator
from estimators.neural.mamba import MambaEstimator

RANDOM_SEED = 42

HAS_CUDA = torch.cuda.is_available()
DEVICES = (["cuda", "cpu"] if HAS_CUDA else ["cpu"])
print(f"torch {torch.__version__} | cuda available: {HAS_CUDA} | devices: {DEVICES}")

# Reduced Lorenz: small enough for a fast sweep, large enough that the launch-
# bound T-loop dominates (the phenomenon under study). Bump these up to approach
# the real config once the pattern is confirmed.
PROF_TRAJ = 128        # total trajectories (vs 2000 in the real config)
PROF_T = 200           # trajectory length (vs 400)
PROF_EPOCHS = 3        # epochs per fit (vs 100)
PROF_BATCH = 32        # matches the real Lorenz batch (the smallest in the suite)


def _sync(device):
    """Block until queued GPU work finishes so wall-clock timings are real
    (CUDA kernels are async; without this the timer measures dispatch, not
    compute). No-op on CPU."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


@contextlib.contextmanager
def timed(label, device, store=None):
    _sync(device)
    t0 = time.perf_counter()
    yield
    _sync(device)
    dt = time.perf_counter() - t0
    print(f"  {label}: {dt:.3f}s")
    if store is not None:
        store[label] = dt


print("Imports complete.")

# %%
# Build a small in-memory Lorenz train/val split (no HDF5 round-trip needed --
# we generate to a temp dir then wrap the arrays in TrajectoryDataset).
import tempfile
from pathlib import Path
from datasets.dataset import load_split

benchmark = LorenzBenchmark(
    trajectory_length=PROF_T, num_trajectories=PROF_TRAJ, random_seed=RANDOM_SEED,
)
_tmp = Path(tempfile.mkdtemp(prefix="prof_lorenz_"))
benchmark.generate_dataset(_tmp)
train_ds = load_split(_tmp, "train")
val_ds = load_split(_tmp, "val")
filter_model = benchmark.get_filter_model()

n_train = np.asarray(train_ds.states).shape[0]
print(f"Lorenz reduced: train={n_train} traj, T={PROF_T}, "
      f"batch={PROF_BATCH} -> ~{max(n_train // PROF_BATCH, 1)} batches/epoch")


# %%
# Estimator factories at the reduced config. Each takes a device so we can
# time the same fit on CPU and GPU. curriculum_epochs>0 so KalmanNet exercises
# both phases.
def make_estimators(device, batch_size=PROF_BATCH, n_substeps=8, phase2_device=None):
    common = dict(
        random_seed=RANDOM_SEED, num_epochs=PROF_EPOCHS,
        batch_size=batch_size, device=device, verbose=False,
    )
    return {
        "KalmanNet": KalmanNetEstimator(
            filter_model, hidden_size=64, curriculum_epochs=max(PROF_EPOCHS // 2, 1),
            phase2_device=phase2_device, **common,
        ),
        "NeuralODE": NeuralODEEstimator(
            filter_model, ode_hidden=128, n_substeps=n_substeps, **common,
        ),
        "PINN": PINNFilterEstimator(
            filter_model, hidden_size=64, lambda_dyn=0.5, lambda_meas=0.1, **common,
        ),
        "Transformer": TransformerEstimator(
            filter_model, d_model=128, n_layers=4, n_heads=8, **common,
        ),
        "Mamba": MambaEstimator(
            filter_model, d_model=128, d_state=32, n_layers=4, **common,
        ),
    }


# %% [markdown]
# ## Cell 2 -- coarse wall-clock table (estimator x device)
#
# The headline: which estimator is worst, and is CPU faster than GPU for the
# launch-bound sequential loops (Issue 12 Lever 1)?

# %%
rows = []
for device in DEVICES:
    print(f"\n=== device={device} ===")
    ests = make_estimators(device)
    for name, est in ests.items():
        store = {}
        try:
            with timed(f"{name} fit()", device, store):
                est.fit(train_ds, val_ds)
            total = store[f"{name} fit()"]
            rows.append((name, device, total, total / PROF_EPOCHS))
        except Exception as exc:  # keep going; record the failure
            print(f"  {name} FAILED: {exc}")
            rows.append((name, device, float("nan"), float("nan")))

print("\nestimator      device   total_s   s/epoch")
for name, device, total, per in sorted(rows, key=lambda r: (r[1], -(r[2] if r[2] == r[2] else -1))):
    print(f"{name:<13} {device:<6} {total:8.3f}  {per:8.3f}")


# %% [markdown]
# ## Cell 3 -- forward / backward / optimizer-step split (one epoch)
#
# Where inside a step does the time go? Uses the primary device (GPU if present).

# %%
prof_device = DEVICES[0]
print(f"forward/backward/step split on device={prof_device}\n")

for name, est in make_estimators(prof_device).items():
    # Reproduce one train epoch's inner work with manual timers around the three
    # phases, using the estimator's own _loss (so the numbers reflect real code).
    import torch as _t
    _t.manual_seed(RANDOM_SEED)
    dev = est._training_device() if hasattr(est, "_training_device") else _t.device(prof_device)

    # Build the network + one batch the same way fit() would.
    obs = _t.as_tensor(np.asarray(train_ds.observations)[:PROF_BATCH], dtype=_t.float32)
    sts = _t.as_tensor(np.asarray(train_ds.states)[:PROF_BATCH], dtype=_t.float32)
    ts = _t.as_tensor(np.asarray(train_ds.timestamps), dtype=_t.float32)

    print(f"{name}:")
    try:
        if name == "KalmanNet":
            net = est._ensure_network().to(dev)
            net._compiled_tf_train = None
            net._compiled_tf_eval = None
            obs_d, sts_d = obs.to(dev), sts.to(dev)
            opt = _t.optim.Adam(net.parameters(), lr=1e-3)
            store = {}
            with timed("forward", dev, store):
                pred, _lv = est._run_sequence_vectorized(net, obs_d, sts_d, ts, dev)
                loss = _t.nn.functional.mse_loss(pred, sts_d)
            with timed("backward", dev, store):
                loss.backward()
            with timed("opt.step", dev, store):
                opt.step()
        else:
            net = est._build_network().to(dev)
            obs_d, sts_d = obs.to(dev), sts.to(dev)
            opt = _t.optim.Adam(net.parameters(), lr=1e-3)
            store = {}
            with timed("forward", dev, store):
                loss = est._loss(net, obs_d, sts_d, ts, dev)
            with timed("backward", dev, store):
                loss.backward()
            with timed("opt.step", dev, store):
                opt.step()
    except Exception as exc:
        print(f"  split FAILED: {exc}")


# %% [markdown]
# ## Cell 4 -- KalmanNet Phase-1 vs Phase-2 s/epoch
#
# The one number this cell exists to produce: **P1 s/epoch vs P2 s/epoch**, the
# split that motivates running Phase 2 on CPU (Issue 12). Phase 1 is teacher-
# forced, fully parallel over T, and its weight-independent prefix is cached
# (Issue 9); Phase 2 is free-running and sequential. history_["phase"] already
# tags each epoch 1 or 2, but carries no timing -- so we wrap the two per-epoch
# entry points (`_run_epoch_phase1_cached`, `_run_epoch`) in a wall-clock timer
# to attribute time to the phase, changing NO estimator code (measurement only,
# the notebook's contract).
#
# Epoch bookkeeping (avoids the off-by-one the old cell hid): with
# curriculum_epochs = max(PROF_EPOCHS//2, 1), the curriculum runs that many
# Phase-1 epochs **plus** num_epochs=PROF_EPOCHS Phase-2 epochs -- the two are
# independent budgets, so total epochs = curriculum_epochs + num_epochs, NOT
# PROF_EPOCHS. We report s/epoch per phase, which is invariant to those counts.

# %%
_curriculum = max(PROF_EPOCHS // 2, 1)
kn = KalmanNetEstimator(
    filter_model, hidden_size=64, random_seed=RANDOM_SEED,
    num_epochs=PROF_EPOCHS, curriculum_epochs=_curriculum,
    batch_size=PROF_BATCH, device=DEVICES[0], phase2_device=None, verbose=True,
)

# Time each epoch by phase without touching estimator code: wrap the two
# per-epoch methods so each records its wall-clock into a per-phase list. Phase 1
# runs on the training device (DEVICES[0]); Phase 2 runs on phase2_device (CPU by
# default), so sync against the right device around each timer.
_phase_secs = {1: [], 2: []}
_p2_dev = kn._phase2_device()
_orig_p1 = kn._run_epoch_phase1_cached
_orig_p2 = kn._run_epoch


def _timed_p1(*a, **k):
    _sync(DEVICES[0]); t0 = time.perf_counter()
    out = _orig_p1(*a, **k)
    _sync(DEVICES[0]); _phase_secs[1].append(time.perf_counter() - t0)
    return out


def _timed_p2(*a, **k):
    _sync(_p2_dev); t0 = time.perf_counter()
    out = _orig_p2(*a, **k)
    _sync(_p2_dev); _phase_secs[2].append(time.perf_counter() - t0)
    return out


kn._run_epoch_phase1_cached = _timed_p1
kn._run_epoch = _timed_p2
try:
    with timed("KalmanNet fit (curriculum)", DEVICES[0]):
        kn.fit(train_ds, val_ds)
finally:
    kn._run_epoch_phase1_cached = _orig_p1
    kn._run_epoch = _orig_p2

n_p1, n_p2 = len(_phase_secs[1]), len(_phase_secs[2])
p1_total, p2_total = sum(_phase_secs[1]), sum(_phase_secs[2])
p1_per = p1_total / max(n_p1, 1)
p2_per = p2_total / max(n_p2, 1)
print(f"\nphase                                  epochs   total_s   s/epoch")
print(f"Phase 1 (teacher-forced, {DEVICES[0]:<4} cached) {n_p1:>6}  {p1_total:8.3f}  {p1_per:8.3f}")
print(f"Phase 2 (free-running, {str(_p2_dev.type):<4})        {n_p2:>6}  {p2_total:8.3f}  {p2_per:8.3f}")
if p2_per > 0 and p1_per > 0:
    print(f"\nPhase 2 is {p2_per / p1_per:.1f}x the per-epoch cost of Phase 1 "
          f"-- the sequential free-running loop, which is why it runs on CPU "
          f"(phase2_device=None => {_p2_dev.type}, Issue 12).")


# %% [markdown]
# ## Cell 5 -- batch-size sweep (launch amortization)
#
# Bigger batches amortize per-step launch overhead. WATCH the update count:
# on the reduced split a large batch => very few optimizer steps/epoch.

# %%
print(f"batch-size sweep on device={DEVICES[0]} (KalmanNet)\n")
for bs in [32, 64, 128]:
    est = KalmanNetEstimator(
        filter_model, hidden_size=64, random_seed=RANDOM_SEED,
        num_epochs=PROF_EPOCHS, curriculum_epochs=0,
        batch_size=bs, device=DEVICES[0], phase2_device=DEVICES[0], verbose=False,
    )
    store = {}
    with timed(f"batch={bs} fit()", DEVICES[0], store):
        est.fit(train_ds, val_ds)
    print(f"  batch={bs}: {store[f'batch={bs} fit()'] / PROF_EPOCHS:.3f}s/epoch, "
          f"~{max(n_train // bs, 1)} batches/epoch")


# %% [markdown]
# ## Cell 6 -- Neural-ODE n_substeps sweep (RK4 inner-loop multiplier)
#
# The forward runs T * n_substeps * 4 drift-MLP evals. Expect ~linear growth in
# n_substeps -- the reason Neural-ODE is the slowest (Issue 11).

# %%
print(f"n_substeps sweep on device={DEVICES[0]} (NeuralODE)\n")
for ns in [2, 4, 8]:
    est = NeuralODEEstimator(
        filter_model, ode_hidden=128, n_substeps=ns, random_seed=RANDOM_SEED,
        num_epochs=PROF_EPOCHS, batch_size=PROF_BATCH, device=DEVICES[0], verbose=False,
    )
    store = {}
    with timed(f"n_substeps={ns} fit()", DEVICES[0], store):
        est.fit(train_ds, val_ds)
    print(f"  n_substeps={ns}: {store[f'n_substeps={ns} fit()'] / PROF_EPOCHS:.3f}s/epoch "
          f"(= T*{ns}*4 = {PROF_T * ns * 4} drift evals/traj)")


# %% [markdown]
# ## Cell 7 -- torch.profiler: the launch-bound / GPU-idle signature
#
# The definitive evidence: a huge `cudaLaunchKernel` / op count with tiny
# per-kernel CUDA time means the loop is launch-bound and the GPU sits idle. On
# CPU-only this still surfaces the op count. One forward+backward is enough.

# %%
from torch.profiler import profile, ProfilerActivity

prof_est = NeuralODEEstimator(
    filter_model, ode_hidden=128, n_substeps=8, random_seed=RANDOM_SEED,
    num_epochs=1, batch_size=PROF_BATCH, device=DEVICES[0], verbose=False,
)
_dev = torch.device(DEVICES[0])
_net = prof_est._build_network().to(_dev)
_obs = torch.as_tensor(np.asarray(train_ds.observations)[:PROF_BATCH], dtype=torch.float32).to(_dev)
_sts = torch.as_tensor(np.asarray(train_ds.states)[:PROF_BATCH], dtype=torch.float32).to(_dev)
_ts = torch.as_tensor(np.asarray(train_ds.timestamps), dtype=torch.float32)

activities = [ProfilerActivity.CPU]
if _dev.type == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=False) as prof:
    loss = prof_est._loss(_net, _obs, _sts, _ts, _dev)
    loss.backward()
    _sync(_dev)

sort_key = "cuda_time_total" if _dev.type == "cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_key, row_limit=20))
print("\nLook for: many small ops (aten::* / cudaLaunchKernel), tiny per-call time")
print("=> launch-bound. On GPU, compare total CUDA time to wall-clock: the gap is idle.")

print("\nProfiling notebook complete.")
