from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from ..base import BaseEstimator


def _atomic_torch_save(payload: dict, path: Path) -> None:
    """torch.save the payload to a temp file in the same directory, then
    os.replace() it onto the final name. torch.save is not atomic, so a Colab
    disconnect mid-write could otherwise leave a truncated checkpoint that loads
    as valid-but-wrong; os.replace is atomic on the same filesystem, so a reader
    sees either the old file or the fully-written new one, never a partial."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

if TYPE_CHECKING:
    import torch
    from datasets.schema import TrajectoryDataset


def precompute_teacher_forced(torch_f, torch_h, states, timestamps, time_invariant=False):
    """Build the weight-INDEPENDENT teacher-forced predictions for a whole batch
    of trajectories in one shot (Issue 9).

      x_prev[:, t] = states[:, t-1]   (zeros at t=0)      # GT shifted right
      x_pred[:, t] = f(x_prev[:, t], t)
      y_pred[:, t] = h(x_pred[:, t], t)

    Returns (x_pred [B,T,nx], y_pred [B,T,ny]) on states.device.

    This is a pure function of (states, timestamps, filter_model) -- none of
    which change during a fit() -- so its result can be computed ONCE and reused
    across every epoch/batch instead of being rebuilt each time. Caching it is
    numerically IDENTICAL to recomputing (same bits, no gradients flow through it:
    the inputs are ground-truth constants), it only removes redundant work.

    Time axis (Issue 10). f/h take a SCALAR t per step. When the level's f/h
    depend on t (nonlinear's cos(1.2*t)), t cannot be folded into one call, so we
    keep the per-step stack. But for a `time_invariant` level (linear / pendulum /
    lorenz -- t is ignored, only the baked-in dt matters), the whole T-loop
    collapses: flatten [B, T, nx] -> [B*T, nx], call f ONCE, reshape back (and
    likewise h). This is EXACTLY equal to the per-step path because every call is
    independent and t-invariant; it turns T kernel-launch sets into one. The
    representative t passed to the flat call (ts[0]) is ignored by construction.

    `time_invariant` MUST default to False so a new/forgotten level takes the safe
    per-step path -- flattening a genuinely t-dependent f would feed one t to all
    B*T rows and silently corrupt it. Read from TorchDynamics.time_invariant."""
    import torch
    B, T, nx = states.shape
    x_prev = torch.zeros_like(states)
    x_prev[:, 1:, :] = states[:, :-1, :]
    ts = timestamps.tolist()
    if time_invariant:
        # reshape (not view): x_prev[:, 1:] is contiguous but be defensive so a
        # non-contiguous input can never raise. t is ignored -> pass ts[0].
        t0 = ts[0]
        x_pred = torch_f(x_prev.reshape(B * T, nx), t0).reshape(B, T, -1)
        y_pred = torch_h(x_pred.reshape(B * T, x_pred.shape[-1]), t0).reshape(B, T, -1)
        return x_pred, y_pred
    x_pred = torch.stack([torch_f(x_prev[:, t, :], ts[t]) for t in range(T)], dim=1)
    y_pred = torch.stack([torch_h(x_pred[:, t, :], ts[t]) for t in range(T)], dim=1)
    return x_pred, y_pred


class SequentialNeuralFilter(BaseEstimator):
    """
    Shared scaffolding for the GPU-train / CPU-infer neural filters
    (Neural-ODE, PINN, Transformer, Mamba). It factors out everything the four
    estimators have in common with `KalmanNetEstimator` -- the per-PHASE runner
    (each phase owns its optimizer/scheduler/best-checkpoint/patience, Issue 16),
    the per-epoch train/val loop, best-checkpoint-in-memory by val loss, gradient
    clipping, optional LR scheduler, optional early stopping, NaN/Inf-loss skip,
    seeding, verbose printing, the phase-tagged `history_` dict, and the save()
    recipe -- so each concrete filter only supplies its network, its forward
    passes, and (if it has one) its `_phase_plan`.

    Single-phase filters (PINN, Neural-ODE, and a curriculum-off Transformer/Mamba)
    run one teacher-forced phase; Transformer/Mamba with the exposure-bias
    curriculum (Issue 13) append a free-running fine-tune phase whose best weights
    fit() loads last -- so the deployed model is the one trained on the deployed
    objective, never the teacher-forced warm-start (Issue 16).

    The hardware-split deployment contract (Issue 0) is enforced here by
    structure: subclasses implement
      _build_network()                                  -> nn.Module
      _forward_train(network, obs, states, ts, device)  -> x_hat [B, T, nx]  (GPU, batched, parallel)
      _loss(network, obs, states, ts, device)           -> scalar loss        (GPU; default: MSE of _forward_train)
      _estimate_sequential_cpu(network, obs, ts)        -> x_hat [N, T, nx]   (CPU, strictly sequential)
      _save_hyperparams()                               -> dict               (extra keys for save())

    `fit()` runs batched on the GPU when available; `estimate()` runs strictly
    sequentially on the CPU. Neither path looks at `dataset.states` inside
    `estimate()`.
    """

    estimator_id: str = "sequential_neural"

    def __init__(
        self,
        filter_model,
        learning_rate: float = 1e-3,
        num_epochs: int = 1,
        batch_size: int = 32,
        device: Optional[str] = None,
        phase2_device: Optional[str] = None,
        random_seed: int = 0,
        grad_clip_norm: float = 0.5,
        weight_decay: float = 0.0,
        scheduler: str = "plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
        min_lr: float = 1e-6,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self._model = filter_model
        self._nx = filter_model.Q.shape[0]
        self._ny = filter_model.R.shape[0]
        self._lr = learning_rate
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._device_name = device
        # Issue 12/15: the free-running fine-tune phase (Transformer/Mamba
        # curriculum) is a launch-bound sequential T-loop that a CPU usually runs
        # faster than a GPU. phase2_device=None defaults that phase to CPU; an
        # explicit value is honored exactly (escape hatch for large-batch GPU).
        # Unused by the single-phase filters (PINN / Neural-ODE), whose only phase
        # is the teacher-forced parallel forward on the training `device`.
        self._phase2_device_name = phase2_device
        self._random_seed = random_seed
        self._grad_clip_norm = grad_clip_norm
        self._weight_decay = weight_decay
        self._scheduler = scheduler
        self._scheduler_factor = scheduler_factor
        self._scheduler_patience = scheduler_patience
        self._min_lr = min_lr
        self._early_stopping_patience = early_stopping_patience
        self._early_stopping_min_delta = early_stopping_min_delta
        self._verbose = verbose
        self._network = None
        self._best_val_loss = float("inf")
        self._best_state_dict = None
        # "phase" tags each row 1 (teacher-forced warm-start) or 2 (free-running
        # fine-tune) so the two regimes are independently inspectable and the
        # regime jump in the val-loss plot is attributable (Issue 16). Every row is
        # phase 1 for the single-phase filters and for a curriculum-off fit.
        self.history_ = {"train_loss": [], "val_loss": [], "lr": [], "epoch": [], "phase": []}
        self.best_epoch_ = None
        self.stopped_epoch_ = None
        # Set per phase by fit() (Issue 16): True while the free-running fine-tune
        # phase is active, read by the Transformer/Mamba `_loss` to pick the
        # free-running forward. Never set for the single-phase filters.
        self._free_running_phase = False
        # True when fit() runs more than one phase, so the verbose log / early-stop
        # message is prefixed with the phase; keeps single-phase logs unchanged.
        self._multi_phase = False
        # Set by fit() (Issue 9): True when the subclass precomputed a per-fit
        # teacher-forced feature cache carried through the DataLoader.
        self._has_feats_cache = False

    # --- interface a subclass must satisfy -------------------------------

    @property
    def estimator_name(self) -> str:
        return self.estimator_id

    @property
    def estimator_type(self) -> str:
        return "neural"

    @property
    def best_val_loss(self) -> float:
        return self._best_val_loss

    def _build_network(self):
        raise NotImplementedError

    def _forward_train(self, network, observations, states, timestamps, device,
                       feats_cache=None):
        """Batched, parallel forward over the whole [B, T, *] sequence on GPU,
        returning x_hat [B, T, nx]. Used by the default `_loss`.

        `feats_cache` (Issue 9): an optional per-batch tensor of precomputed,
        weight-independent input features; subclasses that precompute use it to
        skip rebuilding the teacher-forced input every epoch. None => build as
        before (unchanged behavior)."""
        raise NotImplementedError

    def _forward_free_running(self, network, observations, timestamps, device):
        """FREE-RUNNING forward for the exposure-bias fine-tune (Issues 13/15).

        Builds the innovation feature at each step from the network's OWN previous
        estimate x_hat_{t-1} -- exactly the construction `_estimate_sequential_cpu`
        uses at deployment -- instead of the ground-truth previous state, so
        training matches inference. Returns x_hat [B, T, nx].

        Abstract here: only the innovation-feature filters have a free-running
        regime, and each realizes it at its architecture's O(T) cost rather than
        the old O(T^2)-O(T^3) prefix-rerun (Issue 15):
          * MambaEstimator overrides it with the constant-state `step` recurrence
            (one network step per timestep, linear graph retention).
          * TransformerEstimator overrides it with two-pass scheduled sampling: a
            no-grad sequential generation of the self-fed features, then one
            differentiable parallel causal pass over those frozen features (the
            gradient stops at the fed-back state, so retention is a single pass).
        """
        raise NotImplementedError(
            f"{self.estimator_name} has no free-running forward; only the "
            "innovation-feature filters (Transformer / Mamba) define one."
        )

    def _free_running_row(self, y_t, x_prev, dt_t, t_val, torch_f, torch_h,
                          B, device, dtype):
        """Build ONE free-running input row and its x_pred from the model's OWN
        previous estimate x_prev (Issues 13/15). Shared by the Transformer's
        no-grad generation loop and the Mamba `step` recurrence so both produce
        exactly the features the CPU inference path uses.

          innovation form: [y_t, wrap(y_t - h(f(x_prev))), f(x_prev), dt_t]
          black-box form:  [y_t, dt_t]   (x_pred is an unused zero vector)

        Returns (row [B, in_features], x_pred [B, nx]). Only meaningful for the
        innovation-feature filters; `self._use_innovation_features`,
        `self._angular_idx`, and `self._nx` must exist."""
        import torch
        dt_col = dt_t.view(1, 1).expand(B, 1)
        if getattr(self, "_use_innovation_features", False):
            x_pred = torch_f(x_prev, t_val)                       # [B, nx]
            y_pred = torch_h(x_pred, t_val)                       # [B, ny]
            innovation = wrap_innovation_torch(y_t - y_pred, self._angular_idx)
            row = torch.cat([y_t, innovation, x_pred, dt_col], dim=-1)
        else:
            x_pred = torch.zeros(B, self._nx, device=device, dtype=dtype)
            row = torch.cat([y_t, dt_col], dim=-1)
        return row, x_pred

    def _loss(self, network, observations, states, timestamps, device, feats_cache=None):
        """Default training loss: MSE of the batched forward vs ground truth.
        Override (e.g. PINN) to add physics residual terms.

        `feats_cache` (added by fit() when a subclass precomputes one) is an
        optional per-batch tensor carrying the weight-independent teacher-forced
        input features (Issue 9); the default forward ignores it, subclasses that
        precompute (Transformer/Mamba) read it in `_forward_train`."""
        import torch.nn.functional as F
        pred = self._forward_train(
            network, observations, states, timestamps, device, feats_cache=feats_cache
        )
        return F.mse_loss(pred, states)

    def _unpack_batch(self, batch, device):
        """Move a DataLoader batch to `device`. Handles both the plain
        (obs, states) batch and the (obs, states, feats_cache) batch produced when
        a subclass precomputes features (Issue 9). Returns (obs, states, feats or
        None)."""
        if len(batch) == 3:
            obs_b, states_b, feats_b = batch
            return obs_b.to(device), states_b.to(device), feats_b.to(device)
        obs_b, states_b = batch
        return obs_b.to(device), states_b.to(device), None

    def _wants_feats_cache(self) -> bool:
        """Whether `_precompute_feats` would return a cache for this fit (Issue 9).
        Lets fit() skip the whole-dataset device copy that `_precompute_feats`
        needs when there is nothing to cache. Default False; caching subclasses
        (Transformer/Mamba in innovation mode) override to their gating flag."""
        return False

    def _precompute_feats(self, observations, states, timestamps, device):
        """Optional hook (Issue 9): return an [N, T, F] tensor of weight-
        independent per-step input features to cache once per fit and slice per
        batch, or None (default) to disable caching. When non-None, fit() puts it
        in the TensorDataset alongside (obs, states) so the DataLoader shuffle
        permutes it in lockstep, and passes each batch's slice to `_loss` /
        `_forward_train` via `feats_cache`. Subclasses whose forward builds a
        weight-independent teacher-forced feature (Transformer, Mamba in
        innovation mode) override this; everyone else leaves it None. Only called
        when `_wants_feats_cache()` is True."""
        return None

    def _phase_plan(self):
        """Ordered list of (phase_id, num_epochs, free_running) phases fit() runs,
        each with its OWN optimizer, scheduler, best-checkpoint, and early-stopping
        counter (Issue 16). The LAST phase owns the weights fit() finally loads.

        Default: a single teacher-forced phase over `num_epochs` -- the behavior
        of PINN / Neural-ODE and of a curriculum-off Transformer/Mamba. The
        innovation-feature Transformer/Mamba override this to APPEND a free-running
        fine-tune phase (the deployed objective) so its checkpoint/scheduler/patience
        never share a comparison baseline with the incomparably-scaled teacher-forced
        val loss."""
        return [(1, self._num_epochs, False)]

    def _phase2_device(self):
        """Device for a free-running fine-tune phase (Issues 12/15). Explicit
        phase2_device honored exactly; None defaults to CPU -- the usual win for a
        launch-bound sequential T-loop, and the CPU-only deployment path."""
        import torch
        if self._phase2_device_name is not None:
            return torch.device(self._phase2_device_name)
        return torch.device("cpu")

    def _phase_device(self, free_running: bool):
        """Training device for a phase: the free-running fine-tune runs on
        phase2_device (CPU by default), the teacher-forced parallel forward on the
        main training `device`."""
        return self._phase2_device() if free_running else self._training_device()

    def _estimate_sequential_cpu(self, network, observations, timestamps):
        """Strictly sequential CPU inference: one trajectory / one timestep at a
        time with the NumPy filter_model.f/h. Returns x_hat [N, T, nx] (np)."""
        raise NotImplementedError

    def _save_hyperparams(self) -> dict:
        return {}

    # --- shared machinery (mirrors kalmannet.py) -------------------------

    def _require_torch_dynamics(self) -> None:
        if self._model.torch is None:
            raise ValueError(
                f"{self.estimator_name}.fit() needs FilterModel.torch (batched "
                "torch dynamics) for vectorized GPU training; this model provides "
                "none. Add a TorchDynamics to the level (see _torch_dynamics.py)."
            )

    def _training_device(self):
        """Resolve the training device. An explicit `device` kwarg is honored
        exactly; otherwise auto (cuda if present, else cpu).

        Issue 12 note: PINN and Neural-ODE are single-phase filters whose forward
        is an irreducible sequential T-loop (each step uses the net's own previous
        output). On small/narrow levels that loop is launch-bound and a CPU can be
        FASTER than a GPU (no per-step kernel-launch overhead). If a level trains
        slowly on GPU, pass `device="cpu"` for that (level, estimator) in the
        notebook config -- no code change needed, it flows straight through here.
        (KalmanNet expresses the same idea per-phase via `phase2_device`.)"""
        import torch
        if self._device_name is not None:
            return torch.device(self._device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_scheduler(self, optimizer, num_epochs=None):
        import torch
        if num_epochs is None:
            num_epochs = self._num_epochs
        name = (self._scheduler or "none").lower()
        if name in ("none", "off"):
            return None
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self._scheduler_factor,
                patience=self._scheduler_patience, min_lr=self._min_lr,
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(num_epochs, 1), eta_min=self._min_lr
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(self._scheduler_patience, 1),
                gamma=self._scheduler_factor,
            )
        raise ValueError(f"Unknown scheduler '{self._scheduler}'")

    def _run_epoch(self, epoch, total_epochs, phase, network, optimizer,
                   train_loader, val_loader, train_ts, val_ts, device):
        """Run one epoch (train pass + val pass) for a phase, log a phase-tagged
        history row, and return (train_loss, val_loss, lr). The `_loss` call reads
        `self._free_running_phase` (set per phase by fit()) to pick the forward.
        The best-checkpoint / scheduler / early-stop bookkeeping is owned by
        `_run_phase` (each phase keeps its own), so this only runs the passes."""
        import torch

        current_lr = optimizer.param_groups[0]["lr"]

        network.train()
        train_loss_total, train_batches = 0.0, 0
        for batch in train_loader:
            obs_b, states_b, feats_b = self._unpack_batch(batch, device)
            loss = self._loss(network, obs_b, states_b, train_ts, device, feats_cache=feats_b)
            # Skip update if loss is NaN/Inf to prevent weight corruption.
            if not math.isfinite(loss.item()):
                optimizer.zero_grad()
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), self._grad_clip_norm)
            optimizer.step()
            train_loss_total += loss.item()
            train_batches += 1
        train_loss = train_loss_total / max(train_batches, 1)

        network.eval()
        val_loss_total, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                obs_b, states_b, feats_b = self._unpack_batch(batch, device)
                val_loss_total += self._loss(
                    network, obs_b, states_b, val_ts, device, feats_cache=feats_b
                ).item()
                val_batches += 1
        val_loss = val_loss_total / max(val_batches, 1)

        self.history_["epoch"].append(epoch + 1)
        self.history_["train_loss"].append(train_loss)
        self.history_["val_loss"].append(val_loss)
        self.history_["lr"].append(current_lr)
        self.history_["phase"].append(phase)
        if self._verbose:
            label = f"phase {phase} " if self._multi_phase else ""
            print(
                f"[{self.estimator_name}] {label}epoch {epoch + 1}/{total_epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.2e}"
            )
        return train_loss, val_loss, current_lr

    def _run_phase(self, phase, num_epochs, network, epoch_fn):
        """Train one phase to completion with its OWN optimizer, scheduler, best
        checkpoint, and early-stopping counter (Issue 16). A fresh schedule fits
        the fresh objective (and lets cosine T_max track the phase's own epoch
        budget), and the best-value reset to inf means a free-running phase never
        compares against the incomparably-lower teacher-forced minimum. Returns the
        phase's best in-memory state_dict (None if no finite val loss beat inf).
        `_best_val_loss`/`best_epoch_`/`stopped_epoch_` are left reflecting this
        phase, so after the last phase they describe the deployed objective."""
        import torch

        optimizer = torch.optim.Adam(
            network.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        scheduler = self._build_scheduler(optimizer, num_epochs=num_epochs)

        self._best_val_loss = float("inf")
        best_state_dict = None
        self.best_epoch_ = None
        self.stopped_epoch_ = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            train_loss, val_loss, current_lr = epoch_fn(epoch, num_epochs, optimizer)

            improved = False
            if math.isfinite(val_loss):
                if val_loss < self._best_val_loss - self._early_stopping_min_delta:
                    self._best_val_loss = val_loss
                    best_state_dict = {
                        k: v.cpu().clone() for k, v in network.state_dict().items()
                    }
                    self.best_epoch_ = epoch + 1
                    improved = True
                if scheduler is not None:
                    if self._scheduler == "plateau":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

            epochs_no_improve = 0 if improved else epochs_no_improve + 1
            if (
                self._early_stopping_patience is not None
                and epochs_no_improve >= self._early_stopping_patience
            ):
                self.stopped_epoch_ = epoch + 1
                if self._verbose:
                    label = f"phase {phase} " if self._multi_phase else ""
                    print(
                        f"[{self.estimator_name}] {label}early stopping at epoch "
                        f"{epoch + 1} (best val_loss={self._best_val_loss:.6f} "
                        f"@ epoch {self.best_epoch_})"
                    )
                break

        return best_state_dict

    def fit(self, train_dataset: "TrajectoryDataset", val_dataset: "TrajectoryDataset") -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self._random_seed)
        train_device = self._training_device()
        network = self._build_network().to(train_device)
        self._network = network

        train_obs = torch.as_tensor(np.asarray(train_dataset.observations), dtype=torch.float32)
        train_states = torch.as_tensor(np.asarray(train_dataset.states), dtype=torch.float32)
        val_obs = torch.as_tensor(np.asarray(val_dataset.observations), dtype=torch.float32)
        val_states = torch.as_tensor(np.asarray(val_dataset.states), dtype=torch.float32)
        train_ts = torch.as_tensor(np.asarray(train_dataset.timestamps), dtype=torch.float32)
        val_ts = torch.as_tensor(np.asarray(val_dataset.timestamps), dtype=torch.float32)

        # Issue 9: if the subclass precomputes a weight-independent teacher-forced
        # feature, build it ONCE per split here (on the training device, where the
        # RK4 f/h cost actually lives) instead of every epoch inside _forward_train.
        # It is consumed only by teacher-forced phases; a free-running phase drops
        # it (Issue 15 minor adjunct: no more slicing/shipping an ignored cache).
        # Non-caching estimators skip the whole-dataset device copy entirely.
        if self._wants_feats_cache():
            train_feats = self._precompute_feats(
                train_obs.to(train_device), train_states.to(train_device), train_ts, train_device
            )
            val_feats = self._precompute_feats(
                val_obs.to(train_device), val_states.to(train_device), val_ts, train_device
            )
        else:
            train_feats = val_feats = None
        self._has_feats_cache = train_feats is not None
        train_feats_cpu = train_feats.cpu() if self._has_feats_cache else None
        val_feats_cpu = val_feats.cpu() if self._has_feats_cache else None

        def _make_loader(obs, states, feats, free_running, shuffle):
            # Teacher-forced phases carry the feats cache as a THIRD tensor so the
            # DataLoader shuffle permutes it in lockstep with (obs, states) and it
            # can never desync from its sample. Free-running phases build their own
            # input from the net's previous estimate, so they omit it.
            if (not free_running) and feats is not None:
                ds = TensorDataset(obs, states, feats)
            else:
                ds = TensorDataset(obs, states)
            return DataLoader(ds, batch_size=self._batch_size, shuffle=shuffle)

        self._best_val_loss = float("inf")
        self._best_state_dict = None
        self.history_ = {"train_loss": [], "val_loss": [], "lr": [], "epoch": [], "phase": []}
        self.best_epoch_ = None
        self.stopped_epoch_ = None

        # Each phase (Issue 16) owns its optimizer/scheduler/best/patience; the
        # LAST phase's best weights are the ones fit() loads. Between phases the
        # network is moved onto the next phase's device and seeded with the
        # previous phase's best, so a free-running fine-tune starts from the
        # teacher-forced warm-start (mirrors KalmanNet's per-phase runner).
        plan = self._phase_plan()
        self._multi_phase = len(plan) > 1
        best_state_dict = None
        for phase_id, num_epochs, free_running in plan:
            phase_device = self._phase_device(free_running)
            network.to(phase_device)
            self._free_running_phase = free_running
            train_loader = _make_loader(
                train_obs, train_states, train_feats_cpu, free_running, shuffle=True
            )
            val_loader = _make_loader(
                val_obs, val_states, val_feats_cpu, free_running, shuffle=False
            )

            # Freeze the per-phase loaders/id/device into the epoch closure; the
            # network object is shared across phases (moved, not rebuilt).
            def _epoch_fn(epoch, total, optimizer,
                          _tl=train_loader, _vl=val_loader, _pid=phase_id, _dev=phase_device):
                return self._run_epoch(
                    epoch, total, _pid, network, optimizer, _tl, _vl, train_ts, val_ts, _dev
                )

            best_state_dict = self._run_phase(phase_id, num_epochs, network, _epoch_fn)
            if best_state_dict is not None:
                network.load_state_dict(best_state_dict)

        # The final phase is the deployed objective; its best weights (or the last
        # state, if no finite val loss ever beat inf) are what estimate() runs.
        self._best_state_dict = best_state_dict
        self._network = network.to("cpu")

    def estimate(self, dataset: "TrajectoryDataset") -> np.ndarray:
        import torch
        if self._network is None:
            raise RuntimeError(f"{self.estimator_name} must be fit() before estimate().")
        # Enforce CPU-only inference (microprocessor deployment simulation).
        network = self._network.to("cpu")
        network.eval()
        observations = torch.as_tensor(
            np.asarray(dataset.observations), dtype=torch.float32, device=torch.device("cpu")
        )
        timestamps = np.asarray(dataset.timestamps)
        with torch.inference_mode():
            estimates = self._estimate_sequential_cpu(network, observations, timestamps)
        return np.asarray(estimates)

    def save(self, path: Path) -> None:
        import torch
        if self._network is None:
            raise RuntimeError("No trained network to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self._network.state_dict(),
            "nx": self._nx,
            "ny": self._ny,
            "best_val_loss": self._best_val_loss,
            "estimator_name": self.estimator_name,
        }
        payload.update(self._save_hyperparams())
        _atomic_torch_save(payload, path)

    def load_weights(self, path: Path) -> bool:
        """Rebuild the network and load a saved best-weights checkpoint into it on
        the CPU, marking the estimator as fit() (so estimate() works without a
        retrain). Returns False only when the file is absent; a present-but-corrupt
        checkpoint raises (fail-fast -- never silently retrain over a real defect).
        Pairs with the per-estimator skip/resume loop in the experiment notebook."""
        import torch
        path = Path(path)
        if not path.exists():
            return False
        payload = torch.load(path, map_location="cpu")
        network = self._build_network()
        network.load_state_dict(payload["state_dict"])
        self._network = network.to("cpu")
        self._best_val_loss = float(payload.get("best_val_loss", self._best_val_loss))
        return True

    @classmethod
    def load(cls, path: Path) -> "SequentialNeuralFilter":
        raise NotImplementedError(
            f"{cls.__name__}.load requires a FilterModel. "
            "Reconstruct the estimator from a BenchmarkLevel.get_filter_model() "
            "with the saved hyperparameters, then call load_weights(path) on it "
            "(or torch.load(path) and load_state_dict() on its network)."
        )


def angular_obs_indices(filter_model, ny: int) -> np.ndarray:
    """Integer indices of the angular (bearing) observation components from
    FilterModel.angular_obs_mask, empty when there are none (Issues 5/6). The
    neural filters use the innovation y - h(x_pred) as an INPUT feature; a
    bearing residual must be wrapped to (-pi, pi] there too, or the network trains
    on ~2*pi-wrong values near the branch cut. Returns np.int64 indices; an empty
    array => no wrapping (every current scalar-observation level)."""
    mask = getattr(filter_model, "angular_obs_mask", None)
    if mask is None:
        return np.empty(0, dtype=np.int64)
    mask = np.asarray(mask)
    if mask.shape != (ny,):
        raise ValueError(f"angular_obs_mask must have shape ({ny},); got {mask.shape}.")
    return np.nonzero(mask)[0]


def wrap_innovation_torch(innovation, angular_idx):
    """Wrap the angular components of a torch innovation tensor [..., ny] to
    (-pi, pi] via atan2(sin, cos). No-op when angular_idx is empty. Returns a new
    tensor; does not mutate the input."""
    if len(angular_idx) == 0:
        return innovation
    import torch
    out = innovation.clone()
    idx = torch.as_tensor(np.asarray(angular_idx), dtype=torch.long, device=innovation.device)
    ang = innovation.index_select(-1, idx)
    out.index_copy_(-1, idx, torch.atan2(torch.sin(ang), torch.cos(ang)))
    return out


def wrap_innovation_numpy(innovation: np.ndarray, angular_idx: np.ndarray) -> np.ndarray:
    """NumPy counterpart of wrap_innovation_torch for the sequential CPU inference
    path. Wraps innovation[..., angular_idx] to (-pi, pi]; no-op when empty."""
    if len(angular_idx) == 0:
        return innovation
    out = np.array(innovation, dtype=np.float64, copy=True)
    out[..., angular_idx] = np.arctan2(np.sin(out[..., angular_idx]), np.cos(out[..., angular_idx]))
    return out


def dt_array(timestamps: np.ndarray) -> np.ndarray:
    """Inter-sample intervals dt[t] = timestamps[t] - timestamps[t-1], with
    dt[0] set to dt[1] (no propagation happens before the first sample, but a
    finite seed value keeps any t=0 substep well-defined). Shared by the
    continuous-time filters."""
    ts = np.asarray(timestamps, dtype=np.float64)
    dt = np.empty_like(ts)
    if ts.shape[0] > 1:
        dt[1:] = np.diff(ts)
        dt[0] = dt[1]
    else:
        dt[0] = 1.0
    return dt
