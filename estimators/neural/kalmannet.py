from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from ..base import BaseEstimator
from ._neural_base import _atomic_torch_save

if TYPE_CHECKING:
    import torch
    from datasets.schema import TrajectoryDataset


class _KalmanGainGRU:
    """
    Generalized over an arbitrary (nx, ny) state/observation dimension pair 
    via the benchmark's `FilterModel.f` / `FilterModel.h` instead of a fixed 
    IMU kinematic prior.

    Predict step (every timestep): x_pred = f(x_prev)  -- supplied by the
    benchmark's process model, run outside the network.

    Update step: the GRU consumes the innovation (y_t - h(x_pred)) and the
    previous state correction (x_prev - x_pred_prev), and emits a Kalman
    gain K (nx x ny). The correction x_post = x_pred + K @ innovation.

    If `predict_log_var=True`, an auxiliary head also predicts per-dimension
    log-variance of the state estimate (uncertainty-aware variant). This
    flag is the only difference between KalmanNetEstimator and
    KalmanNetUncertaintyEstimator -- the architecture/weights are otherwise
    identical, so there is no duplicated network code between the two.
    """

    @staticmethod
    def build(nx: int, ny: int, hidden_size: int, predict_log_var: bool):
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                in_features = ny + nx  # innovation + previous correction
                # nn.GRU (single module) drives BOTH forward paths with one set
                # of weights: the per-step free-running/CPU path advances it one
                # timestep at a time via a length-1 sequence in step(), while the
                # Phase-1 teacher-forced training path runs the whole [B, T, *]
                # sequence through it in a single parallel call. Using one nn.GRU
                # (not a GRUCell plus a separate nn.GRU) guarantees the two
                # phases share identical recurrence weights -- the same network
                # is trained teacher-forced and then deployed free-running.
                self.gru = nn.GRU(
                    input_size=in_features,
                    hidden_size=hidden_size,
                    batch_first=True,
                )
                self.out_norm = nn.LayerNorm(hidden_size)
                self.fc_gain = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, nx * ny),
                )
                # Zero-init: an untrained network is a pure process-model
                # predictor (K=0), matching the reference script's behavior.
                nn.init.zeros_(self.fc_gain[-1].weight)
                nn.init.zeros_(self.fc_gain[-1].bias)

                self.predict_log_var = predict_log_var
                if predict_log_var:
                    self.fc_logvar = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                        nn.Linear(hidden_size, nx),
                    )
                    nn.init.zeros_(self.fc_logvar[-1].weight)
                    nn.init.zeros_(self.fc_logvar[-1].bias)

                self.nx = nx
                self.ny = ny
                self.hidden_size = hidden_size

            def step(self, innovation, dx_prev, h):
                """
                innovation: (B, ny), dx_prev: (B, nx), h: (B, hidden_size)
                Returns (K, out, h_next) where `out` is the post-out_norm GRU
                output (B, hidden_size) and h_next keeps the (B, hidden_size)
                shape callers expect.

                Drives the nn.GRU with a length-1 sequence (one timestep): the
                input is unsqueezed to (B, 1, in_features) and the hidden state to
                (1, B, hidden_size) -- the GRU's (num_layers, B, hidden) layout --
                then both layer/seq dims are squeezed back out so the per-step
                contract matches the GRUCell-era one.

                The auxiliary log-variance head is deliberately NOT applied here:
                it does not feed back into the recurrence, so the callers collect
                `out` per step and apply fc_logvar once on the stacked sequence.
                Only fc_gain (which advances x via K) runs inside the loop.
                """
                inp = torch.cat([innovation, dx_prev], dim=1)
                out_seq, h_next = self.gru(inp.unsqueeze(1), h.unsqueeze(0))
                out = self.out_norm(out_seq[:, 0])

                K_flat = self.fc_gain(out)
                K = K_flat.view(-1, self.nx, self.ny)

                return K, out, h_next.squeeze(0)

            def init_hidden(self, batch_size: int, device) -> "torch.Tensor":
                return torch.zeros(batch_size, self.hidden_size, device=device)

        return _Module()


def _torch_batch_step(torch_fn, x_batch: "torch.Tensor", t: float = 0.0) -> "torch.Tensor":
    """Apply a benchmark's BATCHED torch process model (FilterModel.torch.f/h)
    to a batch of state vectors [B, nx] with a single vectorized tensor op on
    the input's own device. No per-row Python loop, no NumPy round-trip -- this
    is what lets KalmanNet's predict step run fully on the GPU during training.
    The batched torch dynamics mirror the NumPy/numba f/h one-for-one."""
    return torch_fn(x_batch, t)


class KalmanNetEstimator(BaseEstimator):
    """
    KalmanNet-style estimator: recurrent (GRU) Kalman gain prediction driven
    by the innovation, composed with the benchmark's own process/observation
    model (via FilterModel). Adapted from `kalman_net_sim2real.py`'s
    architecture but generalized to arbitrary BenchmarkLevel state/obs
    dimensions instead of a fixed 6-dim IMU kinematic prior.

    Hardware-specific execution (deliberate, per the deployment model):
    - fit()/validation: BATCHED torch on the GPU when available. Two-phase
      curriculum (opt-in via `curriculum_epochs`):
        * Phase 1 (teacher-forced): every step's GRU inputs derive from the
          GROUND-TRUTH previous state, so x_pred/innovation/dx_prev for the whole
          trajectory are computable up front and the GRU runs in ONE parallel
          [B, T, *] call -- no sequential recurrence, the only real GPU speedup
          (`_run_sequence_teacher_forced`).
        * Phase 2 (free-running): the existing self-propagating recursion where
          each step depends on the previous CORRECTED state, so it is inherently
          sequential over T (`_run_sequence_vectorized`). This is the deployed
          objective; Phase 1 only warm-starts it.
      The predict step in both uses FilterModel.torch (batched torch f/h), so
      every timestep is a single on-device tensor op -- no per-row Python loop,
      no NumPy round-trip. With curriculum_epochs=0, Phase 1 is skipped and fit()
      is identical to the pre-curriculum free-running training.
    - estimate()/inference: STRICTLY SEQUENTIAL on the CPU -- one trajectory at
      a time, one timestep at a time, using the NumPy f/h on a single state
      vector (`_run_sequence_sequential_cpu`). This simulates microprocessor /
      embedded deployment and measures test-time latency under those conditions.
    """

    estimator_id = "kalmannet"
    _predict_log_var = False

    def __init__(
        self,
        filter_model,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        num_epochs: int = 1,
        curriculum_epochs: int = 0,
        batch_size: int = 32,
        device: Optional[str] = None,
        random_seed: int = 0,
        grad_clip_norm: float = 0.5,
        weight_decay: float = 0.0,
        scheduler: str = "plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
        min_lr: float = 1e-6,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
        compile_model: bool = False,
        verbose: bool = True,
    ) -> None:
        self._model = filter_model
        self._nx = filter_model.Q.shape[0]
        self._ny = filter_model.R.shape[0]
        self._hidden_size = hidden_size
        self._lr = learning_rate
        self._num_epochs = num_epochs
        # Phase-1 (teacher-forced, fully parallel over T) curriculum epochs.
        # 0 => curriculum disabled; fit() runs only the free-running Phase 2 and
        # behaves identically to before this option existed.
        self._curriculum_epochs = curriculum_epochs
        self._batch_size = batch_size
        self._device_name = device
        self._random_seed = random_seed
        self._grad_clip_norm = grad_clip_norm
        self._weight_decay = weight_decay
        self._scheduler = scheduler
        self._scheduler_factor = scheduler_factor
        self._scheduler_patience = scheduler_patience
        self._min_lr = min_lr
        self._early_stopping_patience = early_stopping_patience
        self._early_stopping_min_delta = early_stopping_min_delta
        # Opt-in torch.compile of the Phase-1 teacher-forced parallel forward
        # (the path where compile actually wins on a real GPU). Never applied to
        # the per-step / CPU inference path -- compile is a net loss there.
        self._compile = compile_model
        self._verbose = verbose
        self._network = None
        self._best_val_loss = float("inf")
        self._best_state_dict = None
        # Populated by fit(): per-epoch training diagnostics. The "phase" field
        # tags each row 1 (teacher-forced curriculum) or 2 (free-running) so the
        # two phases are independently inspectable from the flat lists. With
        # curriculum disabled every row is phase 2, matching the old history.
        self.history_ = {"train_loss": [], "val_loss": [], "lr": [], "epoch": [], "phase": []}
        self.best_epoch_ = None
        self.stopped_epoch_ = None

    @property
    def estimator_name(self) -> str:
        return self.estimator_id

    @property
    def estimator_type(self) -> str:
        return "neural"

    @property
    def best_val_loss(self) -> float:
        return self._best_val_loss

    def _build_scheduler(self, optimizer, num_epochs=None):
        import torch

        # T_max for the cosine schedule uses the CALLER's epoch budget so each
        # curriculum phase anneals over its own epoch count, not the global total.
        if num_epochs is None:
            num_epochs = self._num_epochs

        name = (self._scheduler or "none").lower()
        if name in ("none", "off"):
            return None
        if name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self._scheduler_factor,
                patience=self._scheduler_patience,
                min_lr=self._min_lr,
            )
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(num_epochs, 1), eta_min=self._min_lr
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(self._scheduler_patience, 1),
                gamma=self._scheduler_factor,
            )
        raise ValueError(f"Unknown scheduler '{self._scheduler}'")

    def _training_device(self):
        import torch
        if self._device_name is not None:
            return torch.device(self._device_name)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_network(self):
        if self._network is None:
            self._network = _KalmanGainGRU.build(
                self._nx, self._ny, self._hidden_size, self._predict_log_var
            )
        return self._network

    def _run_sequence_vectorized(self, network, observations, states, timestamps, device):
        """
        PHASE-2 (free-running) predict/update recursion over a batch of full
        trajectories, used for GPU training/validation. Every timestep is a
        single batched torch op on `device` (the batched FilterModel.torch.f/h
        plus the GRU); there is no per-row Python loop and no NumPy round-trip,
        so the whole forward pass stays on the GPU. The only loop is over the T
        timesteps, which is the GRU's intrinsic recurrence (each step depends on
        the previous corrected state) and cannot be removed for a sequential
        filter -- this is the irreducible sequential bottleneck that Phase-1
        teacher forcing sidesteps.

        `states` (ground truth) is accepted but UNUSED here so this shares the
        `forward_fn(network, obs, states, ts, device)` signature with
        `_run_sequence_teacher_forced`; free-running never looks at ground truth.

        observations: torch.Tensor [B, T, ny] on `device`.
        timestamps:   torch.Tensor [T] (used for time-varying f, e.g. nonlinear).
        Returns (estimates [B, T, nx], log_vars [B, T, nx] or None).
        """
        import torch

        if self._model.torch is None:
            raise ValueError(
                f"{self.estimator_name}.fit() needs FilterModel.torch (batched "
                "torch dynamics) for vectorized GPU training; this model provides "
                "none. Add a TorchDynamics to the level (see _torch_dynamics.py)."
            )
        torch_f = self._model.torch.f
        torch_h = self._model.torch.h

        B, T, _ = observations.shape
        x = torch.zeros(B, self._nx, device=device, dtype=observations.dtype)
        x_pred_prev = x.clone()
        h = network.init_hidden(B, device)

        predict_log_var = network.predict_log_var
        # The free-running path runs network.step EAGER on purpose: it is the
        # sequential bottleneck and torch.compile is a net loss here (per-step
        # launch overhead dominates and the train/val grad-mode flip thrashes the
        # recompile guard). Compile is reserved for the Phase-1 parallel forward
        # (`_run_sequence_teacher_forced`), where it actually wins on a real GPU.
        estimates, outs = [], []
        for t in range(T):
            t_val = float(timestamps[t])
            x_pred = _torch_batch_step(torch_f, x, t_val)
            y_pred = _torch_batch_step(torch_h, x_pred, t_val)

            innovation = observations[:, t, :] - y_pred
            dx_prev = x - x_pred_prev

            K, out, h = network.step(innovation, dx_prev, h)
            correction = torch.bmm(K, innovation.unsqueeze(2)).squeeze(2)
            x_post = x_pred + correction

            x_pred_prev = x_pred
            x = x_post

            estimates.append(x_post)
            if predict_log_var:
                outs.append(out)

        estimates_seq = torch.stack(estimates, dim=1)
        # fc_logvar is applied ONCE on the stacked [B, T, hidden] sequence rather
        # than T times inside the loop -- it never feeds back into the recurrence.
        if predict_log_var:
            out_seq = torch.stack(outs, dim=1)  # [B, T, hidden]
            log_var_seq = network.fc_logvar(out_seq)  # [B, T, nx]
        else:
            log_var_seq = None
        return estimates_seq, log_var_seq

    def _run_sequence_teacher_forced(self, network, observations, states, timestamps, device):
        """
        PHASE-1 (teacher-forced) forward pass, FULLY PARALLEL over T.

        Identical filter math to `_run_sequence_vectorized`, with one change:
        every step's GRU inputs derive from the GROUND-TRUTH previous state
        instead of the network's own previous correction. Because the inputs no
        longer depend on the network's output, x_pred / innovation / dx_prev for
        the entire trajectory are computable up front and the GRU recurrence runs
        in a SINGLE [B, T, *] call -- there is no per-timestep Python loop over
        the network, which is the only way to get a real GPU speedup (the
        free-running recurrence is an irreducible sequential bottleneck).

        Critical correctness note: the gain K and correction use the teacher
        x_pred (propagated from the ground-truth previous state), NOT a
        self-propagated state. That is the whole point of teacher forcing and
        what makes the pass parallel.

        The seed-at-t=0, x_pred_prev, dx_prev, and innovation definitions mirror
        the free-running ones in `_run_sequence_vectorized` exactly; the ONLY
        difference is teacher (ground-truth) previous state vs. self state.

        observations: torch.Tensor [B, T, ny] on `device`.
        states:       torch.Tensor [B, T, nx] on `device` (ground-truth states).
        timestamps:   torch.Tensor [T] (used for time-varying f/h).
        Returns (estimates [B, T, nx], log_vars [B, T, nx] or None), matching
        `_run_sequence_vectorized`'s return contract.
        """
        import torch

        if self._model.torch is None:
            raise ValueError(
                f"{self.estimator_name}.fit() needs FilterModel.torch (batched "
                "torch dynamics) for vectorized GPU training; this model provides "
                "none. Add a TorchDynamics to the level (see _torch_dynamics.py)."
            )
        torch_f = self._model.torch.f
        torch_h = self._model.torch.h

        B, T, _ = observations.shape
        predict_log_var = network.predict_log_var

        # Teacher "previous state" per step: ground-truth state at t-1, zeros at
        # t=0. This is the teacher analog of free-running `x` (the previous
        # corrected state, seeded at zeros). Shift states right by one along T.
        x_prev = torch.zeros_like(states)
        x_prev[:, 1:, :] = states[:, :-1, :]

        # x_pred[:, t] = f(x_prev[:, t], t) and y_pred[:, t] = h(x_pred[:, t], t).
        # f/h take a SCALAR t per step (the nonlinear level's f uses cos(1.2*t)),
        # so we cannot fold t into one [B*T, nx] call; build the stacked tensor
        # with a per-step comprehension. This touches only the cheap f/h, not the
        # GRU -- the GRU still runs as a single parallel call below. timestamps
        # is read on-host once (tolist) to avoid a GPU->CPU sync per step.
        ts = timestamps.tolist()
        x_pred = torch.stack(
            [_torch_batch_step(torch_f, x_prev[:, t, :], ts[t]) for t in range(T)], dim=1
        )  # [B, T, nx]
        y_pred = torch.stack(
            [_torch_batch_step(torch_h, x_pred[:, t, :], ts[t]) for t in range(T)], dim=1
        )  # [B, T, ny]

        # x_pred_prev[:, t] = x_pred[:, t-1], zeros at t=0 -- mirrors the
        # free-running `x_pred_prev = x_pred` carry (seeded at zeros).
        x_pred_prev = torch.zeros_like(x_pred)
        x_pred_prev[:, 1:, :] = x_pred[:, :-1, :]

        innovation = observations - y_pred            # [B, T, ny]
        dx_prev = x_prev - x_pred_prev                # [B, T, nx]

        # Single parallel GRU call over the whole trajectory: h0 is the same
        # zero seed init_hidden uses, with the (num_layers, B, hidden) layout.
        inp = torch.cat([innovation, dx_prev], dim=-1)  # [B, T, in_features]
        h0 = network.init_hidden(B, device).unsqueeze(0)  # [1, B, hidden]
        out_seq, _ = network.gru(inp, h0)               # [B, T, hidden]
        out_seq = network.out_norm(out_seq)

        K = network.fc_gain(out_seq).view(B, T, self._nx, self._ny)  # [B, T, nx, ny]
        # x_post[:, t] = x_pred[:, t] + K[:, t] @ innovation[:, t], batched over
        # both B and T via einsum -- the teacher x_pred, not a self-propagation.
        correction = torch.einsum("btij,btj->bti", K, innovation)
        estimates_seq = x_pred + correction             # [B, T, nx]

        if predict_log_var:
            log_var_seq = network.fc_logvar(out_seq)    # [B, T, nx]
        else:
            log_var_seq = None
        return estimates_seq, log_var_seq

    def _run_epoch(
        self,
        epoch,
        total_epochs,
        phase,
        network,
        optimizer,
        train_loader,
        val_loader,
        train_ts,
        val_ts,
        device,
        forward_fn,
        loss_fn,
    ):
        """Run one epoch (train pass + val pass) for either phase, log a row into
        the shared history_, and return (train_loss, val_loss, lr).

        forward_fn(network, obs_b, states_b, ts, device) -> (pred, log_var) is
        either the teacher-forced forward (Phase 1, needs ground-truth states) or
        `_run_sequence_vectorized` (Phase 2, ignores states). Both the train and
        val passes share this body; only `forward_fn` differs by phase. The
        best-checkpoint / scheduler / early-stop bookkeeping is owned by the
        caller (each phase keeps its own), so this method only runs the passes
        and reports the per-epoch losses.
        """
        import math
        import torch

        current_lr = optimizer.param_groups[0]["lr"]

        network.train()
        train_loss_total, train_batches = 0.0, 0
        for obs_b, states_b in train_loader:
            obs_b = obs_b.to(device)
            states_b = states_b.to(device)

            pred, log_var = forward_fn(network, obs_b, states_b, train_ts, device)
            loss = loss_fn(pred, log_var, states_b)

            # Skip update if loss is NaN or Inf to prevent weight corruption
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
            for obs_b, states_b in val_loader:
                obs_b = obs_b.to(device)
                states_b = states_b.to(device)
                pred, log_var = forward_fn(network, obs_b, states_b, val_ts, device)
                val_loss_total += loss_fn(pred, log_var, states_b).item()
                val_batches += 1
        val_loss = val_loss_total / max(val_batches, 1)

        self.history_["epoch"].append(epoch + 1)
        self.history_["train_loss"].append(train_loss)
        self.history_["val_loss"].append(val_loss)
        self.history_["lr"].append(current_lr)
        self.history_["phase"].append(phase)

        if self._verbose:
            print(
                f"[{self.estimator_name}] phase {phase} epoch {epoch + 1}/{total_epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={current_lr:.2e}"
            )

        return train_loss, val_loss, current_lr

    def fit(self, train_dataset: "TrajectoryDataset", val_dataset: "TrajectoryDataset") -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self._random_seed)
        device = self._training_device()
        network = self._ensure_network().to(device)

        # Opt-in torch.compile of the Phase-1 teacher-forced PARALLEL forward
        # (`_run_sequence_teacher_forced`) -- the only path where compile wins on
        # a real GPU, since it runs the whole [B, T, *] sequence in one graph. We
        # compile two variants -- one used under grad (train), one under no_grad
        # (val) -- and dispatch on torch.is_grad_enabled() in Phase 1. Splitting
        # by grad mode keeps each compiled graph's guard stable so it never
        # recompiles when train/val alternate (a single shared object would
        # recompile every epoch on the grad_mode guard and fall back to eager
        # after 8). The free-running per-step path is left EAGER on purpose
        # (compile is a net loss there). No-op on torch builds without compile,
        # and never set on the CPU inference path (fit() clears these before the
        # network is moved to CPU). Numerically identical -- compile only fuses
        # kernels, it does not change the computation.
        network._compiled_tf_train = None
        network._compiled_tf_eval = None
        if self._compile and hasattr(torch, "compile"):
            try:
                network._compiled_tf_train = torch.compile(self._run_sequence_teacher_forced)
                network._compiled_tf_eval = torch.compile(self._run_sequence_teacher_forced)
            except Exception:
                network._compiled_tf_train = None
                network._compiled_tf_eval = None

        train_obs = torch.as_tensor(np.asarray(train_dataset.observations), dtype=torch.float32)
        train_states = torch.as_tensor(np.asarray(train_dataset.states), dtype=torch.float32)
        val_obs = torch.as_tensor(np.asarray(val_dataset.observations), dtype=torch.float32)
        val_states = torch.as_tensor(np.asarray(val_dataset.states), dtype=torch.float32)

        # Timestamps are shared across all trajectories in a split; thread them
        # into the (possibly time-varying) batched torch f during training.
        train_ts = torch.as_tensor(np.asarray(train_dataset.timestamps), dtype=torch.float32)
        val_ts = torch.as_tensor(np.asarray(val_dataset.timestamps), dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_obs, train_states),
            batch_size=self._batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_obs, val_states),
            batch_size=self._batch_size,
            shuffle=False,
        )

        mse = nn.MSELoss()

        def _loss(pred, log_var, target):
            if self._predict_log_var and log_var is not None:
                var = torch.exp(log_var)
                return nn.functional.gaussian_nll_loss(pred, target, var, eps=1e-6)
            return mse(pred, target)

        # Phase-1 forward dispatcher: use the torch.compile'd teacher-forced
        # graph if fit() prepared one, picking the train/eval variant by grad
        # mode (keeps each compiled guard stable, no recompile when train/val
        # alternate); fall back to the eager method otherwise. Signature matches
        # _run_sequence_vectorized so both can serve as `forward_fn` in _run_epoch.
        def _teacher_forced_forward(net, obs_b, states_b, ts, dev):
            if torch.is_grad_enabled():
                fn = getattr(net, "_compiled_tf_train", None)
            else:
                fn = getattr(net, "_compiled_tf_eval", None)
            if fn is None:
                return self._run_sequence_teacher_forced(net, obs_b, states_b, ts, dev)
            return fn(net, obs_b, states_b, ts, dev)

        # --- per-phase runner ------------------------------------------------
        # Each phase trains independently: its own optimizer/scheduler (a fresh
        # schedule is cleaner for the fresh objective and lets cosine T_max track
        # the phase's own epoch budget), its own best checkpoint, and its own
        # early-stopping counter. Returns the best in-memory state_dict for the
        # phase (None if no finite val loss ever beat inf). best_epoch_/
        # stopped_epoch_/_best_val_loss are left reflecting whichever phase the
        # caller runs last (Phase 2 -- the deployed model).
        def _run_phase(phase, num_epochs, forward_fn):
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
                train_loss, val_loss, current_lr = self._run_epoch(
                    epoch, num_epochs, phase, network, optimizer,
                    train_loader, val_loader, train_ts, val_ts, device,
                    forward_fn, _loss,
                )

                improved = False
                # Track best checkpoint in memory (only when loss is finite)
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
                        print(
                            f"[{self.estimator_name}] phase {phase} early stopping at "
                            f"epoch {epoch + 1} (best val_loss="
                            f"{self._best_val_loss:.6f} @ epoch {self.best_epoch_})"
                        )
                    break

            return best_state_dict

        self._best_val_loss = float("inf")
        self._best_state_dict = None
        self.history_ = {"train_loss": [], "val_loss": [], "lr": [], "epoch": [], "phase": []}
        self.best_epoch_ = None
        self.stopped_epoch_ = None

        # Phase 1 (teacher-forced, parallel): warm-start only. Runs for
        # curriculum_epochs epochs with its own best checkpoint; at the end we
        # load Phase 1's best weights so Phase 2 starts from them. Skipped
        # entirely when curriculum_epochs == 0 -- then fit() is the pre-curriculum
        # free-running training, and history_ holds only phase-2 rows.
        if self._curriculum_epochs > 0:
            phase1_best = _run_phase(1, self._curriculum_epochs, _teacher_forced_forward)
            if phase1_best is not None:
                network.load_state_dict(phase1_best)

        # Phase 2 (free-running): the deployed objective. Starts from Phase 1's
        # best weights (or the init when curriculum is off), with a fresh best
        # checkpoint, fresh early-stopping counter, and its own history rows.
        # best_epoch_/stopped_epoch_/_best_val_loss reflect THIS phase.
        self._best_state_dict = _run_phase(2, self._num_epochs, self._run_sequence_vectorized)

        # Load Phase-2's best checkpoint if available; otherwise keep last state.
        if self._best_state_dict is not None:
            network.load_state_dict(self._best_state_dict)

        # Drop the compiled forwards so the CPU inference path uses eager code.
        network._compiled_tf_train = None
        network._compiled_tf_eval = None
        self._network = network.to("cpu")

    def _run_sequence_sequential_cpu(self, network, observations, timestamps):
        """
        STRICTLY SEQUENTIAL, CPU-ONLY inference recursion used by estimate().

        This deliberately simulates microprocessor / embedded deployment: the
        filter runs one trajectory at a time, one timestep at a time, on the
        CPU, using the benchmark's NumPy process model f/h (one state vector at
        a time, no batching). It is the opposite of the vectorized GPU training
        path on purpose -- test-time latency is measured under deployment-like
        conditions, not under GPU batch throughput.

        observations: torch.Tensor [N, T, ny] (on CPU).
        Returns (estimates [N, T, nx], log_vars [N, T, nx] or None) as CPU tensors.
        """
        import torch

        cpu = torch.device("cpu")
        N, T, _ = observations.shape
        f = self._model.f
        h = self._model.h

        all_estimates = torch.zeros(N, T, self._nx, dtype=observations.dtype, device=cpu)
        all_log_vars = (
            torch.zeros(N, T, self._nx, dtype=observations.dtype, device=cpu)
            if self._predict_log_var else None
        )

        for i in range(N):  # one trajectory at a time
            x = torch.zeros(1, self._nx, dtype=observations.dtype, device=cpu)
            x_pred_prev = x.clone()
            hidden = network.init_hidden(1, cpu)

            for t in range(T):  # one timestep at a time
                t_val = float(timestamps[t])
                # NumPy process model, single state vector (microprocessor-style).
                x_np = x.squeeze(0).numpy()
                x_pred_np = np.asarray(f(x_np, t_val), dtype=np.float64)
                y_pred_np = np.asarray(h(x_pred_np, t_val), dtype=np.float64)
                x_pred = torch.from_numpy(x_pred_np).to(observations.dtype).unsqueeze(0)
                y_pred = torch.from_numpy(y_pred_np).to(observations.dtype).unsqueeze(0)

                innovation = observations[i, t, :].unsqueeze(0) - y_pred
                dx_prev = x - x_pred_prev

                K, out, hidden = network.step(innovation, dx_prev, hidden)
                correction = torch.bmm(K, innovation.unsqueeze(2)).squeeze(2)
                x_post = x_pred + correction

                x_pred_prev = x_pred
                x = x_post

                all_estimates[i, t] = x_post.squeeze(0)
                # Embedded sim keeps per-step logvar; step() no longer computes it
                # (it now returns the post-norm GRU output), so apply fc_logvar
                # here on this single step -- numerically identical to before.
                if all_log_vars is not None and network.predict_log_var:
                    log_var = network.fc_logvar(out)
                    all_log_vars[i, t] = log_var.squeeze(0)

        return all_estimates, all_log_vars

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
            estimates, _ = self._run_sequence_sequential_cpu(network, observations, timestamps)

        return estimates.numpy()

    def save(self, path: Path) -> None:
        if self._network is None:
            raise RuntimeError("No trained network to save.")
        _atomic_torch_save(
            {
                "state_dict": self._network.state_dict(),
                "nx": self._nx,
                "ny": self._ny,
                "hidden_size": self._hidden_size,
                "predict_log_var": self._predict_log_var,
                "best_val_loss": self._best_val_loss,
                "estimator_name": self.estimator_name,
            },
            path,
        )

    def load_weights(self, path: Path) -> bool:
        """Rebuild the network and load a saved best-weights checkpoint into it on
        the CPU, marking the estimator as fit() (so estimate() works without a
        retrain). Returns False only when the file is absent; a present-but-corrupt
        checkpoint raises (fail-fast). Mirrors SequentialNeuralFilter.load_weights
        for the per-estimator skip/resume loop in the experiment notebook."""
        import torch
        path = Path(path)
        if not path.exists():
            return False
        payload = torch.load(path, map_location="cpu")
        # _ensure_network builds into self._network; reset first so a re-load on an
        # already-fit estimator rebuilds rather than reusing stale weights.
        self._network = None
        network = self._ensure_network()
        network.load_state_dict(payload["state_dict"])
        self._network = network.to("cpu")
        self._best_val_loss = float(payload.get("best_val_loss", self._best_val_loss))
        return True

    @classmethod
    def load(cls, path: Path) -> "KalmanNetEstimator":
        raise NotImplementedError(
            f"{cls.__name__}.load requires a FilterModel. "
            "Reconstruct the estimator from a BenchmarkLevel.get_filter_model(), "
            "then call load_weights(path) on it (or torch.load(path) and "
            "load_state_dict() on its network)."
        )


class KalmanNetUncertaintyEstimator(KalmanNetEstimator):
    """
    Uncertainty-aware KalmanNet variant (a.k.a. KalmanNetPlus): identical
    architecture to KalmanNetEstimator plus an auxiliary log-variance head,
    trained with a Gaussian NLL loss instead of plain MSE. No network code
    is duplicated -- this subclass only flips `_predict_log_var` and
    `estimator_id`.

    Not executed by the default benchmark notebook; available for opt-in use.
    """

    estimator_id = "kalmannet_uncertainty"
    _predict_log_var = True

    def estimate_with_uncertainty(self, dataset: "TrajectoryDataset"):
        """Returns (estimates [N,T,nx], variance [N,T,nx]).

        Like estimate(), runs strictly sequentially on the CPU (deployment-like).
        """
        import torch

        if self._network is None:
            raise RuntimeError(f"{self.estimator_name} must be fit() before estimate().")

        network = self._network.to("cpu")
        network.eval()
        observations = torch.as_tensor(
            np.asarray(dataset.observations), dtype=torch.float32, device=torch.device("cpu")
        )
        timestamps = np.asarray(dataset.timestamps)

        with torch.inference_mode():
            estimates, log_var = self._run_sequence_sequential_cpu(network, observations, timestamps)

        return estimates.numpy(), torch.exp(log_var).numpy()
