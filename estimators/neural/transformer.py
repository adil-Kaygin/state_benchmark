from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from ._neural_base import (
    SequentialNeuralFilter,
    dt_array,
    angular_obs_indices,
    wrap_innovation_torch,
    wrap_innovation_numpy,
)

if TYPE_CHECKING:
    import torch


class _CausalTransformer:
    """A causal (decoder-style) Transformer mapping per-step input features to a
    state estimate. Built from plain PyTorch (nn.TransformerEncoder + a causal
    mask) -- no third-party dependency.

    Per-step input features (Issue 3):
      innovation form (use_innovation_features=True):
        [y_t, innovation_t, x_pred_t, dt_t]   width = ny + ny + nx + 1
        where x_pred_t = f(x_prev_{t-1}) and innovation_t = y_t - h(x_pred_t).
      black-box form (use_innovation_features=False):
        [y_t, dt_t]                           width = ny + 1

    The causal mask makes x_hat_t depend only on y_{<=t} (a filter, not a
    smoother). Sinusoidal positional encoding is added over the time axis.
    head(z_t) gives either x_hat_t directly or, with residual_head, the
    correction x_hat_t = x_pred_t + head(z_t).
    """

    @staticmethod
    def build(nx: int, ny: int, in_features: int, d_model: int, n_heads: int,
              n_layers: int, dim_feedforward: int, dropout: float, max_len: int,
              residual_head: bool):
        import torch
        import torch.nn as nn

        class _PositionalEncoding(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
                div = torch.exp(
                    torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(pos * div)
                pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
                self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

            def forward(self, x):  # x: [B, T, d_model]
                T = x.shape[1]
                if T > self.pe.shape[1]:
                    raise ValueError(
                        f"TransformerEstimator: sequence length {T} exceeds max_len "
                        f"{self.pe.shape[1]}; increase max_len."
                    )
                return x + self.pe[:, :T, :]

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_proj = nn.Linear(in_features, d_model)
                self.pos = _PositionalEncoding()
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
                self.head = nn.Linear(d_model, nx)
                if residual_head:
                    nn.init.zeros_(self.head.weight)
                    nn.init.zeros_(self.head.bias)
                self.nx = nx
                self.ny = ny
                self.residual_head = residual_head

            def forward(self, feats, x_pred):
                """feats: [B, T, in_features], x_pred: [B, T, nx] (the model-based
                prediction used by the residual head). Returns x_hat [B, T, nx]
                via a single causal-masked pass over the whole sequence."""
                z = self.input_proj(feats)
                z = self.pos(z)
                T = z.shape[1]
                mask = nn.Transformer.generate_square_subsequent_mask(T).to(z.device)
                z = self.encoder(z, mask=mask, is_causal=True)
                out = self.head(z)
                return x_pred + out if self.residual_head else out

        return _Module()


class TransformerEstimator(SequentialNeuralFilter):
    """
    Causal (decoder-style) Transformer filter mapping the observation sequence
    y_{1:t} to the state estimate x_hat_t. Self-attention gives each estimate
    explicit long-range access to the past window of observations -- a different
    inductive bias from a recurrent filter. Causal masking enforces x_hat_t
    depends only on y_{<=t} (a filter, not a smoother).

    Process-model usage (Issue 3): with use_innovation_features=True the input
    features include x_pred = f(x_prev) and innovation = y - h(x_pred) (same
    conditioning KalmanNet uses); requires filter_model.torch.f/.h on the GPU
    training path (ValueError if None). With use_innovation_features=False the
    Transformer is a pure black-box on raw y and needs no filter_model.torch.

    Training regime (Issues 3, 13, 15 & 16): a causal Transformer is fully parallel
    over T at training time -- one masked forward pass computes x_hat_t for all t.
    Phase 1 (teacher-forced warm-start, `num_epochs` epochs) builds the input
    x_pred from the GROUND-TRUTH previous state -- the legitimate parallel
    teacher-forced INPUT construction. But at inference x_pred comes from the
    model's OWN previous estimate, so a teacher-forced-only model suffers exposure
    bias. `curriculum_epochs > 0` (innovation features only) ADDS Phase 2, a
    free-running fine-tune of that-many epochs on the deployed objective; total
    epochs = num_epochs + curriculum_epochs (additive, matching KalmanNet, Issue
    16). The two phases are structurally separate: each owns its optimizer,
    scheduler, best-checkpoint and patience, and fit() loads Phase 2's best weights
    -- so the deployed model is the fine-tuned one, never silently the warm-start.

    Phase 2 avoids the old O(T^2)-O(T^3) prefix-rerun (Issue 15) with two-pass
    scheduled sampling: pass one generates the self-fed features step by step WITH
    NO GRADIENT (exactly as the CPU inference path does), pass two runs a single
    differentiable parallel causal forward over those frozen features. Gradient
    stops at the fed-back state, so retention is one parallel pass, not T of them.
    curriculum_epochs=0 keeps the fast teacher-forced-only single phase; the
    raw-[y,dt] branch (use_innovation_features=False) feeds no state back and has
    no exposure bias, so the curriculum is a no-op there.

    Hardware split (Issue 0): fit() is a single parallel [B, T, *] pass on GPU.
    estimate() runs causally one step at a time on CPU -- for each trajectory,
    for each t, the model is run on the prefix y_{1:t} (innovation features from
    the model's own x_hat_{t-1} via NumPy f/h) and the last position's output is
    taken as x_hat_t. The prefix is recomputed each step (no KV cache): this is
    the honest deployment cost of an attention model and is what the latency
    metric reports.
    """

    estimator_id = "transformer"

    def __init__(
        self,
        filter_model,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 1024,
        use_innovation_features: bool = True,
        residual_head: bool = True,
        curriculum_epochs: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(filter_model, **kwargs)
        self._d_model = d_model
        self._n_heads = n_heads
        self._n_layers = n_layers
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._max_len = max_len
        self._use_innovation_features = use_innovation_features
        self._residual_head = residual_head
        # Issue 13/16: number of FREE-RUNNING fine-tune epochs (Phase 2) added
        # AFTER the num_epochs teacher-forced warm-start (Phase 1), to close the
        # exposure-bias gap. 0 => teacher-forced only (single phase, previous fast
        # behavior). Only meaningful with use_innovation_features=True: the
        # raw-[y,dt] branch feeds no state back, so it has no exposure bias and this
        # is a no-op there.
        self._curriculum_epochs = curriculum_epochs
        # Angular (bearing) observation indices whose innovation must be wrapped
        # to (-pi, pi] before it becomes an input feature (Issues 5/6). Empty for
        # every scalar-observation level -> wrapping is a no-op there.
        self._angular_idx = angular_obs_indices(filter_model, self._ny)
        if use_innovation_features:
            # [y, innovation, x_pred, dt]
            self._in_features = self._ny + self._ny + self._nx + 1
        else:
            # [y, dt]
            self._in_features = self._ny + 1
        # Fail fast (Issue 16) on a curriculum config where a phase would be
        # silently empty: with the fine-tune on, both the warm-start (num_epochs)
        # and the fine-tune (curriculum_epochs) budgets must be positive.
        if curriculum_epochs < 0:
            raise ValueError(f"{self.estimator_id}: curriculum_epochs must be >= 0.")
        if use_innovation_features and curriculum_epochs > 0 and self._num_epochs <= 0:
            raise ValueError(
                f"{self.estimator_id}: curriculum_epochs={curriculum_epochs} adds a "
                f"free-running fine-tune phase after the teacher-forced warm-start, "
                f"but num_epochs={self._num_epochs} leaves that warm-start empty. "
                f"Set num_epochs > 0 (Issue 16)."
            )

    def _build_network(self):
        return _CausalTransformer.build(
            self._nx, self._ny, self._in_features, self._d_model, self._n_heads,
            self._n_layers, self._dim_feedforward, self._dropout, self._max_len,
            self._residual_head,
        )

    def _save_hyperparams(self) -> dict:
        return {
            "d_model": self._d_model,
            "n_heads": self._n_heads,
            "n_layers": self._n_layers,
            "dim_feedforward": self._dim_feedforward,
            "dropout": self._dropout,
            "max_len": self._max_len,
            "use_innovation_features": self._use_innovation_features,
            "residual_head": self._residual_head,
            "curriculum_epochs": self._curriculum_epochs,
        }

    # --- Issue 13/15/16: free-running fine-tune to close the exposure-bias gap -

    def _phase_plan(self):
        """Teacher-forced warm-start (Phase 1, num_epochs), then -- only with the
        innovation-feature curriculum on -- a free-running fine-tune (Phase 2,
        curriculum_epochs), the deployed objective whose best weights fit() loads
        last (Issue 16). Off / black-box => single teacher-forced phase."""
        if self._use_innovation_features and self._curriculum_epochs > 0:
            return [(1, self._num_epochs, False), (2, self._curriculum_epochs, True)]
        return [(1, self._num_epochs, False)]

    def _phase_device(self, free_running):
        """Unlike Mamba's launch-bound step loop, the Transformer's free-running
        fine-tune is dominated by attention COMPUTE (the no-grad prefix generation
        plus the one parallel causal pass), which wants the training device. So the
        fine-tune stays on `device` by default; an explicit phase2_device still
        overrides it (escape hatch), and the teacher-forced phase is unaffected."""
        if free_running and self._phase2_device_name is None:
            return self._training_device()
        return super()._phase_device(free_running)

    def _loss(self, network, observations, states, timestamps, device, feats_cache=None):
        """Teacher-forced MSE during Phase 1; free-running MSE (own previous
        estimate fed back, matching deployment) during Phase 2."""
        import torch.nn.functional as F
        if self._free_running_phase:
            pred = self._forward_free_running(network, observations, timestamps, device)
            return F.mse_loss(pred, states)
        return super()._loss(
            network, observations, states, timestamps, device, feats_cache=feats_cache
        )

    def _forward_free_running(self, network, observations, timestamps, device):
        """Two-pass scheduled sampling (Issue 15), replacing the O(T^2)-O(T^3)
        differentiable prefix-rerun.

        Pass 1 (NO GRADIENT) generates the self-fed input features step by step,
        feeding the model's OWN previous estimate x_hat_{t-1} into f/h exactly as
        `_estimate_sequential_cpu` does at deployment, and re-running the causal
        forward on the growing prefix to get each x_hat_t. It fills a preallocated
        [B, T, *] buffer (no per-step re-stack of the whole prefix), so the only
        retained tensors are the frozen features -- no autograd graph is built.

        Pass 2 (DIFFERENTIABLE) runs the existing single parallel causal-masked
        forward over those frozen features and returns x_hat, so the loss is taken
        there. Gradient flows through the network weights but stops at the fed-back
        state (a constant here) -- the standard scheduled-sampling construction --
        so retention is exactly one parallel pass while the inputs match the
        deployment distribution. Returns x_hat [B, T, nx]."""
        import torch

        B, T, _ = observations.shape
        use_innov = self._use_innovation_features
        dt = torch.as_tensor(dt_array(timestamps), dtype=observations.dtype, device=device)
        if use_innov:
            self._require_torch_dynamics()
            torch_f = self._model.torch.f
            torch_h = self._model.torch.h
            ts = timestamps.tolist()
        else:
            torch_f = torch_h = None
            ts = [0.0] * T

        feats_buf = torch.zeros(B, T, self._in_features, device=device, dtype=observations.dtype)
        xpred_buf = torch.zeros(B, T, self._nx, device=device, dtype=observations.dtype)
        x_prev = torch.zeros(B, self._nx, device=device, dtype=observations.dtype)  # own x_hat_{t-1}

        with torch.no_grad():
            for t in range(T):
                row, x_pred = self._free_running_row(
                    observations[:, t, :], x_prev, dt[t], ts[t],
                    torch_f, torch_h, B, device, observations.dtype,
                )
                feats_buf[:, t, :] = row
                xpred_buf[:, t, :] = x_pred
                # Honest no-KV-cache deployment cost: rerun the prefix each step.
                x_hat_seq = network(feats_buf[:, : t + 1, :], xpred_buf[:, : t + 1, :])
                x_prev = x_hat_seq[:, -1, :]  # own output feeds the next step

        # Pass 2: one differentiable parallel pass over the frozen self-fed feats.
        return network(feats_buf, xpred_buf)

    # --- Issue 9: precompute the weight-independent teacher-forced features ---

    def _wants_feats_cache(self) -> bool:
        # Only innovation mode has a teacher-forced (state-dependent) feature to
        # cache; the raw-[y,dt] branch does not.
        return self._use_innovation_features

    def _precompute_feats(self, observations, states, timestamps, device):
        """Build the full per-step input `feats` [N, T, 2*ny+nx+1] ONCE per fit
        (Issue 9). Only meaningful in innovation mode -- the raw-[y,dt] branch has
        no teacher-forced (state-dependent) part, so return None there and let
        _forward_train build its trivial zero-x_pred features per batch. The
        teacher-forced x_pred/y_pred come from the shared helper; the result is a
        pure function of (states, timestamps, model) and is numerically identical
        to rebuilding it every epoch."""
        if not self._use_innovation_features:
            return None
        import torch
        from ._neural_base import precompute_teacher_forced
        self._require_torch_dynamics()
        torch_f = self._model.torch.f
        torch_h = self._model.torch.h
        N, T, _ = observations.shape
        dt = torch.as_tensor(dt_array(timestamps), dtype=observations.dtype, device=device)
        dt_col = dt.view(1, T, 1).expand(N, T, 1)
        x_pred, y_pred = precompute_teacher_forced(
            torch_f, torch_h, states, timestamps,
            time_invariant=self._model.torch.time_invariant,
        )
        innovation = wrap_innovation_torch(observations - y_pred, self._angular_idx)
        return torch.cat([observations, innovation, x_pred, dt_col], dim=-1)

    # --- GPU parallel forward (single masked pass over T) ----------------

    def _forward_train(self, network, observations, states, timestamps, device,
                       feats_cache=None):
        import torch

        B, T, _ = observations.shape
        dt = torch.as_tensor(dt_array(timestamps), dtype=observations.dtype, device=device)
        dt_col = dt.view(1, T, 1).expand(B, T, 1)

        if not self._use_innovation_features:
            feats = torch.cat([observations, dt_col], dim=-1)
            x_pred = torch.zeros(B, T, self._nx, device=device, dtype=observations.dtype)
            return network(feats, x_pred)

        if feats_cache is not None:
            # Issue 9: reuse the per-fit teacher-forced features; recover x_pred
            # from its known slice [y(ny), innovation(ny), x_pred(nx), dt(1)] so
            # the residual head sees the SAME x_pred as the uncached path.
            feats = feats_cache
            x_pred = feats[..., 2 * self._ny: 2 * self._ny + self._nx]
            return network(feats, x_pred)

        # Innovation features: x_prev_{t} = ground-truth state at t-1 (zeros at
        # t=0) -- the parallel teacher-forced INPUT construction. x_pred = f(x_prev),
        # innovation = y - h(x_pred). All built up front from GT, so the masked
        # forward stays a single parallel pass over T. (Fallback path when fit()
        # did not precompute a cache, e.g. a direct _forward_train call.)
        self._require_torch_dynamics()
        torch_f = self._model.torch.f
        torch_h = self._model.torch.h
        ts = timestamps.tolist()

        x_prev = torch.zeros_like(states)
        x_prev[:, 1:, :] = states[:, :-1, :]
        x_pred = torch.stack([torch_f(x_prev[:, t, :], ts[t]) for t in range(T)], dim=1)
        y_pred = torch.stack([torch_h(x_pred[:, t, :], ts[t]) for t in range(T)], dim=1)
        innovation = wrap_innovation_torch(observations - y_pred, self._angular_idx)
        feats = torch.cat([observations, innovation, x_pred, dt_col], dim=-1)
        return network(feats, x_pred)

    # --- CPU strictly-sequential inference (prefix recompute, no KV cache) -

    def _estimate_sequential_cpu(self, network, observations, timestamps):
        import torch

        N, T, _ = observations.shape
        f = self._model.f
        h = self._model.h
        ts = np.asarray(timestamps, dtype=np.float64)
        dt = dt_array(timestamps)
        obs_t = observations  # [N, T, ny] CPU tensor
        out = np.zeros((N, T, self._nx), dtype=np.float64)

        for i in range(N):  # one trajectory at a time
            y_i = obs_t[i]  # [T, ny]
            x_prev = np.zeros(self._nx, dtype=np.float64)  # model's own previous estimate
            # Per-step feature rows; we append one row at a time and rerun the
            # model on the growing prefix (honest no-KV-cache deployment cost).
            feat_rows = []
            xpred_rows = []
            for t in range(T):  # one timestep at a time
                t_val = float(ts[t])
                if self._use_innovation_features:
                    x_pred = np.asarray(f(x_prev, t_val), dtype=np.float64)
                    y_pred = np.asarray(h(x_pred, t_val), dtype=np.float64)
                    y_obs = y_i[t].numpy().astype(np.float64)
                    innovation = wrap_innovation_numpy(y_obs - y_pred, self._angular_idx)
                    row = np.concatenate([y_obs, innovation, x_pred, [dt[t]]])
                else:
                    x_pred = np.zeros(self._nx, dtype=np.float64)
                    row = np.concatenate([y_i[t].numpy().astype(np.float64), [dt[t]]])
                feat_rows.append(row)
                xpred_rows.append(x_pred)

                feats = torch.from_numpy(
                    np.asarray(feat_rows, dtype=np.float32)
                ).unsqueeze(0)  # [1, t+1, in_features]
                xpred = torch.from_numpy(
                    np.asarray(xpred_rows, dtype=np.float32)
                ).unsqueeze(0)  # [1, t+1, nx]
                x_hat_seq = network(feats, xpred)  # [1, t+1, nx]
                x_hat = x_hat_seq[0, -1].numpy().astype(np.float64)  # last position = x_hat_t
                out[i, t] = x_hat
                x_prev = x_hat
        return out
