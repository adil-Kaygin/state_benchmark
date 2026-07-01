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


def _selective_scan_parallel(deltaA, deltaB_x):
    """Parallel associative scan of the first-order linear recurrence

        h_t = deltaA_t * h_{t-1} + deltaB_x_t        (elementwise, h_0 = 0)

    over the time axis, returning the full sequence h_{1:T}. Shapes:
      deltaA, deltaB_x: [B, T, D, N]   ->   h: [B, T, D, N]

    Implementation: Heinsen (2023) log-space parallel scan. With a_t > 0,

        log h_t = a*_t  +  logcumsumexp_j<=t ( log b_j - a*_j )
        a*_t    = cumsum_{k<=t} log a_k       (log cumulative product of a)

    using complex logs to carry the sign of b_j. This is numerically stable
    (no division by vanishing cumulative products) and parallel over T via
    torch.cumsum / torch.logcumsumexp, which is the whole point of Mamba's
    selective scan -- training parallelizes over T with no teacher forcing.
    The math is the same recurrence the sequential inference path runs.
    """
    import torch

    a = deltaA.clamp_min(1e-20)
    log_a = torch.log(a)                                  # a_t > 0
    a_star = torch.cumsum(log_a, dim=1)                   # cumulative log-product

    b = deltaB_x.to(torch.complex64)
    log_b = torch.log(b)                                  # complex log carries sign
    # log h_t = a_star_t + logcumsumexp(log_b - a_star)
    log_h = a_star.to(torch.complex64) + torch.logcumsumexp(log_b - a_star.to(torch.complex64), dim=1)
    h = torch.exp(log_h).real
    return h


class _MambaBlock:
    """A single dependency-free Mamba (selective SSM) block in plain PyTorch.

    Per the Mamba architecture: an input projection expands the d_model channel
    to d_inner = expand * d_model with a gated branch, a depthwise causal conv
    over time, a SiLU activation, then the selective SSM with input-dependent
    Delta, B, C (A and D are learned per-channel), and an output projection back
    to d_model. The SSM recurrence

        h_t = exp(Delta_t * A) (.) h_{t-1} + (Delta_t * B_t) (.) x_t
        y_t = C_t (.) h_t  +  D (.) x_t

    runs via the parallel associative scan at training (`forward`) and as a
    plain O(T) recurrence at inference (`step`) -- numerically the same math.
    """

    @staticmethod
    def build(d_model: int, d_state: int, d_conv: int, expand: int):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        d_inner = expand * d_model
        dt_rank = max(d_model // 16, 1)

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.d_model = d_model
                self.d_inner = d_inner
                self.d_state = d_state
                self.d_conv = d_conv
                self.dt_rank = dt_rank

                self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
                self.conv1d = nn.Conv1d(
                    d_inner, d_inner, kernel_size=d_conv, groups=d_inner,
                    padding=d_conv - 1, bias=True,
                )
                # x_proj produces input-dependent Delta(low-rank), B, C.
                self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
                self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

                # A is parameterized as -exp(A_log) (negative, stable) per
                # (d_inner, d_state); D is a per-channel skip.
                A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
                self.A_log = nn.Parameter(torch.log(A))
                self.D = nn.Parameter(torch.ones(d_inner))
                self.out_proj = nn.Linear(d_inner, d_model, bias=False)

            def _ssm_params(self, x):
                """x: [B, T, d_inner] -> (delta [B,T,d_inner], A [d_inner,d_state],
                B [B,T,d_state], C [B,T,d_state])."""
                A = -torch.exp(self.A_log)  # [d_inner, d_state]
                x_dbl = self.x_proj(x)      # [B, T, dt_rank + 2*d_state]
                delta, B, C = torch.split(
                    x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
                )
                delta = F.softplus(self.dt_proj(delta))  # [B, T, d_inner]
                return delta, A, B, C

            def forward(self, u):
                """Training/GPU path: full sequence via the parallel scan.
                u: [B, T, d_model] -> [B, T, d_model]."""
                B_, T, _ = u.shape
                xz = self.in_proj(u)                       # [B, T, 2*d_inner]
                x, z = xz.chunk(2, dim=-1)                 # each [B, T, d_inner]

                # Depthwise causal conv over time (truncate the right pad).
                x_conv = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
                x = F.silu(x_conv)

                delta, A, Bm, Cm = self._ssm_params(x)
                # Discretize: deltaA = exp(delta . A), deltaB_x = (delta . B) . x.
                deltaA = torch.exp(delta.unsqueeze(-1) * A)                 # [B,T,d_inner,d_state]
                deltaB_x = (delta.unsqueeze(-1) * Bm.unsqueeze(2)) * x.unsqueeze(-1)
                h = _selective_scan_parallel(deltaA, deltaB_x)             # [B,T,d_inner,d_state]
                y = torch.einsum("btdn,btn->btd", h, Cm)                   # [B,T,d_inner]
                y = y + x * self.D
                y = y * F.silu(z)
                return self.out_proj(y)

            def step(self, u_t, conv_state, ssm_state):
                """Inference/CPU path: one timestep of the O(T) recurrence.
                u_t: [B, d_model]; conv_state: [B, d_inner, d_conv];
                ssm_state: [B, d_inner, d_state]. Returns
                (y_t [B, d_model], conv_state, ssm_state)."""
                xz = self.in_proj(u_t)                     # [B, 2*d_inner]
                x, z = xz.chunk(2, dim=-1)                 # each [B, d_inner]

                # Roll the conv FIFO and apply the depthwise conv at this step.
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = x
                x_conv = torch.sum(
                    conv_state * self.conv1d.weight.squeeze(1), dim=-1
                ) + self.conv1d.bias
                x = F.silu(x_conv)                         # [B, d_inner]

                delta, A, Bm, Cm = self._ssm_params(x.unsqueeze(1))
                delta = delta[:, 0]                        # [B, d_inner]
                Bm = Bm[:, 0]                              # [B, d_state]
                Cm = Cm[:, 0]                              # [B, d_state]
                deltaA = torch.exp(delta.unsqueeze(-1) * A)               # [B,d_inner,d_state]
                deltaB_x = (delta.unsqueeze(-1) * Bm.unsqueeze(1)) * x.unsqueeze(-1)
                ssm_state = deltaA * ssm_state + deltaB_x                 # [B,d_inner,d_state]
                y = torch.einsum("bdn,bn->bd", ssm_state, Cm) + x * self.D
                y = y * F.silu(z)
                return self.out_proj(y), conv_state, ssm_state

            def init_state(self, batch_size, device, dtype):
                conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype)
                ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype)
                return conv_state, ssm_state

        return _Module()


class _MambaNet:
    """Stacks the input embedding, N Mamba blocks (with residual + LayerNorm),
    and the state head over the per-step input features."""

    @staticmethod
    def build(nx: int, ny: int, in_features: int, d_model: int, d_state: int,
              d_conv: int, expand: int, n_layers: int, residual_head: bool):
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Linear(in_features, d_model)
                self.blocks = nn.ModuleList(
                    [_MambaBlock.build(d_model, d_state, d_conv, expand) for _ in range(n_layers)]
                )
                self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
                self.head = nn.Linear(d_model, nx)
                if residual_head:
                    nn.init.zeros_(self.head.weight)
                    nn.init.zeros_(self.head.bias)
                self.nx = nx
                self.ny = ny
                self.residual_head = residual_head

            def forward(self, feats, x_pred):
                """feats: [B, T, in_features], x_pred: [B, T, nx]. Parallel scan
                path. Returns x_hat [B, T, nx]."""
                h = self.embed(feats)
                for block, norm in zip(self.blocks, self.norms):
                    h = h + block(norm(h))
                out = self.head(h)
                return x_pred + out if self.residual_head else out

            def init_states(self, batch_size, device, dtype):
                return [b.init_state(batch_size, device, dtype) for b in self.blocks]

            def step(self, feat_t, x_pred_t, states):
                """One timestep of the recurrence (inference).
                feat_t: [B, in_features], x_pred_t: [B, nx],
                states: list of (conv_state, ssm_state) per block."""
                h = self.embed(feat_t)
                new_states = []
                for block, norm, (conv_s, ssm_s) in zip(self.blocks, self.norms, states):
                    y, conv_s, ssm_s = block.step(norm(h), conv_s, ssm_s)
                    h = h + y
                    new_states.append((conv_s, ssm_s))
                out = self.head(h)
                x_hat = x_pred_t + out if self.residual_head else out
                return x_hat, new_states

        return _Module()


class MambaEstimator(SequentialNeuralFilter):
    """
    Selective state-space (Mamba) filter -- a learned linear state-space
    recursion with input-dependent (selective) A, B, C, Delta, the natural
    generalization of the Kalman recursion. It trains in parallel via an
    associative selective scan and runs as a cheap constant-memory linear
    recurrence at inference, exactly the embedded-deployment story this benchmark
    measures (likely the cheapest learned filter in the suite at deployment).

    Process-model usage (Issue 4): same options as the Transformer -- innovation
    features [y, innovation=y-h(x_pred), x_pred, dt] (use_innovation_features=True,
    requires filter_model.torch.f/.h on GPU, ValueError if None) or a black-box
    on raw [y, dt] (use_innovation_features=False). residual_head toggles
    x_hat = x_pred + head vs x_hat = head.

    Dependency policy (Issue 4): the default is a from-scratch selective SSM in
    plain PyTorch -- a numerically-stable log-space PARALLEL associative scan for
    training and a plain O(T) recurrence for inference. There is NO mamba_ssm
    requirement on this machine. (The official mamba_ssm CUDA kernels would be a
    drop-in faster training scan with identical math, but cannot be installed
    here; the plain-torch scan is the implemented path.)

    Training regime (Issues 4 & 13): the selective scan is associative => parallel
    over T, so the default Phase-1 pass runs the whole [B, T, *] sequence through
    the scan in one teacher-forced pass (innovation features built from
    GROUND-TRUTH-prev). Because inference feeds the model's OWN prev estimate,
    teacher-forced-only training suffers exposure bias, so `curriculum_epochs > 0`
    adds a FREE-RUNNING fine-tune for the last that-many epochs (Issue 13): a
    sequential T-loop feeding the model's own x_hat_{t-1} into f/h exactly as
    `_estimate_sequential_cpu` does, matching train to deployment.
    curriculum_epochs=0 keeps the fast parallel-scan-only behavior; the
    raw-[y,dt] branch has no exposure bias so the curriculum is a no-op there.

    Hardware split (Issue 0): fit()/val batched on GPU via the parallel scan;
    estimate() strictly sequential on CPU as the O(T) constant-state recurrence,
    one trajectory / one timestep at a time with NumPy f/h for the innovation.
    """

    estimator_id = "mamba"

    def __init__(
        self,
        filter_model,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        use_innovation_features: bool = True,
        residual_head: bool = True,
        use_mamba_ssm_kernels: bool = True,
        curriculum_epochs: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(filter_model, **kwargs)
        self._d_model = d_model
        self._d_state = d_state
        self._d_conv = d_conv
        self._expand = expand
        self._n_layers = n_layers
        self._use_innovation_features = use_innovation_features
        self._residual_head = residual_head
        # Issue 13: trailing epochs trained FREE-RUNNING (own previous estimate)
        # to close the exposure-bias gap; 0 => parallel-scan teacher-forced only
        # (previous behavior). No-op with use_innovation_features=False.
        self._curriculum_epochs = curriculum_epochs
        self._free_running_phase = False
        # Angular (bearing) observation indices whose innovation must be wrapped
        # to (-pi, pi] before it becomes an input feature (Issues 5/6). Empty for
        # every scalar-observation level -> wrapping is a no-op there.
        self._angular_idx = angular_obs_indices(filter_model, self._ny)
        # Kept for API/config compatibility; mamba_ssm cannot be installed on
        # this machine, so the plain-torch parallel scan is always used. The
        # flag is honored only insofar as the fast path is attempted if present.
        self._use_mamba_ssm_kernels = use_mamba_ssm_kernels
        if use_innovation_features:
            self._in_features = self._ny + self._ny + self._nx + 1  # [y, innov, x_pred, dt]
        else:
            self._in_features = self._ny + 1                        # [y, dt]

    def _build_network(self):
        return _MambaNet.build(
            self._nx, self._ny, self._in_features, self._d_model, self._d_state,
            self._d_conv, self._expand, self._n_layers, self._residual_head,
        )

    def _save_hyperparams(self) -> dict:
        return {
            "d_model": self._d_model,
            "d_state": self._d_state,
            "d_conv": self._d_conv,
            "expand": self._expand,
            "n_layers": self._n_layers,
            "use_innovation_features": self._use_innovation_features,
            "residual_head": self._residual_head,
            "use_mamba_ssm_kernels": self._use_mamba_ssm_kernels,
            "curriculum_epochs": self._curriculum_epochs,
        }

    # --- Issue 13: free-running fine-tune to close the exposure-bias gap -----

    def _on_epoch_start(self, epoch: int) -> None:
        """Enter the free-running phase for the last `curriculum_epochs` epochs
        (only when innovation features feed state back)."""
        self._free_running_phase = (
            self._use_innovation_features
            and self._curriculum_epochs > 0
            and epoch >= self._num_epochs - self._curriculum_epochs
        )

    def _loss(self, network, observations, states, timestamps, device, feats_cache=None):
        """Teacher-forced (parallel scan) MSE during Phase 1; free-running MSE
        (own previous estimate fed back, matching deployment) during the
        curriculum tail."""
        import torch.nn.functional as F
        if self._free_running_phase:
            pred = self._forward_free_running(network, observations, timestamps, device)
            return F.mse_loss(pred, states)
        return super()._loss(
            network, observations, states, timestamps, device, feats_cache=feats_cache
        )

    # --- Issue 9: precompute the weight-independent teacher-forced features ---

    def _wants_feats_cache(self) -> bool:
        # Only innovation mode has a teacher-forced feature to cache.
        return self._use_innovation_features

    def _precompute_feats(self, observations, states, timestamps, device):
        """Build the full per-step input `feats` [N, T, 2*ny+nx+1] ONCE per fit
        (Issue 9), identical to the Transformer's cache. Only in innovation mode;
        the raw-[y,dt] branch has no teacher-forced part -> None. Numerically
        identical to rebuilding the features each epoch."""
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

    # --- GPU parallel-scan forward --------------------------------------

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
            # from its slice [y(ny), innovation(ny), x_pred(nx), dt(1)].
            feats = feats_cache
            x_pred = feats[..., 2 * self._ny: 2 * self._ny + self._nx]
            return network(feats, x_pred)

        # Innovation features from GROUND-TRUTH previous state (parallel
        # teacher-forced input), so the scan stays a single parallel pass over T.
        # (Fallback when fit() did not precompute a cache.)
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

    # --- CPU strictly-sequential O(T) recurrence ------------------------

    def _estimate_sequential_cpu(self, network, observations, timestamps):
        import torch

        N, T, _ = observations.shape
        f = self._model.f
        h = self._model.h
        ts = np.asarray(timestamps, dtype=np.float64)
        dt = dt_array(timestamps)
        cpu = torch.device("cpu")
        obs_t = observations
        out = np.zeros((N, T, self._nx), dtype=np.float64)

        for i in range(N):  # one trajectory at a time
            x_prev = np.zeros(self._nx, dtype=np.float64)  # model's own previous estimate
            states = network.init_states(1, cpu, torch.float32)
            for t in range(T):  # one timestep at a time, constant-size hidden state
                t_val = float(ts[t])
                y_t = obs_t[i, t].numpy().astype(np.float64)
                if self._use_innovation_features:
                    x_pred = np.asarray(f(x_prev, t_val), dtype=np.float64)
                    y_pred = np.asarray(h(x_pred, t_val), dtype=np.float64)
                    innovation = wrap_innovation_numpy(y_t - y_pred, self._angular_idx)
                    row = np.concatenate([y_t, innovation, x_pred, [dt[t]]])
                else:
                    x_pred = np.zeros(self._nx, dtype=np.float64)
                    row = np.concatenate([y_t, [dt[t]]])
                feat_t = torch.from_numpy(row.astype(np.float32)).unsqueeze(0)        # [1, in_features]
                xpred_t = torch.from_numpy(x_pred.astype(np.float32)).unsqueeze(0)    # [1, nx]
                x_hat_t, states = network.step(feat_t, xpred_t, states)
                x_hat = x_hat_t.squeeze(0).numpy().astype(np.float64)
                out[i, t] = x_hat
                x_prev = x_hat
        return out
