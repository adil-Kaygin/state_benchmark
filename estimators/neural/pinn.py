from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from ._neural_base import SequentialNeuralFilter

if TYPE_CHECKING:
    import torch


class _PINNGRU:
    """GRU backbone for the physics-informed filter, structured like
    KalmanNet's predict/update recurrence but with the physics encoded in the
    LOSS rather than in the gain form.

    Per step the GRU consumes innovation features [innovation, x_pred] (the same
    conditioning KalmanNet uses: innovation = y_t - h(f(x_{t-1})), and the
    predicted state x_pred = f(x_{t-1})) and emits a state correction dx, giving
    x_hat_t = x_pred + dx. The recurrence is causal so it runs sequentially at
    inference; the only new thing versus KalmanNet is the training objective.

    One nn.GRU drives both forward paths with one weight set: the GPU path runs
    the whole [B, T, *] sequence step-by-step (parallel over B), and the CPU
    inference path advances it one length-1 timestep at a time via step().
    """

    @staticmethod
    def build(nx: int, ny: int, hidden_size: int):
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                in_features = ny + nx  # innovation + predicted state
                self.gru = nn.GRU(input_size=in_features, hidden_size=hidden_size, batch_first=True)
                self.out_norm = nn.LayerNorm(hidden_size)
                self.fc_out = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                    nn.Linear(hidden_size, nx),
                )
                # Zero-init the correction head: an untrained filter is a pure
                # process-model predictor (dx=0), matching KalmanNet's K=0 init.
                nn.init.zeros_(self.fc_out[-1].weight)
                nn.init.zeros_(self.fc_out[-1].bias)
                self.nx = nx
                self.ny = ny
                self.hidden_size = hidden_size

            def step(self, innovation, x_pred, h):
                """One timestep on a length-1 sequence.
                innovation: (B, ny), x_pred: (B, nx), h: (B, hidden).
                Returns (dx (B, nx), h_next (B, hidden))."""
                inp = torch.cat([innovation, x_pred], dim=1)
                out_seq, h_next = self.gru(inp.unsqueeze(1), h.unsqueeze(0))
                out = self.out_norm(out_seq[:, 0])
                dx = self.fc_out(out)
                return dx, h_next.squeeze(0)

            def init_hidden(self, batch_size: int, device):
                return torch.zeros(batch_size, self.hidden_size, device=device)

        return _Module()


class PINNFilterEstimator(SequentialNeuralFilter):
    """
    Physics-Informed Neural Network filter. A causal GRU maps innovation
    features to a state estimate; the training objective adds residual terms
    that force the estimate to obey the benchmark's KNOWN dynamics f/h (no model
    mismatch -- the same f/h EKF gets), not just state-MSE:

        r_data = x_hat_t - x_t                     (supervised state error)
        r_dyn  = x_hat_t - f(x_hat_{t-1}, t-1)     (process-model consistency)
        r_meas = y_t      - h(x_hat_t, t)          (observation consistency, no GT)

        loss = ||r_data||^2 + lambda_dyn ||r_dyn||^2 + lambda_meas ||r_meas||^2

    r_meas is self-supervised (uses only y, never x). Setting
    lambda_dyn = lambda_meas = 0 recovers a plain supervised filter -- the
    ablation baseline.

    Process-model usage (Issue 2): both f and h are used in the LOSS (GPU via
    filter_model.torch.f/.h, batched and differentiable); filter_model.torch is
    required for fit() (raises ValueError if None). The forward pass also uses
    f/h to build the innovation conditioning, exactly like KalmanNet.

    Hardware split (Issue 0): fit()/val batched on GPU (residuals computed on
    the produced [B, T, nx] sequence in parallel over the batch -- no
    teacher-forcing curriculum); estimate() strictly sequential on CPU with
    NumPy f/h, one trajectory / one timestep at a time (the physics loss is
    training-only).
    """

    estimator_id = "pinn"

    def __init__(
        self,
        filter_model,
        hidden_size: int = 64,
        lambda_dyn: float = 1.0,
        lambda_meas: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(filter_model, **kwargs)
        self._hidden_size = hidden_size
        self._lambda_dyn = lambda_dyn
        self._lambda_meas = lambda_meas

    def _build_network(self):
        return _PINNGRU.build(self._nx, self._ny, self._hidden_size)

    def _save_hyperparams(self) -> dict:
        return {
            "hidden_size": self._hidden_size,
            "lambda_dyn": self._lambda_dyn,
            "lambda_meas": self._lambda_meas,
        }

    # --- GPU batched forward + physics-informed loss --------------------

    def _forward_train(self, network, observations, states, timestamps, device,
                       feats_cache=None):
        """Free-running causal recursion over the batch, returning x_hat
        [B, T, nx]. Parallel over B; the T loop is the filter's intrinsic
        recurrence. Uses the batched torch f/h to build the innovation.

        Thin wrapper over `_forward_train_fused` for the default MSE-only path.
        feats_cache is unused (free-running forward is not cacheable); accepted
        only to match the base signature."""
        x_hat, _, _ = self._forward_train_fused(
            network, observations, states, timestamps, device,
            want_dyn=False, want_meas=False,
        )
        return x_hat

    def _forward_train_fused(self, network, observations, states, timestamps,
                             device, want_dyn, want_meas):
        """Single-pass forward that ALSO folds the physics residuals into the one
        T-loop it already runs (Issue 11), instead of two extra standalone
        T-loops in `_loss`. Returns (x_hat [B,T,nx], f_prev or None, h_hat or None):

          f_prev[:, t-1] = f(x_hat_{t-1}, ts[t-1])   for t = 1..T-1  (want_dyn)
          h_hat[:, t]    = h(x_hat_t,     ts[t])      for t = 0..T-1  (want_meas)

        These are exactly the tensors the old `_loss` built in its separate loops
        (`f_prev` = f over x_hat[:, :-1]; `h_hat` = h over x_hat), so the derived
        r_dyn/r_meas and the loss are numerically identical -- only the loop
        structure changed. When a want_* flag is False (the lambda==0 ablation)
        that residual's f/h is never evaluated, matching the old short-circuit.
        """
        import torch

        self._require_torch_dynamics()
        torch_f = self._model.torch.f
        torch_h = self._model.torch.h
        B, T, _ = observations.shape
        ts = timestamps.tolist()

        x = torch.zeros(B, self._nx, device=device, dtype=observations.dtype)
        h = network.init_hidden(B, device)
        estimates = []
        # r_dyn needs f(x_hat_{t-1}); accumulate one f-eval per step t>=1 on the
        # PREVIOUS estimate, giving the same [B, T-1, nx] stack as f over
        # x_hat[:, :-1]. r_meas needs h(x_hat_t); accumulate one h-eval per step.
        f_prev_list = [] if want_dyn else None
        h_hat_list = [] if want_meas else None
        x_hat_prev = None
        for t in range(T):
            t_val = ts[t]
            x_pred = torch_f(x, t_val)
            y_pred = torch_h(x_pred, t_val)
            innovation = observations[:, t, :] - y_pred
            dx, h = network.step(innovation, x_pred, h)
            x = x_pred + dx
            estimates.append(x)

            # f(x_hat_{t-1}, ts[t-1]): available from step t>=1 onward. Uses the
            # SAME (state, scalar-t) args as the old post-hoc loop over
            # x_hat[:, :-1] with ts[k], k=t-1.
            if want_dyn and x_hat_prev is not None:
                f_prev_list.append(torch_f(x_hat_prev, ts[t - 1]))
            # h(x_hat_t, ts[t]): SAME args as the old loop over x_hat with ts[t].
            if want_meas:
                h_hat_list.append(torch_h(x, t_val))
            x_hat_prev = x

        x_hat = torch.stack(estimates, dim=1)  # [B, T, nx]
        f_prev = torch.stack(f_prev_list, dim=1) if (want_dyn and f_prev_list) else None
        h_hat = torch.stack(h_hat_list, dim=1) if want_meas else None
        return x_hat, f_prev, h_hat

    def _loss(self, network, observations, states, timestamps, device, feats_cache=None):
        # feats_cache is unused: PINN's forward is free-running (each step uses the
        # net's own previous output), so it is not cacheable. The param exists only
        # to match the base _loss signature that fit() now calls with feats_cache.
        import torch.nn.functional as F

        # Fold the r_dyn/r_meas f/h sweeps into the single forward T-loop (Issue
        # 11). want_* gate exactly on the old lambda!=0 short-circuits so the
        # ablation baseline (lambda=0) evaluates no extra f/h -- identical work.
        want_dyn = self._lambda_dyn != 0.0
        want_meas = self._lambda_meas != 0.0
        x_hat, f_prev, h_hat = self._forward_train_fused(
            network, observations, states, timestamps, device,
            want_dyn=want_dyn, want_meas=want_meas,
        )
        loss = F.mse_loss(x_hat, states)  # r_data

        # r_dyn: x_hat_t vs f(x_hat_{t-1}, t-1) for t >= 1. f_prev was built in the
        # forward loop; identical to the old separate-loop tensor.
        if want_dyn and x_hat.shape[1] > 1:
            r_dyn = x_hat[:, 1:, :] - f_prev
            loss = loss + self._lambda_dyn * (r_dyn ** 2).mean()

        # r_meas: y_t vs h(x_hat_t, t) -- self-supervised (no ground truth).
        if want_meas:
            r_meas = observations - h_hat
            loss = loss + self._lambda_meas * (r_meas ** 2).mean()

        return loss

    # --- CPU strictly-sequential inference (physics loss is train-only) --

    def _estimate_sequential_cpu(self, network, observations, timestamps):
        import torch

        N, T, _ = observations.shape
        f = self._model.f
        h = self._model.h
        ts = np.asarray(timestamps, dtype=np.float64)
        obs_np = observations.numpy()
        out = np.zeros((N, T, self._nx), dtype=np.float64)

        for i in range(N):  # one trajectory at a time
            x = np.zeros(self._nx, dtype=np.float64)
            hidden = network.init_hidden(1, torch.device("cpu"))
            for t in range(T):  # one timestep at a time
                t_val = float(ts[t])
                x_pred = np.asarray(f(x, t_val), dtype=np.float64)
                y_pred = np.asarray(h(x_pred, t_val), dtype=np.float64)
                innovation = obs_np[i, t, :] - y_pred
                innov_t = torch.from_numpy(innovation.astype(np.float32)).unsqueeze(0)
                xpred_t = torch.from_numpy(x_pred.astype(np.float32)).unsqueeze(0)
                dx, hidden = network.step(innov_t, xpred_t, hidden)
                x = x_pred + dx.squeeze(0).numpy().astype(np.float64)
                out[i, t] = x
        return out
