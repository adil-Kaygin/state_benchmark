from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np

from ._neural_base import SequentialNeuralFilter, dt_array

if TYPE_CHECKING:
    import torch


class _NeuralODENet:
    """Builds the two learned components of the Neural-ODE filter:

      g_theta(x, t)  -- the drift MLP, dx/dt = g_theta(x, t). Input is the
                        state augmented with a scalar time feature; output is
                        the [nx] derivative. Batched: [B, nx] -> [B, nx].
      c_phi(feat)    -- the correction MLP. Input is [innovation, x_pred] of
                        width (ny + nx); output is the [nx] state correction
                        applied after the observation.

    The same nn.Module weights drive both the parallel-over-batch GPU forward
    and the strictly-sequential CPU inference -- only the integrator host
    (torch on GPU vs torch-on-CPU with NumPy f/h) differs between the two.
    """

    @staticmethod
    def build(nx: int, ny: int, ode_hidden: int, ode_layers: int, correction_hidden: int):
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # Drift g_theta: [x, t] -> dx/dt. The +1 input is a scalar time
                # feature so a time-varying learned drift is representable.
                drift_layers = [nn.Linear(nx + 1, ode_hidden), nn.Tanh()]
                for _ in range(max(ode_layers - 1, 0)):
                    drift_layers += [nn.Linear(ode_hidden, ode_hidden), nn.Tanh()]
                drift_layers += [nn.Linear(ode_hidden, nx)]
                self.drift = nn.Sequential(*drift_layers)
                # Zero-init the last drift layer: an untrained filter then has
                # zero learned drift, a stable starting point mirroring
                # KalmanNet's K=0 init.
                nn.init.zeros_(self.drift[-1].weight)
                nn.init.zeros_(self.drift[-1].bias)

                # Correction c_phi: [innovation, x_pred] -> dx.
                self.correction = nn.Sequential(
                    nn.Linear(ny + nx, correction_hidden), nn.ReLU(),
                    nn.Linear(correction_hidden, correction_hidden), nn.ReLU(),
                    nn.Linear(correction_hidden, nx),
                )
                nn.init.zeros_(self.correction[-1].weight)
                nn.init.zeros_(self.correction[-1].bias)
                self.nx = nx
                self.ny = ny

            def drift_fn(self, x, t_val):
                """g_theta(x, t): [B, nx] -> [B, nx]. t_val is a python float
                broadcast as a constant time-feature column."""
                t_col = torch.full((x.shape[0], 1), float(t_val), dtype=x.dtype, device=x.device)
                return self.drift(torch.cat([x, t_col], dim=1))

            def correct(self, innovation, x_pred):
                return self.correction(torch.cat([innovation, x_pred], dim=1))

        return _Module()


class NeuralODEEstimator(SequentialNeuralFilter):
    """
    Continuous-time recursive filter. Between observations the latent state is
    propagated by a LEARNED ODE dx/dt = g_theta(x, t) integrated over each
    inter-observation dt with a dependency-free explicit RK4 (n_substeps fixed
    steps). At each observation a LEARNED, innovation-driven correction is
    applied, in the spirit of a continuous-discrete EKF but with both the drift
    and the correction learned:

        x_pred(t_k) = x(t_{k-1}) + integral_{t_{k-1}}^{t_k} g_theta(x, t) dt
        innov       = y_k - h(x_pred(t_k))            # h = true filter_model.h
        x_post(t_k) = x_pred(t_k) + c_phi([innov, x_pred])

    Process-model usage (Issue 1):
      - h (observation model): the TRUE filter_model.h is always used to form
        the innovation (filter_model.torch.h on GPU, NumPy filter_model.h on
        the CPU inference path).
      - f (process model): the learned drift g_theta REPLACES f by default. With
        use_model_drift=True the drift is learned as a RESIDUAL on top of the
        known dynamics: dx/dt ~= (f(x) - x)/dt_nominal + g_theta(x, t).

    Hardware split (Issue 0):
      - fit()/val: fully batched on GPU. The loop over T is intrinsic (a filter
        is causal) but each step is a single batched [B, nx] RK4 solve + batched
        correction using filter_model.torch.h -- parallelism is over the batch B.
        No teacher-forcing curriculum: the integrator handles the within-step
        continuous propagation, there is no GRU recurrence to warm-start.
      - estimate()/inference: strictly sequential on CPU, one trajectory / one
        timestep at a time, RK4 stepping with NumPy filter_model.h (and
        filter_model.f if use_model_drift). This is the embedded deployment
        latency the benchmark measures.

    Integrator (dependency policy, Issue 1): default solver="rk4" is a
    dependency-free explicit RK4 in plain PyTorch (fully differentiable for
    backprop-through-the-solver on GPU). solver="dopri5" uses torchdiffeq if
    importable; lazy import, clear ImportError otherwise (never a silent
    fallback).
    """

    estimator_id = "neural_ode"

    def __init__(
        self,
        filter_model,
        ode_hidden: int = 64,
        ode_layers: int = 2,
        n_substeps: int = 4,
        solver: str = "rk4",
        correction_hidden: int = 64,
        use_model_drift: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(filter_model, **kwargs)
        if solver not in ("rk4", "dopri5"):
            raise ValueError(
                f"NeuralODEEstimator: unknown solver '{solver}'. "
                "Use 'rk4' (dependency-free) or 'dopri5' (needs torchdiffeq)."
            )
        self._ode_hidden = ode_hidden
        self._ode_layers = ode_layers
        self._n_substeps = max(int(n_substeps), 1)
        self._solver = solver
        self._correction_hidden = correction_hidden
        self._use_model_drift = use_model_drift
        # Nominal dt for the residual-drift scaling; refined per call from the
        # actual timestamps in fit()/estimate().
        self._dt_nominal = 1.0

    def _build_network(self):
        return _NeuralODENet.build(
            self._nx, self._ny, self._ode_hidden, self._ode_layers, self._correction_hidden
        )

    def _save_hyperparams(self) -> dict:
        return {
            "ode_hidden": self._ode_hidden,
            "ode_layers": self._ode_layers,
            "n_substeps": self._n_substeps,
            "solver": self._solver,
            "correction_hidden": self._correction_hidden,
            "use_model_drift": self._use_model_drift,
        }

    # --- shared drift assembly (model-drift residual option) -------------

    def _torch_drift(self, network, x, t_val, torch_f):
        """Total drift dx/dt at batched state x and time t_val on the GPU path.
        Pure learned drift by default; with use_model_drift, add the known
        dynamics as a residual: (f(x) - x)/dt_nominal + g_theta."""
        learned = network.drift_fn(x, t_val)
        if not self._use_model_drift:
            return learned
        f_x = torch_f(x, t_val)
        return (f_x - x) / self._dt_nominal + learned

    def _rk4_step_torch(self, network, x, t_val, dt, torch_f):
        """One RK4 integration over dt (batched, on device), n_substeps stages."""
        sub_dt = dt / self._n_substeps
        for _ in range(self._n_substeps):
            k1 = self._torch_drift(network, x, t_val, torch_f)
            k2 = self._torch_drift(network, x + 0.5 * sub_dt * k1, t_val, torch_f)
            k3 = self._torch_drift(network, x + 0.5 * sub_dt * k2, t_val, torch_f)
            k4 = self._torch_drift(network, x + sub_dt * k3, t_val, torch_f)
            x = x + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    # --- GPU batched forward (parallel over B; T loop is intrinsic) ------

    def _forward_train(self, network, observations, states, timestamps, device):
        import torch

        self._require_torch_dynamics()
        if self._solver != "rk4":
            # dopri5 (torchdiffeq) is the optional path; guard the dep loudly.
            try:
                import torchdiffeq  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "NeuralODEEstimator solver='dopri5' needs torchdiffeq. "
                    "Install torchdiffeq or use solver='rk4'."
                ) from exc

        torch_f = self._model.torch.f
        torch_h = self._model.torch.h

        B, T, _ = observations.shape
        ts = timestamps.tolist()
        dt = dt_array(timestamps)
        self._dt_nominal = float(np.mean(dt[1:])) if T > 1 else float(dt[0])

        if self._model.x0_mean is not None:
            x0 = torch.as_tensor(np.asarray(self._model.x0_mean), dtype=observations.dtype, device=device)
            x = x0.unsqueeze(0).expand(B, self._nx).contiguous()
        else:
            x = torch.zeros(B, self._nx, device=device, dtype=observations.dtype)

        estimates = []
        for t in range(T):
            t_val = ts[t]
            x_pred = self._rk4_step_torch(network, x, t_val, float(dt[t]), torch_f)
            y_pred = torch_h(x_pred, t_val)
            innovation = observations[:, t, :] - y_pred
            x_post = x_pred + network.correct(innovation, x_pred)
            estimates.append(x_post)
            x = x_post
        return torch.stack(estimates, dim=1)  # [B, T, nx]

    # --- CPU strictly-sequential inference ------------------------------

    def _rk4_step_numpy(self, network, x_vec, t_val, dt):
        """One RK4 integration over dt on a single state vector (NumPy host) --
        the embedded deployment integrator. The learned drift is evaluated
        through the network on a length-1 batch; the model drift (if any) uses
        NumPy filter_model.f."""
        import torch
        f = self._model.f
        sub_dt = dt / self._n_substeps

        def drift(xv):
            xt = torch.from_numpy(np.asarray(xv, dtype=np.float32)).unsqueeze(0)
            learned = network.drift_fn(xt, t_val).squeeze(0).numpy().astype(np.float64)
            if not self._use_model_drift:
                return learned
            f_x = np.asarray(f(xv, t_val), dtype=np.float64)
            return (f_x - xv) / self._dt_nominal + learned

        x = np.asarray(x_vec, dtype=np.float64)
        for _ in range(self._n_substeps):
            k1 = drift(x)
            k2 = drift(x + 0.5 * sub_dt * k1)
            k3 = drift(x + 0.5 * sub_dt * k2)
            k4 = drift(x + sub_dt * k3)
            x = x + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x

    def _estimate_sequential_cpu(self, network, observations, timestamps):
        import torch

        N, T, _ = observations.shape
        h = self._model.h
        dt = dt_array(timestamps)
        ts = np.asarray(timestamps, dtype=np.float64)
        if T > 1:
            self._dt_nominal = float(np.mean(dt[1:]))

        out = np.zeros((N, T, self._nx), dtype=np.float64)
        x0 = (
            np.asarray(self._model.x0_mean, dtype=np.float64)
            if self._model.x0_mean is not None else np.zeros(self._nx, dtype=np.float64)
        )
        obs_np = observations.numpy()
        for i in range(N):  # one trajectory at a time
            x = x0.copy()
            for t in range(T):  # one timestep at a time
                t_val = float(ts[t])
                x_pred = self._rk4_step_numpy(network, x, t_val, float(dt[t]))
                y_pred = np.asarray(h(x_pred, t_val), dtype=np.float64)
                innovation = obs_np[i, t, :] - y_pred
                innov_t = torch.from_numpy(innovation.astype(np.float32)).unsqueeze(0)
                xpred_t = torch.from_numpy(x_pred.astype(np.float32)).unsqueeze(0)
                correction = network.correct(innov_t, xpred_t).squeeze(0).numpy().astype(np.float64)
                x = x_pred + correction
                out[i, t] = x
        return out
