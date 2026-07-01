from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .base import (
    BenchmarkLevel,
    BaseSimulator,
    FilterModel,
    split_counts as _split_counts,
)
from ._numba_dynamics import build_vehicle_tracking_numba_dynamics
from ._torch_dynamics import build_vehicle_tracking_torch_dynamics


# Floor on the sensor-to-target range: guards the range/bearing Jacobians against
# a divide-by-zero if the target passes exactly over a sensor. The true range is
# essentially never this small, so it never distorts a real reading. Kept in
# sync with _numba_dynamics._VT_RANGE_EPS.
_RANGE_EPS = 1.0e-9


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle (or array of angles) to (-pi, pi] via atan2(sin, cos).

    MUST be applied to every bearing component of an innovation y - h(x) -- this
    is the #1 footgun of a polar measurement: a raw subtraction of two bearings
    near the +/-pi branch cut can be ~2*pi wrong, which silently wrecks the
    EKF/UKF gain and any neural innovation feature. h() itself returns bearings
    already on (-pi, pi] (from atan2), so wrapping is only needed at the
    innovation site, never inside h.
    """
    return np.arctan2(np.sin(a), np.cos(a))


class VehicleTrackingSimulator(BaseSimulator):
    """Constant-velocity ground vehicle observed by K fixed range/bearing sensors.

    step: constant-velocity (CV) update + Gaussian process noise from Q.
    observe: stacked [range, bearing] per sensor + per-sensor Gaussian noise from
    the block-diagonal R; emitted bearings are wrapped to (-pi, pi]. With
    dropout_prob > 0, a sensor's [range, bearing] slot is emitted as NaN with that
    per-step, per-sensor probability (never 0.0 -- a fabricated zero would be a
    silent, wrong measurement).
    """

    def __init__(
        self,
        sensors: np.ndarray,   # [K, 2] fixed sensor positions
        dt: float,
        Q: np.ndarray,         # [4, 4] process-noise covariance
        R: np.ndarray,         # [2K, 2K] block-diagonal observation-noise covariance
        dropout_prob: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._sensors = np.ascontiguousarray(sensors, dtype=np.float64)
        self._K = self._sensors.shape[0]
        self._dt = dt
        self._Q = Q
        self._R = R
        self._dropout_prob = dropout_prob
        self._rng = rng if rng is not None else np.random.default_rng()

        self._F = np.eye(4)
        self._F[0, 2] = dt
        self._F[1, 3] = dt

    def step(
        self,
        state: np.ndarray,
        control: Optional[np.ndarray],
        dt: float,
    ) -> np.ndarray:
        new_state = self._F @ state
        noise = self._rng.multivariate_normal(np.zeros(4), self._Q)
        return new_state + noise

    def observe(self, state: np.ndarray) -> np.ndarray:
        px, py = state[0], state[1]
        out = np.empty(2 * self._K)
        noise = self._rng.multivariate_normal(np.zeros(2 * self._K), self._R)
        for k in range(self._K):
            dx = px - self._sensors[k, 0]
            dy = py - self._sensors[k, 1]
            r = np.sqrt(dx * dx + dy * dy)
            bearing = np.arctan2(dy, dx)
            out[2 * k] = r + noise[2 * k]
            # Wrap the noised bearing back onto (-pi, pi].
            out[2 * k + 1] = wrap_angle(bearing + noise[2 * k + 1])
        if self._dropout_prob > 0.0:
            for k in range(self._K):
                if self._rng.random() < self._dropout_prob:
                    out[2 * k] = np.nan
                    out[2 * k + 1] = np.nan
        return out


class VehicleTrackingBenchmark(BenchmarkLevel):
    """Multi-sensor range/bearing vehicle-tracking benchmark (Issues 5 & 6).

    A ground vehicle in Cartesian state x = [px, py, vx, vy] moves under a
    constant-velocity (CV) model (linear f, so F is the constant CV matrix, with
    a discrete white-noise-acceleration Q). K fixed sensor stations each report
    [range, bearing] with a DIFFERENT noise level; the readings stack into one
    y in R^{2K} and R is block-diagonal (one 2x2 block per sensor). This is the
    canonical radar/sonar multi-sensor fusion problem: the nonlinearity is pushed
    entirely into the (well-conditioned polar) measurement h, and the filter must
    fuse one precise sensor with several poor ones.

    Bearing angle-wrap (mandatory footgun): every innovation y - h(x) with a
    bearing component is wrapped to (-pi, pi] (see `wrap_angle`). This applies to
    EKF/UKF innovations and to any neural innovation feature. h() returns bearings
    already on (-pi, pi]; wrapping happens only at the innovation site.

    Heterogeneous noise: the per-sensor (sigma_r, sigma_b) differ (a precise
    station + cheap ones) and a global scalar `noise_scale` multiplies all of
    them for sweeps. Sensors are placed around the scene so the geometry (GDOP)
    varies along the trajectory.

    Optional dropout (`dropout_prob`, OFF by default): a sensor's slot is emitted
    as NaN with that per-step, per-sensor probability. When on, the metadata says
    so loudly and downstream estimators must gate (not average over) NaN; the
    default 0.0 keeps EKF/UKF and the current estimators running unmodified.

    Follow-ups noted but out of scope: a coordinated-turn variant (state
    [px,py,vx,vy,omega], nonlinear f) and NaN-gating estimators.

    Hardware/estimator contract is unchanged (Issue 0): states [N,T,4],
    observations [N,T,2K], timestamps [T]; ground truth (Cartesian states) is
    never noised, clipped, or dropped -- only observations carry noise/dropout.
    """

    def __init__(
        self,
        trajectory_length: int = 200,
        num_trajectories: int = 2000,
        random_seed: int = 42,
        dt: float = 0.1,
        num_sensors: int = 3,
        sensor_range_noise: Sequence[float] = (0.5, 2.0, 5.0),        # metres, per sensor
        sensor_bearing_noise_deg: Sequence[float] = (0.5, 2.0, 5.0),  # degrees, per sensor
        process_noise_intensity: float = 0.1,                        # DWNA intensity q
        noise_scale: float = 1.0,                                    # global multiplier for sweeps
        scene_size: float = 100.0,                                  # metres (square scene)
        initial_speed_range: float = 5.0,                          # m/s, per-axis uniform half-width
        initial_state_var: float = 4.0,                            # variance of the position prior
        sensor_positions: Optional[np.ndarray] = None,              # [K, 2]; default = scene corners
        dropout_prob: float = 0.0,
    ) -> None:
        if len(sensor_range_noise) != num_sensors or len(sensor_bearing_noise_deg) != num_sensors:
            raise ValueError(
                f"sensor_range_noise ({len(sensor_range_noise)}) and "
                f"sensor_bearing_noise_deg ({len(sensor_bearing_noise_deg)}) must each "
                f"have num_sensors={num_sensors} entries."
            )
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1]; got {dropout_prob}.")

        self._trajectory_length = trajectory_length
        self._num_trajectories = num_trajectories
        self._random_seed = random_seed
        self._dt = dt
        self._K = num_sensors
        self._noise_scale = noise_scale
        self._scene_size = scene_size
        self._initial_speed_range = initial_speed_range
        self._initial_state_var = initial_state_var
        self._dropout_prob = dropout_prob

        # Sensor placement: user-provided, else spread around the scene so the
        # range/bearing geometry (GDOP) varies as the vehicle moves. The default
        # spreads them near the corners/edges of the [0, scene_size]^2 box.
        if sensor_positions is not None:
            sensors = np.ascontiguousarray(sensor_positions, dtype=np.float64)
            if sensors.shape != (num_sensors, 2):
                raise ValueError(
                    f"sensor_positions must have shape ({num_sensors}, 2); got {sensors.shape}."
                )
        else:
            sensors = self._default_sensor_positions(num_sensors, scene_size)
        self._sensors = sensors

        # Discrete white-noise-acceleration (DWNA) Q for the per-axis CV model.
        q = process_noise_intensity
        self._Q = self._dwna_Q(dt, q)

        # Block-diagonal R: one diag(sigma_r^2, sigma_b^2) block per sensor, with
        # bearing noise converted from degrees to radians and the global
        # noise_scale applied to every standard deviation (so R scales by its
        # square).
        sigma_r = np.asarray(sensor_range_noise, dtype=np.float64) * noise_scale
        sigma_b = np.deg2rad(np.asarray(sensor_bearing_noise_deg, dtype=np.float64)) * noise_scale
        R = np.zeros((2 * self._K, 2 * self._K))
        for k in range(self._K):
            R[2 * k, 2 * k] = sigma_r[k] ** 2
            R[2 * k + 1, 2 * k + 1] = sigma_b[k] ** 2
        self._R = R

    @staticmethod
    def _default_sensor_positions(num_sensors: int, scene_size: float) -> np.ndarray:
        """Spread the sensors around the scene box so the observing geometry
        varies. Uses the corners in order, then the edge midpoints if K > 4."""
        s = scene_size
        candidates = np.array([
            [0.0, 0.0], [s, 0.0], [s, s], [0.0, s],           # corners
            [0.5 * s, 0.0], [s, 0.5 * s], [0.5 * s, s], [0.0, 0.5 * s],  # edge mids
        ])
        if num_sensors > candidates.shape[0]:
            raise ValueError(
                f"default sensor placement supports up to {candidates.shape[0]} sensors; "
                f"got num_sensors={num_sensors}. Pass sensor_positions explicitly."
            )
        return np.ascontiguousarray(candidates[:num_sensors], dtype=np.float64)

    @staticmethod
    def _dwna_Q(dt: float, q: float) -> np.ndarray:
        """Per-axis discrete white-noise-acceleration process covariance (the
        textbook CV Q), intensity q shared by x and y axes."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        Q = np.zeros((4, 4))
        # px-vx block (x axis)
        Q[0, 0] = dt4 / 4.0
        Q[0, 2] = dt3 / 2.0
        Q[2, 0] = dt3 / 2.0
        Q[2, 2] = dt2
        # py-vy block (y axis)
        Q[1, 1] = dt4 / 4.0
        Q[1, 3] = dt3 / 2.0
        Q[3, 1] = dt3 / 2.0
        Q[3, 3] = dt2
        return q * Q

    @property
    def name(self) -> str:
        return "vehicle_tracking"

    @property
    def description(self) -> str:
        base = (
            f"Multi-sensor range/bearing vehicle tracking: constant-velocity "
            f"Cartesian state observed by {self._K} heterogeneous-noise sensors."
        )
        if self._dropout_prob > 0.0:
            base += (
                f" WARNING: sensor dropout is ON (dropout_prob={self._dropout_prob}); "
                "observations contain NaN slots that estimators MUST gate, not average over."
            )
        return base

    @property
    def state_dimension(self) -> int:
        return 4

    @property
    def observation_dimension(self) -> int:
        return 2 * self._K

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("px", "py", "vx", "vy")

    def _sample_initial_states(self, rng: np.random.Generator, n_traj: int) -> np.ndarray:
        """x0 ~ position uniform in the scene box, velocity uniform in
        +/- initial_speed_range per axis. Returns [n_traj, 4]."""
        pos = rng.uniform(0.0, self._scene_size, size=(n_traj, 2))
        vel = rng.uniform(-self._initial_speed_range, self._initial_speed_range, size=(n_traj, 2))
        x0 = np.empty((n_traj, 4))
        x0[:, 0:2] = pos
        x0[:, 2:4] = vel
        return x0

    def generate_dataset(self, output_dir: Path) -> None:
        from datasets.schema import DatasetMetadata
        from datasets.hdf5_writer import HDF5Writer

        rng = np.random.default_rng(self._random_seed)
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = _split_counts(self._num_trajectories)

        nx = self.state_dimension
        ny = self.observation_dimension
        T = self._trajectory_length
        dt = self._dt

        for split_name, n_traj in splits.items():
            states = np.zeros((n_traj, T, nx))
            observations = np.zeros((n_traj, T, ny))
            timestamps = np.arange(T, dtype=float) * dt

            sim = VehicleTrackingSimulator(
                sensors=self._sensors, dt=dt, Q=self._Q, R=self._R,
                dropout_prob=self._dropout_prob, rng=rng,
            )

            x0 = self._sample_initial_states(rng, n_traj)
            for i in range(n_traj):
                x = x0[i]
                for t in range(T):
                    states[i, t] = x                      # ground truth: never noised
                    observations[i, t] = sim.observe(x)   # noise/dropout only here
                    x = sim.step(x, None, dt)

            metadata = DatasetMetadata(
                benchmark_name=self.name,
                state_dimension=self.state_dimension,
                observation_dimension=self.observation_dimension,
                trajectory_length=self._trajectory_length,
                num_trajectories=n_traj,
                random_seed=self._random_seed,
                generation_time=datetime.datetime.now(datetime.UTC).isoformat(),
            )
            HDF5Writer(output_dir / f"{split_name}.h5").write(
                states, observations, timestamps, metadata
            )

    def get_filter_model(self) -> FilterModel:
        dt = self._dt
        sensors = self._sensors
        K = self._K
        eps = _RANGE_EPS

        F_cv = np.eye(4)
        F_cv[0, 2] = dt
        F_cv[1, 3] = dt

        def f(x: np.ndarray, t: float = 0.0) -> np.ndarray:
            return F_cv @ x

        def h(x: np.ndarray, t: float = 0.0) -> np.ndarray:
            out = np.empty(2 * K)
            for k in range(K):
                dx = x[0] - sensors[k, 0]
                dy = x[1] - sensors[k, 1]
                r = np.sqrt(dx * dx + dy * dy)
                out[2 * k] = r
                out[2 * k + 1] = np.arctan2(dy, dx)
            return out

        def F_jac(x: np.ndarray) -> np.ndarray:
            return F_cv

        def H_jac(x: np.ndarray) -> np.ndarray:
            H = np.zeros((2 * K, 4))
            for k in range(K):
                dx = x[0] - sensors[k, 0]
                dy = x[1] - sensors[k, 1]
                r2 = dx * dx + dy * dy
                r = np.sqrt(r2)
                if r < eps:
                    r = eps
                    r2 = eps * eps
                H[2 * k, 0] = dx / r
                H[2 * k, 1] = dy / r
                H[2 * k + 1, 0] = -dy / r2
                H[2 * k + 1, 1] = dx / r2
            return H

        # Position prior centered at the scene center with `initial_state_var`;
        # velocity prior from the uniform speed range (var = (2a)^2 / 12).
        x0_mean = np.array([0.5 * self._scene_size, 0.5 * self._scene_size, 0.0, 0.0])
        vel_var = (2.0 * self._initial_speed_range) ** 2 / 12.0
        x0_cov = np.diag([
            self._initial_state_var, self._initial_state_var, vel_var, vel_var,
        ])

        # Bearing components (odd indices 2k+1) are angles: their innovation must
        # be wrapped to (-pi, pi]. Range components (even) are not.
        angular_obs_mask = np.zeros(2 * K, dtype=bool)
        angular_obs_mask[1::2] = True

        return FilterModel(
            f=f, h=h, F=F_jac, H=H_jac,
            Q=self._Q.copy(), R=self._R.copy(),
            x0_mean=x0_mean, x0_cov=x0_cov,
            numba=build_vehicle_tracking_numba_dynamics(sensors, dt),
            torch=build_vehicle_tracking_torch_dynamics(sensors, dt),
            angular_obs_mask=angular_obs_mask,
        )
