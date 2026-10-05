"""Dynamic-window sampling with rectangular footprint and braking-tail checks."""
from dataclasses import dataclass

import numpy as np
from graphmap.pose import Pose, PoseArray
from scipy.spatial import cKDTree

from ...motion import approach, ramp_step
from ...pointcloud import PointObstacles
from ...types import Bounds2D, Footprint, Twist2D, planar_state, pose_array, positive, wrap_angle


@dataclass(frozen=True)
class DWAConfig:
    max_speed: float = 0.9
    max_yaw_rate: float = 1.5
    linear_acceleration: float = 0.9
    angular_acceleration: float = 2.5
    prediction_time: float = 1.6
    integration_dt: float = 0.08
    linear_samples: int = 7
    angular_samples: int = 17
    lookahead: float = 1.3
    goal_tolerance: float = 0.18
    yaw_tolerance: float = 0.10
    position_weight: float = 3.0
    path_weight: float = 2.0
    heading_weight: float = 0.65
    clearance_weight: float = 0.08
    speed_weight: float = 0.15

    def __post_init__(self):
        for name in ("max_speed", "max_yaw_rate", "linear_acceleration", "angular_acceleration",
                     "prediction_time", "integration_dt", "lookahead", "goal_tolerance", "yaw_tolerance"):
            positive(getattr(self, name), name)
        for name in ("position_weight", "path_weight", "heading_weight", "clearance_weight", "speed_weight"):
            positive(getattr(self, name), name, zero=True)
        for name in ("linear_samples", "angular_samples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")


@dataclass(frozen=True)
class DWAResult:
    command: Twist2D
    trajectory: PoseArray
    safe: bool
    clearance: float
    candidates: int
    feasible_candidates: int


class DynamicWindow:
    """Stateless local controller; measured velocity is supplied on every call.

    Commands are setpoints reached under acceleration limits, not instantaneous
    twists. Forward-only differential-drive/unicycle; an in-place turn is allowed.
    """
    def __init__(self, config: DWAConfig | None = None, footprint: Footprint | None = None):
        self.config = config or DWAConfig()
        self.footprint = footprint or Footprint()

    def _validate(self, pose, velocity, obstacles, dt):
        state = planar_state(pose)
        positive(dt, "control dt")
        if dt > self.config.prediction_time:
            raise ValueError("control dt must not exceed prediction_time")
        if pose.wrd_frame != obstacles.cloud.frame:
            raise ValueError("pose and obstacles must have the same explicit frame")
        if not -1e-9 <= velocity.linear <= self.config.max_speed + 1e-9:
            raise ValueError("measured forward speed is outside configured limits")
        if abs(velocity.angular) > self.config.max_yaw_rate + 1e-9:
            raise ValueError("measured yaw rate is outside configured limits")
        return state

    def _rollout(self, state, velocity, commands, dt, *, stop_only=False):
        cfg = self.config
        states = np.broadcast_to(state, (len(commands), 3)).copy()
        speeds = np.tile([velocity.linear, velocity.angular], (len(commands), 1))
        trajectory, sweep = [states.copy()], [np.zeros(len(commands))]

        def phase(duration, targets):
            nonlocal states, speeds
            count = max(1, int(np.ceil(duration / cfg.integration_dt)))
            h = duration / count
            for _ in range(count):
                old_speeds = speeds
                states, speeds = ramp_step(states, speeds, targets, cfg.linear_acceleration,
                                          cfg.angular_acceleration, h)
                # Every footprint point moves by at most this arc-length bound.
                vmax = np.maximum(np.abs(old_speeds[:, 0]), np.abs(speeds[:, 0]))
                wmax = np.maximum(np.abs(old_speeds[:, 1]), np.abs(speeds[:, 1]))
                sweep.append((vmax + self.footprint.radius * wmax) * h / 2)
                trajectory.append(states.copy())

        phase(dt, commands)
        if not stop_only:
            phase(max(0.0, cfg.prediction_time - dt), commands)
        endpoint = len(trajectory) - 1
        brake_time = max(np.max(np.abs(speeds[:, 0])) / cfg.linear_acceleration,
                         np.max(np.abs(speeds[:, 1])) / cfg.angular_acceleration)
        if brake_time > 0:
            phase(brake_time, np.zeros_like(commands))
        return np.stack(trajectory, axis=1), np.stack(sweep, axis=1), endpoint

    def _clearance(self, trajectories, sweep, obstacles, bounds):
        clear = obstacles.clearances(trajectories, self.footprint, bounds)
        # Both endpoints minus half the maximum travel bound cover each interval.
        intervals = np.minimum(clear[:, :-1], clear[:, 1:]) - sweep[:, 1:]
        return np.minimum(clear.min(axis=1), intervals.min(axis=1))

    def stop(self, pose: Pose, velocity: Twist2D, obstacles: PointObstacles,
             bounds: Bounds2D, dt: float) -> DWAResult:
        state = self._validate(pose, velocity, obstacles, dt)
        cfg = self.config
        commands = np.array([[approach(velocity.linear, 0, cfg.linear_acceleration, dt),
                              approach(velocity.angular, 0, cfg.angular_acceleration, dt)]], dtype=float)
        traj, sweep, _ = self._rollout(state, velocity, commands, dt, stop_only=True)
        clearance = float(self._clearance(traj, sweep, obstacles, bounds)[0])
        safe = clearance > 0
        return DWAResult(Twist2D(*commands[0]), pose_array(traj[0], pose.wrd_frame, pose.ego_frame),
                         safe, clearance, 1, int(safe))

    def step(self, pose: Pose, velocity: Twist2D, goal: Pose, path: PoseArray,
             obstacles: PointObstacles, bounds: Bounds2D, dt: float) -> DWAResult:
        state = self._validate(pose, velocity, obstacles, dt)
        target_state = planar_state(goal)
        if goal.wrd_frame != pose.wrd_frame or path.wrd_frame != pose.wrd_frame:
            raise ValueError("goal and path frames must match the robot")
        xy = np.asarray(path.position)[..., :2]
        if xy.ndim != 2 or len(xy) == 0 or not np.isfinite(xy).all():
            raise ValueError("path must be a nonempty 1D PoseArray")
        cfg = self.config
        distance_to_goal = np.linalg.norm(state[:2] - target_state[:2])
        near_goal = distance_to_goal <= cfg.goal_tolerance
        vlo = max(0.0, velocity.linear - cfg.linear_acceleration * dt)
        vhi = min(cfg.max_speed, velocity.linear + cfg.linear_acceleration * dt)
        if near_goal:
            vhi = vlo
        wlo = max(-cfg.max_yaw_rate, velocity.angular - cfg.angular_acceleration * dt)
        whi = min(cfg.max_yaw_rate, velocity.angular + cfg.angular_acceleration * dt)
        vs = np.unique(np.r_[np.linspace(vlo, vhi, cfg.linear_samples), np.clip(0, vlo, vhi)])
        ws = np.unique(np.r_[np.linspace(wlo, whi, cfg.angular_samples), np.clip(0, wlo, whi)])
        v, w = np.meshgrid(vs, ws, indexing="ij")
        commands = np.column_stack((v.ravel(), w.ravel()))
        trajectories, sweep, endpoint = self._rollout(state, velocity, commands, dt)
        clearance = self._clearance(trajectories, sweep, obstacles, bounds)
        # Also verify that braking immediately after this control interval is viable.
        stop_traj, stop_sweep, _ = self._rollout(state, velocity, commands, dt, stop_only=True)
        clearance = np.minimum(clearance, self._clearance(stop_traj, stop_sweep, obstacles, bounds))
        feasible = clearance > 0
        if not feasible.any():
            result = self.stop(pose, velocity, obstacles, bounds, dt)
            return DWAResult(result.command, result.trajectory, result.safe, result.clearance, len(commands), 0)

        nearest = int(np.argmin(np.linalg.norm(xy - state[:2], axis=1)))
        remaining = xy[nearest:]
        lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(remaining, axis=0), axis=1))]
        idx = min(int(np.searchsorted(lengths, cfg.lookahead)), len(remaining) - 1)
        target = remaining[idx]
        final = trajectories[:, endpoint]
        if near_goal:
            heading = np.abs(wrap_angle(final[:, 2] - target_state[2]))
            score = 4 * heading + 3 * np.linalg.norm(final[:, :2] - target_state[:2], axis=1) + commands[:, 0]
            # Do not keep rotating once stopped at the requested orientation.
            score += 0.15 * np.abs(commands[:, 1])
        else:
            target_distance = np.linalg.norm(final[:, :2] - target, axis=1)
            path_distance = cKDTree(remaining).query(final[:, :2])[0]
            direction = np.arctan2(target[1] - state[1], target[0] - state[0])
            heading = 1 - np.cos(final[:, 2] - direction)
            score = (cfg.position_weight * target_distance + cfg.path_weight * path_distance
                     + cfg.heading_weight * heading - cfg.speed_weight * commands[:, 0])
        score += cfg.clearance_weight / (0.1 + np.maximum(clearance, 0))
        score[~feasible] = np.inf
        best = int(np.argmin(score))
        return DWAResult(Twist2D(*commands[best]),
                         pose_array(trajectories[best], pose.wrd_frame, pose.ego_frame), True,
                         float(clearance[best]), len(commands), int(feasible.sum()))

