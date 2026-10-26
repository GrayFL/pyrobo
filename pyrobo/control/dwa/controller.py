"""Model-space dynamic window using the same force-driven plant as execution."""
from dataclasses import dataclass

import numpy as np
from graphmap.pose import Pose, PoseArray

from ...dynamics import ChassisDynamics, ChassisState, DynamicsConfig
from ...kinematics import Ackermann, DifferentialDrive, ModelCommand
from ...pointcloud import PointObstacles
from ...types import Bounds2D, Footprint, Twist2D, planar_state, pose_array, positive, wrap_angle
from .path import PathProgress, PathTracker, polyline_distances


@dataclass(frozen=True)
class DWAConfig:
    max_speed: float = 0.9
    max_yaw_rate: float = 1.5
    linear_acceleration: float = 0.9
    angular_acceleration: float = 2.5
    prediction_time: float = 1.8
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
    heading_slow_angle: float = np.pi / 3
    turn_in_place_angle: float = np.pi / 3
    turn_exit_angle: float = np.pi / 9
    turn_gain: float = 1.0
    min_turn_yaw_rate: float = 0.3
    yaw_reversal_weight: float = 0.2
    braking_timeout: float = 4.0
    allow_reverse: bool = True

    def __post_init__(self):
        for name in ("max_speed", "max_yaw_rate", "linear_acceleration", "angular_acceleration",
                     "prediction_time", "integration_dt", "lookahead", "goal_tolerance", "yaw_tolerance",
                     "heading_slow_angle", "turn_in_place_angle", "turn_exit_angle", "turn_gain", "braking_timeout"):
            positive(getattr(self, name), name)
        for name in ("position_weight", "path_weight", "heading_weight", "clearance_weight", "speed_weight",
                     "min_turn_yaw_rate", "yaw_reversal_weight"):
            positive(getattr(self, name), name, zero=True)
        if not self.turn_exit_angle < self.turn_in_place_angle <= np.pi or self.heading_slow_angle > np.pi:
            raise ValueError("turn angles require 0 < exit < enter <= pi and heading_slow_angle <= pi")
        for name in ("linear_samples", "angular_samples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")


@dataclass(frozen=True)
class DWAResult:
    command: ModelCommand
    trajectory: PoseArray
    safe: bool
    clearance: float
    candidates: int
    feasible_candidates: int
    mode: str = "following"


class DynamicWindow:
    """Sample wheel speeds or speed/steering, then predict forces and actuator state.

    A command is a model-native reference, not an instantaneous measured Twist.
    Collision checks include both prediction + emergency brake and one-cycle +
    emergency brake. A braking tail that fails to stop is never marked safe.
    """
    def __init__(self, config: DWAConfig | None = None, footprint: Footprint | None = None,
                 *, model: DifferentialDrive | Ackermann | None = None,
                 dynamics: DynamicsConfig | None = None):
        self.config = config or (DWAConfig(prediction_time=2.4, lookahead=1.6)
                                 if isinstance(model, Ackermann) else DWAConfig())
        self.footprint = footprint or Footprint()
        cfg = self.config
        self.model = model or DifferentialDrive(max_speed=cfg.max_speed, max_yaw_rate=cfg.max_yaw_rate)
        self.plant = ChassisDynamics(self.model, dynamics or DynamicsConfig(
            drive_acceleration=cfg.linear_acceleration, yaw_acceleration=cfg.angular_acceleration))
        self.reset()

    def reset(self):
        self._turn_active = False
        self._turn_direction = 0.0
        self.tracker = PathTracker()

    def _validate(self, pose, velocity, obstacles, dt, chassis=None):
        positive(dt, "control dt")
        if dt > self.config.prediction_time:
            raise ValueError("control dt must not exceed prediction_time")
        if pose.wrd_frame != obstacles.cloud.frame:
            raise ValueError("pose and obstacles must have the same explicit frame")
        if chassis is None:
            if isinstance(self.model, Ackermann):
                raise ValueError("Ackermann requires measured ChassisState, including steering at rest")
            chassis = ChassisState(pose, velocity)
        if (chassis.pose.wrd_frame != pose.wrd_frame or not np.allclose(chassis.pose.matrix, pose.matrix)
                or chassis.velocity != velocity):
            raise ValueError("ChassisState must match observed pose and velocity")
        return self.plant.pack(chassis)

    def _rollout(self, state, commands, dt, *, stop_only=False, braking=False):
        cfg, m = self.config, self.model
        values = np.broadcast_to(state, (len(commands), 10)).copy()
        trajectory, sweep = [values.copy()], [np.zeros(len(commands))]
        def phase(duration, mode):
            nonlocal values
            if duration <= 1e-12:
                return
            count = max(1, int(np.ceil(duration / self.plant.config.substep - 1e-10)))
            h = duration / count
            stride = max(1, int(np.floor(cfg.integration_dt / h + 1e-10)))
            travel = np.zeros(len(commands))
            for i in range(count):
                previous = values
                values = self.plant.integrate(values, commands, h, mode=mode)
                # The plant uses this exact constant midpoint twist per substep.
                mid = (previous[:, 3:5] + values[:, 3:5]) / 2
                travel += (np.abs(mid[:, 0]) + self.footprint.radius * np.abs(mid[:, 1])) * h
                if (i + 1) % stride == 0 or i == count - 1:
                    trajectory.append(values.copy())
                    sweep.append(travel / 2)
                    travel = np.zeros(len(commands))

        phase(dt, "estop" if braking else "track")
        if not stop_only and not braking:
            phase(max(0., cfg.prediction_time - dt), "track")
        endpoint = len(trajectory) - 1
        elapsed = 0.
        while np.any(np.max(np.abs(values[:, 3:5]), axis=1) > 1e-4) and elapsed < cfg.braking_timeout:
            h = min(cfg.integration_dt, cfg.braking_timeout - elapsed)
            phase(h, "estop")
            elapsed += h
        stopped = np.max(np.abs(values[:, 3:5]), axis=1) <= 1e-4
        return np.stack(trajectory, axis=1), np.stack(sweep, axis=1), endpoint, stopped

    def _clearance(self, trajectories, sweep, obstacles, bounds):
        clear = obstacles.clearances(trajectories[..., :3], self.footprint, bounds)
        intervals = np.minimum(clear[:, :-1], clear[:, 1:]) - sweep[:, 1:]
        return np.minimum(clear.min(axis=1), intervals.min(axis=1))

    def stop(self, pose: Pose, velocity: Twist2D, obstacles: PointObstacles,
             bounds: Bounds2D, dt: float, *, chassis: ChassisState | None = None) -> DWAResult:
        state = self._validate(pose, velocity, obstacles, dt, chassis)
        command = self.model.command(mode="estop")
        commands = self.model.encode_command(command)[None]
        traj, sweep, _, stopped = self._rollout(state, commands, dt, braking=True)
        clearance = float(self._clearance(traj, sweep, obstacles, bounds)[0])
        safe = clearance > 0 and bool(stopped[0])
        return DWAResult(command, pose_array(traj[0, :, :3], pose.wrd_frame, pose.ego_frame),
                         safe, clearance, 1, int(safe), "braking")

    def _commands(self, state, speed_limit, desired_turn, mode, dt):
        cfg, m, dyn = self.config, self.model, self.plant.config
        v, w = state[3:5]
        # Bound native references around measured actuators. Servo lag is part
        # of the window; force/jerk limits bound reachable states in the rollout.
        horizon = dt + max(dyn.velocity_time_constant, dyn.yaw_time_constant)
        vlo = max(-m.max_reverse_speed if cfg.allow_reverse else 0., v - dyn.brake_acceleration * horizon)
        vhi = min(m.max_speed, v + dyn.drive_acceleration * horizon, speed_limit)
        vhi = max(vlo, vhi)
        if isinstance(m, DifferentialDrive):
            if mode in ("turning", "turn_braking", "aligning"):
                vs = np.array([0.])
            else:
                vs = np.unique(np.r_[np.linspace(vlo, vhi, cfg.linear_samples), np.clip(0, vlo, vhi)])
            reach = dyn.yaw_acceleration * horizon
            ws = np.unique(np.r_[np.linspace(max(-m.max_yaw_rate, w - reach),
                                             min(m.max_yaw_rate, w + reach), cfg.angular_samples),
                                 0., np.clip(desired_turn, -m.max_yaw_rate, m.max_yaw_rate)])
            # Native wheel-space grid plus straight and pure-turn references.
            extent = max(abs(vlo), abs(vhi)) + m.track * max(abs(ws.min()), abs(ws.max())) / 2
            left = np.linspace(max(-m.max_wheel_speed, v - m.track*w/2 - extent),
                               min(m.max_wheel_speed, v - m.track*w/2 + extent), cfg.angular_samples)
            right = np.linspace(max(-m.max_wheel_speed, v + m.track*w/2 - extent),
                                min(m.max_wheel_speed, v + m.track*w/2 + extent), cfg.angular_samples)
            l, r = np.meshgrid(left, right)
            pairs = np.column_stack((l.ravel(), r.ravel()))
            pv, _ = m.references(pairs)
            pairs = pairs[(pv >= vlo - 1e-9) & (pv <= vhi + 1e-9)]
            if mode in ("turning", "turn_braking", "aligning"):
                pairs = np.empty((0, 2))
            anchors = np.array([m.encode_command(m.command(speed, yaw)) for speed in vs for yaw in ws])
            return np.unique(np.vstack((pairs, anchors)), axis=0)
        vs = np.unique(np.r_[np.linspace(vlo, vhi, cfg.linear_samples), np.clip(0., vlo, vhi)])
        steering = np.unique(np.r_[np.linspace(-m.max_steering, m.max_steering, cfg.angular_samples),
                                   np.clip(desired_turn, -m.max_steering, m.max_steering), state[5], 0.])
        vgrid, sgrid = np.meshgrid(vs, steering)
        return np.column_stack((vgrid.ravel(), sgrid.ravel()))

    def step(self, pose: Pose, velocity: Twist2D, goal: Pose, path: PoseArray,
             obstacles: PointObstacles, bounds: Bounds2D, dt: float, *,
             chassis: ChassisState | None = None, progress: PathProgress | None = None) -> DWAResult:
        state = self._validate(pose, velocity, obstacles, dt, chassis)
        target_state = planar_state(goal)
        if goal.wrd_frame != pose.wrd_frame or path.wrd_frame != pose.wrd_frame:
            raise ValueError("goal and path frames must match the robot")
        xy = np.asarray(path.position)[..., :2]
        if xy.ndim != 2 or len(xy) == 0 or not np.isfinite(xy).all():
            raise ValueError("path must be a nonempty 1D PoseArray")
        cfg, m = self.config, self.model
        if progress is None:
            if not np.array_equal(self.tracker.xy, xy):
                self.tracker.set_path(path)
            progress = self.tracker.update(pose, cfg.lookahead)
        target = progress.target.translation[:2]
        remaining = progress.remaining.position[:, :2]
        distance_to_goal = np.linalg.norm(state[:2] - target_state[:2])
        near_goal = distance_to_goal <= cfg.goal_tolerance
        direction = np.arctan2(target[1] - state[1], target[0] - state[0])
        signed_error = float(wrap_angle(direction - state[2]))
        heading_error = abs(signed_error)
        diff = isinstance(m, DifferentialDrive)
        if diff and not near_goal:
            if self._turn_active and heading_error <= cfg.turn_exit_angle and abs(velocity.angular) < .05:
                self._turn_active = False
            if not self._turn_active and heading_error >= cfg.turn_in_place_angle and abs(velocity.linear) < .05:
                self._turn_active = True
                self._turn_direction = float(np.sign(velocity.angular) if abs(velocity.angular) > 1e-3 else (np.sign(signed_error) or 1))
            if self._turn_active and heading_error < np.pi / 2 and signed_error != 0:
                self._turn_direction = float(np.sign(signed_error))
        elif near_goal:
            self._turn_active = False
        mode = ("aligning" if near_goal and diff else
                "turn_braking" if self._turn_active and heading_error <= cfg.turn_exit_angle else
                "turning" if self._turn_active else "following")
        speed_limit = m.max_speed * max(.0 if diff else .25, np.cos(min(heading_error / cfg.heading_slow_angle, 1) * np.pi / 2))
        desired = cfg.turn_gain * signed_error
        if mode == "turning":
            desired = self._turn_direction * min(m.max_yaw_rate, max(cfg.min_turn_yaw_rate, cfg.turn_gain * heading_error))
        elif mode == "turn_braking":
            desired = 0.
        elif near_goal and diff:
            desired = cfg.turn_gain * float(wrap_angle(target_state[2] - state[2]))
        elif not diff:
            desired = np.arctan2(2 * m.wheelbase * np.sin(signed_error), max(np.linalg.norm(target - state[:2]), .1))
        commands = self._commands(state, speed_limit, desired, mode, dt)
        trajectories, sweep, endpoint, stopped = self._rollout(state, commands, dt)
        clearance = self._clearance(trajectories, sweep, obstacles, bounds)
        short, short_sweep, _, short_stopped = self._rollout(state, commands, dt, stop_only=True)
        clearance = np.minimum(clearance, self._clearance(short, short_sweep, obstacles, bounds))
        feasible = (clearance > 0) & stopped & short_stopped
        if not feasible.any():
            result = self.stop(pose, velocity, obstacles, bounds, dt, chassis=chassis)
            return DWAResult(result.command, result.trajectory, result.safe, result.clearance, len(commands), 0, "blocked")
        final = trajectories[:, endpoint]
        speed, turn = m.references(commands)
        if near_goal and diff:
            heading = np.abs(wrap_angle(final[:, 2] - target_state[2]))
            score = 4 * heading + 3 * np.linalg.norm(final[:, :2] - target_state[:2], axis=1) + abs(speed)
            score += .15 * np.abs(turn)
        elif mode == "turn_braking":
            score = 5 * turn**2 + 3 * abs(speed)
        elif mode == "turning":
            directed_error = signed_error
            if directed_error * self._turn_direction < 0:
                directed_error += self._turn_direction * 2 * np.pi
            heading = np.abs(final[:, 2] - state[2] - directed_error)
            score = 4 * heading + 3 * abs(speed) + .1 * ((turn - desired) / m.max_yaw_rate)**2
        else:
            target_distance = np.linalg.norm(final[:, :2] - target, axis=1)
            path_distance = polyline_distances(final[:, :2], remaining)
            heading = 1 - np.cos(final[:, 2] - direction)
            speed_cost = ((speed - speed_limit) / m.max_speed)**2
            score = (cfg.position_weight * target_distance + cfg.path_weight * path_distance
                     + cfg.heading_weight * heading + cfg.speed_weight * speed_cost)
            score += .15 * np.maximum(-speed, 0.) / max(m.max_reverse_speed, .01)
            if not diff and progress.remaining_distance < cfg.lookahead:
                # A car must align while moving. Do not switch to a fictitious
                # in-place rotation at the last waypoint.
                weight = 2 * (1 - min(distance_to_goal / cfg.lookahead, 1))
                score += weight * (1 - np.cos(final[:, 2] - target_state[2]))
                if near_goal:
                    score += 2 * np.linalg.norm(final[:, :2] - target_state[:2], axis=1)
            if diff and abs(velocity.angular) > 1e-3:
                reversal = np.maximum(-turn * np.sign(velocity.angular), 0)
                score += cfg.yaw_reversal_weight * (reversal / max(cfg.angular_acceleration * dt, .1))**2
        score += cfg.clearance_weight / (.1 + np.maximum(clearance, 0))
        score[~feasible] = np.inf
        best = int(np.argmin(score))
        return DWAResult(m.decode_command(commands[best]),
                         pose_array(trajectories[best, :, :3], pose.wrd_frame, pose.ego_frame), True,
                         float(clearance[best]), len(commands), int(feasible.sum()), mode)
