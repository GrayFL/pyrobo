"""Shared, NumPy-batched force-driven plant for manual control and prediction.

Modelled after assets/ChassisDynamics: implicit passive axes, discrete-inverse
servos, finite motor/tire budgets and explicit state guards. No plotting or I/O.
"""
from dataclasses import dataclass, field

import numpy as np
from graphmap.pose import Pose

from .kinematics import Ackermann, DifferentialDrive, ModelCommand
from .motion import advance
from .types import Twist2D, planar_state, pose2d, positive


@dataclass(frozen=True)
class Drag:
    coulomb: float = 2.0
    linear: float = 8.0
    quadratic: float = 1.0
    epsilon: float = 0.02

    def __post_init__(self):
        for name, value in vars(self).items():
            positive(value, name, zero=name != "epsilon")

    def __call__(self, speed):
        speed = np.asarray(speed)
        return (self.coulomb * np.tanh(speed / self.epsilon) + self.linear * speed
                + self.quadratic * np.abs(speed) * speed)

    def derivative(self, speed):
        return (self.coulomb / self.epsilon * (1 - np.tanh(speed / self.epsilon) ** 2)
                + self.linear + 2 * self.quadratic * np.abs(speed))


def implicit_axis(position, speed, effort, mass, spring, drag: Drag, dt):
    """Backward Euler of M*qddot = effort - K*q - drag(qdot).

    The monotone scalar solve is vectorized across all candidates. Bracketed
    Newton steps preserve dissipativity even for large steps and stiff drag.
    """
    positive(dt, "dt")
    if (not all(np.isfinite(value).all() for value in (position, speed, effort, mass, spring))
            or np.any(np.asarray(mass) <= 0) or np.any(np.asarray(spring) < 0)):
        raise ValueError("axis inputs must be finite, mass positive and spring nonnegative")
    effective = mass + spring * dt * dt
    free = (mass * speed + dt * (effort - spring * position)) / effective
    lo, hi = np.minimum(0., free), np.maximum(0., free)
    value = (lo + hi) / 2
    scale = dt / effective
    for _ in range(32):
        residual = value + scale * drag(value) - free
        if np.all(np.abs(residual) < 1e-12):
            break
        lo, hi = np.where(residual < 0, value, lo), np.where(residual > 0, value, hi)
        trial = value - residual / (1 + scale * drag.derivative(value))
        value = np.where((trial >= lo) & (trial <= hi), trial, (lo + hi) / 2)
    return position + dt * value, value


def _effort(position, speed, acceleration, mass, spring, drag, dt):
    next_speed = speed + dt * acceleration
    return mass * acceleration + spring * (position + dt * next_speed) + drag(next_speed)


@dataclass(frozen=True)
class DynamicsConfig:
    mass: float = 20.0
    yaw_inertia: float = 1.2
    steering_inertia: float = 0.05
    drive_force: float = 50.0
    brake_force: float = 70.0
    wheel_force: float = 35.0
    yaw_torque: float = 3.0
    steering_torque: float = 1.0
    longitudinal_drag: Drag = field(default_factory=Drag)
    yaw_drag: Drag = field(default_factory=lambda: Drag(.08, .5, .1))
    steering_drag: Drag = field(default_factory=lambda: Drag(.02, .3, 0))
    steering_spring: float = .8
    steering_speed_spring: float = .6
    velocity_time_constant: float = .3
    yaw_time_constant: float = .25
    steering_frequency: float = 6.0
    steering_damping: float = 1.0
    drive_acceleration: float = .9
    brake_acceleration: float = 1.2
    yaw_acceleration: float = 2.5
    steering_acceleration: float = 6.0
    linear_jerk: float = 3.0
    yaw_jerk: float = 8.0
    steering_jerk: float = 40.0
    friction: float = .6
    gravity: float = 9.81
    lateral_acceleration: float = 1.2
    substep: float = .02
    timeout: float = .3
    stop_epsilon: float = 1e-4

    def __post_init__(self):
        nonnegative = {"drive_force", "brake_force", "wheel_force", "yaw_torque", "steering_torque",
                       "steering_spring", "steering_speed_spring", "friction"}
        for name, value in vars(self).items():
            if not isinstance(value, Drag):
                positive(value, name, zero=name in nonnegative)


@dataclass(frozen=True)
class ChassisState:
    """Measured plant state; pose remains graphmap.Pose, never a second pose type."""
    pose: Pose
    velocity: Twist2D = field(default_factory=Twist2D)
    steering: float = 0.0
    steering_rate: float = 0.0
    acceleration: float = 0.0
    yaw_acceleration: float = 0.0
    steering_acceleration: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        planar_state(self.pose)
        if not np.isfinite([self.steering, self.steering_rate, self.acceleration, self.yaw_acceleration,
                            self.steering_acceleration, self.timestamp]).all():
            raise ValueError("chassis state must be finite")


@dataclass(frozen=True)
class DynamicsResult:
    state: ChassisState
    force: float
    torque: float
    lateral_acceleration: float
    tire_utilization: float
    flags: tuple[str, ...]


class ChassisDynamics:
    """Stateless plant. Explicit state makes manual/autonomous handoff lossless.

    Internal arrays end in [x,y,yaw,v,w,steer,steer_rate,a,yaw_a,steer_a].
    Both public step and DWA use integrate(), including identical substeps.
    """
    def __init__(self, model: DifferentialDrive | Ackermann | None = None,
                 config: DynamicsConfig | None = None):
        self.model = model or DifferentialDrive()
        if not isinstance(self.model, (DifferentialDrive, Ackermann)):
            raise TypeError("unsupported kinematic model")
        self.config = config or DynamicsConfig()

    def pack(self, state: ChassisState) -> np.ndarray:
        m = self.model
        v, w = state.velocity.linear, state.velocity.angular
        if not -m.max_reverse_speed - 1e-8 <= v <= m.max_speed + 1e-8 or abs(w) > m.max_yaw_rate + 1e-8:
            raise ValueError("measured velocity outside model limits")
        if isinstance(m, Ackermann):
            if abs(state.steering) > m.max_steering + 1e-8 or abs(state.steering_rate) > m.max_steering_rate + 1e-8:
                raise ValueError("measured steering outside model limits")
            if not np.isclose(w, m.yaw_rate(v, state.steering), atol=1e-8):
                raise ValueError("Ackermann velocity must agree with measured steering")
        elif abs(v) + abs(w) * m.track / 2 > m.max_wheel_speed + 1e-8:
            raise ValueError("measured wheel speed outside model limits")
        return np.array([*planar_state(state.pose), v, w, state.steering, state.steering_rate,
                         state.acceleration, state.yaw_acceleration, state.steering_acceleration])

    def unpack(self, values, source: ChassisState, dt) -> ChassisState:
        return ChassisState(pose2d(*values[:3], frame=source.pose.wrd_frame, body=source.pose.ego_frame),
                            Twist2D(*values[3:5]), *values[5:], timestamp=source.timestamp + dt)

    def step(self, state: ChassisState, command: ModelCommand, dt: float,
             *, command_age: float = 0.0) -> DynamicsResult:
        positive(dt, "dt")
        positive(command_age, "command_age", zero=True)
        values = self.pack(state)[None]
        target = self.model.encode_command(command)[None]
        flags = set()
        count = max(1, int(np.ceil(dt / self.config.substep - 1e-10)))
        h = dt / count
        for i in range(count):
            mode = command.mode
            if command_age + i * h >= self.config.timeout and mode != "estop":
                mode = "soft_stop"
                flags.add("command_timeout")
            values, force, torque, current_flags = self._substep(values, target, h, mode)
            flags.update(current_flags)
        lateral = float(values[0, 3] * values[0, 4])
        if isinstance(self.model, DifferentialDrive):
            utilization = 2 * np.hypot(abs(force[0]) / 2 + abs(torque[0]) / self.model.track,
                                       self.config.mass * lateral / 2)
        else:
            utilization = np.hypot(force[0], self.config.mass * lateral)
        budget = self.config.friction * self.config.mass * self.config.gravity
        utilization = float(utilization / budget) if budget else (np.inf if utilization else 0.)
        if utilization > 1 + 1e-6:
            flags.add("tire_envelope_exceeded")
        return DynamicsResult(self.unpack(values[0], state, dt), float(force[0]), float(torque[0]),
                              lateral, utilization, tuple(sorted(flags)))

    def integrate(self, states, commands, dt: float, *, mode="track"):
        """Batched prediction without watchdog expiry (commands are refreshed)."""
        positive(dt, "dt")
        values = np.asarray(states, dtype=float).copy()
        count = max(1, int(np.ceil(dt / self.config.substep - 1e-10)))
        for _ in range(count):
            values, _, _, _ = self._substep(values, commands, dt / count, mode)
        return values

    def _substep(self, old, commands, h, mode):
        c, m = self.config, self.model
        diff = isinstance(m, DifferentialDrive)
        s = old.copy()
        flags = set()
        v, w, delta, rate = (old[:, i] for i in (3, 4, 5, 6))
        vr, turn = m.references(commands)
        if mode in ("soft_stop", "estop"):
            vr, turn = np.zeros_like(v), np.zeros_like(v)
        ay = min(c.lateral_acceleration, c.friction * c.gravity)
        guard = np.maximum(np.abs(v), np.abs(vr))
        spring = c.steering_spring + c.steering_speed_spring * v * v
        if diff:
            wlimit = ay / np.maximum(guard, 1e-12)
            turn = np.clip(turn, -wlimit, wlimit)
        else:
            limit = np.minimum(m.max_steering, np.arctan(m.wheelbase * ay / np.maximum(guard ** 2, 1e-12)))
            limit = np.minimum(limit, np.arctan(m.wheelbase * m.max_yaw_rate / np.maximum(guard, 1e-12)))
            limit = np.minimum(limit, c.steering_torque / np.maximum(c.steering_spring + c.steering_speed_spring * guard**2, 1e-12))
            turn = np.clip(turn, -limit, limit)
            curvature = np.maximum(np.abs(np.tan(delta) / m.wheelbase), 1e-12)
            cap = np.minimum(np.sqrt(ay / curvature), m.max_yaw_rate / curvature)
            vr = np.clip(vr, -cap, cap)
        if mode == "track":
            raw = np.asarray(commands)
            raw_v = raw.mean(axis=-1) if diff else raw[:, 0]
            raw_turn = (raw[:, 1] - raw[:, 0]) / m.track if diff else raw[:, 1]
            if np.any(abs(raw_v - vr) > 1e-9) or np.any(abs(raw_turn - turn) > 1e-9):
                flags.add("reference_limited")
        a_goal = (vr - v) / c.velocity_time_constant
        limit = np.where(v * a_goal < 0, c.brake_acceleration, c.drive_acceleration)
        accel = np.clip(np.clip(a_goal, -limit, limit), old[:, 7] - c.linear_jerk * h, old[:, 7] + c.linear_jerk * h)
        force = _effort(0., v, accel, c.mass, 0., c.longitudinal_drag, h)
        if diff:
            aa_goal = np.clip((turn - w) / c.yaw_time_constant, -c.yaw_acceleration, c.yaw_acceleration)
            aa = np.clip(aa_goal, old[:, 8] - c.yaw_jerk * h, old[:, 8] + c.yaw_jerk * h)
            torque = _effort(0., w, aa, c.yaw_inertia, 0., c.yaw_drag, h)
            torque_limit = c.yaw_torque
        else:
            aa_goal = c.steering_frequency**2 * (turn - delta) - 2 * c.steering_damping * c.steering_frequency * rate
            aa_goal = np.clip(aa_goal, -c.steering_acceleration, c.steering_acceleration)
            aa = np.clip(aa_goal, old[:, 9] - c.steering_jerk * h, old[:, 9] + c.steering_jerk * h)
            torque = _effort(delta, rate, aa, c.steering_inertia, spring, c.steering_drag, h)
            torque_limit = c.steering_torque
        force_request, torque_request = force.copy(), torque.copy()
        lo = np.where(v > c.stop_epsilon, -c.brake_force, -c.drive_force)
        hi = np.where(v < -c.stop_epsilon, c.brake_force, c.drive_force)
        force, torque = np.clip(force, lo, hi), np.clip(torque, -torque_limit, torque_limit)
        if mode == "coast":
            force, torque = np.zeros_like(v), np.zeros_like(v)
        elif mode == "estop":
            force = -np.sign(v) * c.brake_force
            torque = -np.sign(w) * c.yaw_torque if diff else np.clip(spring * delta - rate, -torque_limit, torque_limit)
        available = np.sqrt(np.maximum(0., (c.friction * c.mass * c.gravity)**2 - (c.mass * v * w)**2))
        if diff:
            demand = np.abs(force) / 2 + np.abs(torque) / m.track
            scale = np.minimum(1., np.minimum(c.wheel_force, available / 2) / np.maximum(demand, 1e-12))
            force, torque = force * scale, torque * scale
        else:
            force = np.clip(force, -available, available)
        if mode in ("track", "soft_stop") and (np.any(abs(force - force_request) > 1e-8) or np.any(abs(torque - torque_request) > 1e-8)):
            flags.add("effort_limited")
        # Exact inverse lets unsaturated servos bypass the nonlinear solve.
        s[:, 3] = v + h * accel
        saturated = np.abs(force - force_request) > 1e-10
        if saturated.any():
            s[saturated, 3] = implicit_axis(0., v[saturated], force[saturated], c.mass, 0., c.longitudinal_drag, h)[1]
        if diff:
            s[:, 4] = w + h * aa
            saturated = np.abs(torque - torque_request) > 1e-10
            if saturated.any():
                s[saturated, 4] = implicit_axis(0., w[saturated], torque[saturated], c.yaw_inertia, 0., c.yaw_drag, h)[1]
        else:
            new_rate = rate + h * aa
            saturated = np.abs(torque - torque_request) > 1e-10
            if saturated.any():
                new_rate[saturated] = implicit_axis(delta[saturated], rate[saturated], torque[saturated],
                                                     c.steering_inertia, spring[saturated], c.steering_drag, h)[1]
            s[:, 6] = np.clip(new_rate, -m.max_steering_rate, m.max_steering_rate)
            s[:, 5] = np.clip(delta + h * s[:, 6], -m.max_steering, m.max_steering)
            contact = (np.abs(s[:, 5]) >= m.max_steering) & (s[:, 5] * s[:, 6] > 0)
            s[contact, 6] = 0.
            if contact.any() or np.any(new_rate != s[:, 6]):
                flags.add("steering_stop_reaction")
        if mode == "estop":
            s[:, 3] = np.where(v * s[:, 3] < 0, 0., s[:, 3])
            if diff:
                s[:, 4] = np.where(w * s[:, 4] < 0, 0., s[:, 4])
        rest = (mode != "track") | (vr == 0)
        s[:, 3] = np.where(rest & (np.abs(s[:, 3]) < c.stop_epsilon) & (np.abs((s[:, 3] - v) / h) < .02), 0., s[:, 3])
        if diff:
            s[:, 4] = np.where((turn == 0) & (np.abs(s[:, 4]) < c.stop_epsilon), 0., s[:, 4])
        before = s[:, 3:5].copy()
        s[:, 3] = np.clip(s[:, 3], -m.max_reverse_speed, m.max_speed)
        if diff:
            s[:, 4] = np.clip(s[:, 4], -m.max_yaw_rate, m.max_yaw_rate)
            scale = np.minimum(1., m.max_wheel_speed / np.maximum(abs(s[:, 3]) + m.track * abs(s[:, 4]) / 2, 1e-12))
            s[:, 3:5] *= scale[:, None]
        else:
            s[:, 4] = m.yaw_rate(s[:, 3], s[:, 5])
            scale = np.minimum(1., m.max_yaw_rate / np.maximum(np.abs(s[:, 4]), 1e-12))
            s[:, 3:5] *= scale[:, None]
        if np.any(abs(s[:, 3] - before[:, 0]) > 1e-10) or (diff and np.any(abs(s[:, 4] - before[:, 1]) > 1e-10)):
            flags.add("speed_hard_projection")
        if np.any(np.abs(s[:, 3] * s[:, 4]) > ay + 1e-8):
            flags.add("lateral_envelope_exceeded")
        s[:, 7:9] = (s[:, 3:5] - old[:, 3:5]) / h
        s[:, 9] = (s[:, 6] - old[:, 6]) / h
        if np.any(abs(s[:, 7] - old[:, 7]) > c.linear_jerk * h + 1e-7):
            flags.add("linear_jerk_relaxed")
        axis, jerk = (8, c.yaw_jerk) if diff else (9, c.steering_jerk)
        if np.any(abs(s[:, axis] - old[:, axis]) > jerk * h + 1e-7):
            flags.add("angular_jerk_relaxed")
        mid = (old[:, 3:5] + s[:, 3:5]) / 2
        s[:, :3] = advance(old[:, :3], mid[:, 0], mid[:, 1], h)
        return s, force, torque, flags
