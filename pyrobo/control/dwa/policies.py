"""Request-specific search constraints, separate from the shared optimizer."""
from dataclasses import dataclass
import numpy as np

from ...core import Footprint, MotionPolicy
from pyrobo.core.geometry import planar_state, wrap_angle
from .path import PathTracker, path_lookahead


@dataclass(slots=True)
class PolicyContext:
    request: object
    state: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    anchors: np.ndarray
    desired: np.ndarray
    mode: str = 'following'
    target: np.ndarray | None = None
    remaining: np.ndarray | None = None
    goal: np.ndarray | None = None
    remaining_distance: float = 0.
    distance_to_goal: float = 0.
    signed_error: float = 0.
    direction: float = 0.
    turn_direction: float = 0.
    previous: np.ndarray | None = None
    preferred_turn_sign: float = 0.
    rollout_target: np.ndarray | None = None
    recovery_origin: np.ndarray | None = None
    recovery_distance: float = 0.


class NavigationPolicy:
    def __init__(self, footprint=None):
        self.footprint = footprint or Footprint()
        self.reset()

    def reset(self):
        self.tracker = PathTracker()
        self.turn_active, self.turn_direction = False, 0.
        self.stall_pose = self.recovery_origin = self.recovery_site = self.goal_pose = None
        self.stall_since = self.last_time = None
        self.recovery_until, self.recovery_attempts = -np.inf, 0

    def recovery(self, observation, state, goal, mode, config):
        now = observation.timestamp
        if (self.goal_pose is None or not np.array_equal(goal, self.goal_pose)
                or self.last_time is None or now-self.last_time > config.prediction_time):
            self.stall_pose, self.stall_since = state.copy(), now
            self.recovery_until, self.recovery_attempts, self.recovery_site = -np.inf, 0, None
        self.goal_pose, self.last_time = goal.copy(), now
        limit = self.footprint.length/2
        if self.recovery_site is not None and np.linalg.norm(state[:2]-self.recovery_site) > self.footprint.length:
            self.recovery_attempts, self.recovery_site = 0, None
        if now < self.recovery_until:
            if (mode != 'following' or not config.allow_reverse or
                    np.linalg.norm(state[:2]-self.recovery_origin) >= .8*limit):
                self.recovery_until = -np.inf
                self.stall_pose, self.stall_since = state.copy(), now
            else:
                return True, limit
        elif self.recovery_origin is not None:
            self.recovery_origin = None
            self.stall_pose, self.stall_since = state.copy(), now
        moved = (np.linalg.norm(state[:2]-self.stall_pose[:2]) > .05 or
                 abs(float(wrap_angle(state[2]-self.stall_pose[2]))) > .15)
        if moved or mode != 'following':
            self.stall_pose, self.stall_since = state.copy(), now
        stopped = max(abs(observation.chassis.velocity.linear), abs(observation.chassis.velocity.angular)) < .02
        if (mode == 'following' and config.allow_reverse and stopped and self.recovery_attempts < 3
                and now-self.stall_since >= config.prediction_time):
            self.recovery_origin = state[:2].copy()
            self.recovery_site = state[:2].copy() if self.recovery_site is None else self.recovery_site
            self.recovery_until = now+2*config.prediction_time
            self.recovery_attempts += 1
            return True, limit
        return False, limit

    def prepare(self, observation, request, space, config, codec, previous):
        chassis = observation.chassis
        state = planar_state(chassis.pose)
        goal = planar_state(request.goal)
        xy = request.path.xy
        if not np.array_equal(self.tracker.xy, xy):
            self.tracker.set_path(request.path.poses)
            # A replan can change which side the next target is on. Retaining
            # the old spin direction can otherwise request an extra full turn.
            self.turn_active, self.turn_direction = False, 0.
        if request.local_target is None:
            progress = self.tracker.update(chassis.pose, config.lookahead)
            target, remaining = progress.target.translation[:2], progress.remaining.position[:, :2]
            remaining_distance = progress.remaining_distance
        else:
            target = request.local_target.translation[:2]
            remaining = request.remaining_path.xy if request.remaining_path is not None else xy
            remaining_distance = request.remaining_distance or 0.
        distance = np.linalg.norm(state[:2] - goal[:2])
        near = distance <= config.goal_tolerance
        direction = np.arctan2(target[1] - state[1], target[0] - state[0])
        error = float(wrap_angle(direction - state[2]))
        if near and not space.can_spin:
            # At the goal, a position carrot may be behind the rear axle and
            # request the opposite steering direction. Align to the goal pose
            # while using the allowed positional tolerance for a short arc.
            direction = goal[2]
            error = float(wrap_angle(direction-state[2]))
        if space.can_spin and not near:
            if self.turn_active and abs(error) <= config.turn_exit_angle and abs(chassis.velocity.angular) < .05:
                self.turn_active = False
            if not self.turn_active and abs(error) >= config.turn_in_place_angle and abs(chassis.velocity.linear) < .05:
                self.turn_active = True
                self.turn_direction = float(np.sign(error) or 1)
            if self.turn_active and abs(error) < np.pi/2 and error:
                self.turn_direction = float(np.sign(error))
        elif near:
            self.turn_active = False
        mode = ('pose_aligning' if near and not space.can_spin else
                'aligning' if near and space.can_spin else
                'turn_braking' if self.turn_active and abs(error) <= config.turn_exit_angle else
                'turning' if self.turn_active else 'following')
        recovering, recovery_distance = self.recovery(observation, state, goal, mode, config)
        if recovering:
            mode = 'recovering'
        lo, hi = space.bounds()
        speed_limit = hi[0] * max(0. if space.can_spin else .25,
                                  np.cos(min(abs(error)/config.heading_slow_angle, 1)*np.pi/2))
        lo[0] = lo[0] if config.allow_reverse else 0.
        # A car needs translational motion to turn. Heading error regulates its
        # preferred speed, but must not remove faster, verified turning arcs
        # from the model's search space and trap it in a zero-speed minimum.
        if space.can_spin:
            hi[0] = speed_limit
        if mode == 'pose_aligning':
            hi[0] = min(hi[0], config.goal_tolerance/config.prediction_time)
            lo[0] = max(lo[0], -hi[0])
        desired_turn = config.turn_gain * error
        if mode == 'turning':
            desired_turn = self.turn_direction * min(hi[1], max(config.min_turn_yaw_rate, config.turn_gain*abs(error)))
        elif mode == 'turn_braking':
            desired_turn = 0.
        elif mode == 'aligning':
            # Supply a sub-grid rate that closes the angular error over the
            # rollout horizon. A fixed gain alone can overshoot the yaw tolerance
            # for every nonzero sample, leaving a differential base stopped.
            desired_turn = config.turn_gain * float(wrap_angle(goal[2] - state[2])) / config.prediction_time
        elif not space.can_spin:
            desired_turn = space.turn_for_heading(error, max(np.linalg.norm(target - state[:2]), .1))
        if mode in ('turning', 'turn_braking', 'aligning'):
            lo[0] = hi[0] = 0.
        if recovering:
            cap = recovery_distance/config.prediction_time
            lo[0], hi[0] = max(lo[0], -cap), min(hi[0], cap)
            speed_limit, desired_turn = lo[0], 0.
        desired = np.array([min(speed_limit, hi[0]), np.clip(desired_turn, lo[1], hi[1])])
        anchors = np.array([desired, [0., 0.], space.state_coordinates(codec.pack(chassis))])
        if previous is not None:
            anchors = np.vstack((anchors, previous))
        # Steering preview and terminal scoring have different time horizons.
        # Chasing a fixed carrot at the end of every rollout silently caps cruise
        # speed at lookahead / prediction_time, even in a wide straight corridor.
        rollout_target = target
        preview = desired[0]*config.prediction_time
        if mode == 'following' and preview > config.lookahead and len(remaining):
            rollout_target, _ = path_lookahead(remaining, remaining[0], preview)
        return PolicyContext(request, state, lo, hi, anchors, desired, mode, target, remaining, goal,
                             remaining_distance, distance, error, direction, self.turn_direction, previous,
                             rollout_target=rollout_target,
                             recovery_origin=self.recovery_origin if recovering else None,
                             recovery_distance=recovery_distance)


class TeleopPolicy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.turn_sign, self.hold_until = 0., -np.inf

    def prepare(self, observation, request, space, config, codec, previous):
        desired = space.desired_coordinates(request)
        lo, hi = space.bounds()
        lo[0], hi[0] = max(lo[0], min(0., desired[0])), min(hi[0], max(0., desired[0]))
        fixed = request.policy == MotionPolicy.TELEOP_STOP or desired[0] == 0
        deviation = 0. if fixed else request.max_turn_deviation
        lo[1], hi[1] = max(lo[1], desired[1] - deviation), min(hi[1], desired[1] + deviation)
        # No zero-turn anchor in strict STOP: it would silently change intent.
        anchors = np.array([desired, [0., desired[1]]])
        if previous is not None:
            anchors = np.vstack((anchors, previous))
        return PolicyContext(request, planar_state(observation.chassis.pose), lo, hi, anchors, desired,
                             mode=request.policy.value, previous=previous,
                             preferred_turn_sign=self.turn_sign if observation.timestamp < self.hold_until else 0.)

    def selected(self, timestamp, search, desired):
        deviation = search[1] - desired[1]
        if abs(deviation) > .08:
            sign = float(np.sign(deviation))
            if sign != self.turn_sign or timestamp >= self.hold_until:
                self.turn_sign, self.hold_until = sign, timestamp + .6

    def trajectory_mask(self, context, poses, times, space):
        if context.request.policy != MotionPolicy.TELEOP_AVOID:
            return np.ones(len(poses), dtype=bool)
        yaw = context.state[2]
        offset = poses[..., :2] - context.state[:2]
        lateral = -np.sin(yaw)*offset[..., 0] + np.cos(yaw)*offset[..., 1]
        angle = space.yaw_reference(context.desired[None])[0] * times
        reference = context.desired[0] * times * np.sinc(angle/(2*np.pi)) * np.sin(angle/2)
        lateral_ok = abs(lateral-reference) <= context.request.max_lateral_deviation + 1e-9
        heading_ok = abs(wrap_angle(poses[..., 2]-yaw-angle)) <= context.request.max_heading_deviation + 1e-9
        return np.all(lateral_ok & heading_ok, axis=1)
