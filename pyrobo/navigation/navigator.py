"""Global task lifecycle and reversible progress, independent of local algorithm."""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from graphmap.pose import Pose, PoseArray

from ..core import MotionPolicy, MotionRequest, NavigationStatus, Path, RobotObservation
from ..control.interfaces import LocalMotionOptimizer
from ..planning.interfaces import GlobalPlanner
from pyrobo.core.state import ChassisState
from pyrobo.core.commands import ModelCommand
from ..geometry.cache import SnapshotIndex
from pyrobo.core.obstacles import PointCloud
from ..safety import SafetyConfig, SafetyGuard
from pyrobo.core.obstacles import Bounds2D
from pyrobo.core.state import Twist2D
from pyrobo.core.geometry import path_poses, planar_state, pose2d, positive, wrap_angle
from .defaults import components
from .tracking import PathTracker


@dataclass(frozen=True)
class NavigationInput:
    timestamp: float
    pose: Pose
    velocity: Twist2D
    goal: Pose
    obstacles: PointCloud
    bounds: Bounds2D
    dt: float = .1
    chassis: ChassisState | None = None


@dataclass(frozen=True)
class NavigationOutput:
    timestamp: float
    command: ModelCommand
    status: NavigationStatus
    safe: bool
    path: PoseArray
    trajectory: PoseArray
    diagnostics: dict = field(default_factory=dict)


class Navigator:
    """Observe during any mode; request local motion only when AUTO owns control.

    `step(NavigationInput)` remains the convenience, guarded legacy composition.
    Inject GlobalPlanner / LocalMotionOptimizer implementations to replace algorithms.
    """
    def __init__(self, *, footprint=None, planner=None, controller=None, model=None, dynamics=None,
                 global_planner: GlobalPlanner | None = None, local_optimizer: LocalMotionOptimizer | None = None,
                 command_space=None, settings=None, index=None, safety=None, point_radius=.03, z_range=None,
                 max_cloud_age=.3, retry_interval=.5, stall_timeout=4.):
        for name, value in (('point_radius', point_radius), ('max_cloud_age', max_cloud_age)):
            positive(value, name, zero=True)
        positive(retry_interval, 'retry_interval')
        positive(stall_timeout, 'stall_timeout')
        self.index = index or SnapshotIndex(point_radius=point_radius, z_range=z_range)
        (self.footprint, self.model, self.command_space, self.dynamics, self.planner,
         self.optimizer, self.settings) = components(footprint=footprint, planner=planner, controller=controller,
            model=model, dynamics=dynamics, global_planner=global_planner, local_optimizer=local_optimizer,
            command_space=command_space, settings=settings, index=self.index, max_cloud_age=max_cloud_age)
        self.controller = self.optimizer  # compatibility alias, never needed by task logic
        self.safety = safety or SafetyGuard(self.dynamics, self.command_space, self.footprint, index=self.index,
                                            config=SafetyConfig(max_cloud_age=max_cloud_age))
        self.point_radius, self.z_range = point_radius, z_range
        self.max_cloud_age, self.retry_interval, self.stall_timeout = max_cloud_age, retry_interval, stall_timeout
        self.reset()

    def reset(self):
        for component in (self.planner, self.optimizer, self.index):
            reset = getattr(component, 'reset', None)
            if reset is not None:
                reset()
        self._path = np.empty((0, 2))
        self._goal_pose, self._signature = None, None
        self._observation, self._last_timestamp = None, None
        self._last_plan = -np.inf
        self._progress_position, self._progress_time = None, None
        self.tracker, self.progress = PathTracker(), None
        self._planning = {}
        self._planner_status, self._replanned = 'cached', False
        self.status = NavigationStatus.NO_PATH
        self.last_motion = None

    @property
    def goal(self):
        return self._goal_pose

    @property
    def path(self):
        o = self._observation
        frame = o.frame if o else (self.goal.wrd_frame if self.goal is not None else 'world')
        body = o.chassis.pose.ego_frame if o else 'robot'
        return Path(path_poses(self._path, frame, body))

    def set_goal(self, goal: Pose):
        planar_state(goal)
        self._goal_pose = goal

    def _at_goal(self, observation):
        if self.goal is None:
            return False, False
        current, goal = planar_state(observation.chassis.pose), planar_state(self.goal)
        return (np.linalg.norm(current[:2] - goal[:2]) <= self.settings.goal_tolerance,
                abs(float(wrap_angle(current[2] - goal[2]))) <= self.settings.yaw_tolerance)

    def observe(self, observation: RobotObservation, *, goal: Pose | None = None):
        """Update progress and invalidate unsafe paths without solving local motion."""
        if goal is not None:
            self.set_goal(goal)
        o = observation
        if self.goal is not None and self.goal.wrd_frame != o.frame:
            raise ValueError('robot, goal and cloud must share an explicit world frame')
        if o.obstacles.timestamp is not None:
            age = o.timestamp - o.obstacles.timestamp
            if age < -1e-9 or age > self.max_cloud_age + 1e-9:
                raise ValueError('cloud timestamp is in the future or exceeds max_cloud_age')
        if self._observation is not o and self._last_timestamp is not None and o.timestamp <= self._last_timestamp:
            raise ValueError('timestamps must strictly increase; call reset() before replay')
        current = planar_state(o.chassis.pose)
        signature = (*planar_state(self.goal), o.frame, o.bounds) if self.goal is not None else None
        if self._observation is o and signature == self._signature:
            return self.progress
        self._replanned, self._planner_status, self._planning = False, 'cached', {}
        if signature != self._signature:
            reset = getattr(self.planner, 'reset', None)
            if reset:
                reset()
            self._path = np.empty((0, 2))
            self.tracker.reset()
            self._last_plan = -np.inf
            self._progress_position, self._progress_time = current[:2].copy(), o.timestamp
            self._signature = signature
        self._observation, self._last_timestamp = o, o.timestamp
        self.progress = self.tracker.update(o.chassis.pose, self.settings.lookahead) if len(self._path) else None
        at_position, at_heading = self._at_goal(o)
        if self.progress is not None and (not at_position or (not self.command_space.can_spin and not at_heading)):
            if not self.index.get(o.obstacles).path_free(self.progress.remaining.position[:, :2], self.footprint,
                                                        o.bounds, goal_yaw=planar_state(self.goal)[2]):
                self._path, self.progress, self._last_plan = np.empty((0, 2)), None, -np.inf
        return self.progress

    def suspend_tracking(self, observation):
        """Keep the stall clock from treating deliberate manual/paused motion as failure."""
        self._progress_position = planar_state(observation.chassis.pose)[:2].copy()
        self._progress_time = observation.timestamp

    def _plan(self, o):
        goal, current = planar_state(self.goal), planar_state(o.chassis.pose)
        planning_goal, approach = self.goal, None
        distance = self.command_space.goal_approach_distance(self.settings.lookahead)
        if distance > 0 and self.settings.goal_approach_distance is not None:
            distance = self.settings.goal_approach_distance
        if distance > 0:
            direction = np.array([np.cos(goal[2]), np.sin(goal[2])])
            if self.settings.goal_approach_distance is None:
                world = self.index.get(o.obstacles)
                body = getattr(self.planner, 'safety_footprint', self.footprint)
                if not world.rectangle_segment_free(goal[:2]-distance*direction, goal[:2], body, o.bounds):
                    # For fixed goal heading, shorter entry segments are nested
                    # subsets of the same swept rectangle. Find available space
                    # without weakening the body margin or using a disk proxy.
                    lo, hi = 0., distance
                    for _ in range(14):
                        middle = (lo+hi)/2
                        if world.rectangle_segment_free(goal[:2]-middle*direction, goal[:2], body, o.bounds):
                            lo = middle
                        else:
                            hi = middle
                    fitted = max(0., lo-self.footprint.margin)
                    if fitted >= max(self.settings.goal_tolerance, self.footprint.length/2):
                        distance = fitted
            offset = current[:2] - goal[:2]
            on_approach = (-distance <= offset @ direction <= 0 and
                abs(direction[0]*offset[1] - direction[1]*offset[0]) < self.settings.goal_tolerance and
                abs(float(wrap_angle(current[2] - goal[2]))) < np.pi/2)
            if self.settings.goal_approach_distance is None:
                # Once inside the terminal approach region, do not reintroduce
                # an entry point behind the car. Let the planner connect the
                # measured pose to the goal (with all normal collision checks).
                # An explicitly requested entry length remains authoritative.
                facing = np.array([np.cos(current[2]), np.sin(current[2])])
                on_approach |= (np.linalg.norm(offset) < distance and offset @ direction <= 0
                               and offset @ facing < 0 and facing @ direction > 0)
            if not on_approach:
                approach = goal[:2] - distance*direction
                planning_goal = pose2d(*approach, goal[2], frame=o.frame)
        world = self.index.get(o.obstacles)
        if approach is not None and not world.rectangle_segment_free(approach, goal[:2],
                getattr(self.planner, 'safety_footprint', self.footprint), o.bounds):
            plan = None
        else:
            plan = self.planner.plan(o.chassis.pose, planning_goal, world, o.bounds)
        self._last_plan, self._replanned = o.timestamp, True
        self._planner_status = plan.status if plan else 'invalid_goal_approach'
        if plan is not None:
            self._planning = {**plan.diagnostics, 'elapsed': plan.elapsed, 'extensions': plan.extensions,
                              'goal_approach_distance': distance}
            poses = plan.path.poses if isinstance(plan.path, Path) else plan.path
            Path(poses)
            if poses.wrd_frame != o.frame:
                raise ValueError('planner output frame must match observation')
            self._path = poses.position[:, :2].copy() if plan.success else np.empty((0, 2))
            if plan.success and approach is not None:
                self._path = np.vstack((self._path, goal[:2]))
        if len(self._path):
            self.tracker.set_path(self.path.poses)
            self.progress = self.tracker.update(o.chassis.pose, self.settings.lookahead)

    def make_motion_request(self, observation: RobotObservation, *, goal=None):
        self.observe(observation, goal=goal)
        o, cfg = observation, self.settings
        at_position, at_heading = self._at_goal(o)
        current = planar_state(o.chassis.pose)
        if self._progress_position is None or np.linalg.norm(current[:2] - self._progress_position) > .15 or at_position:
            self.suspend_tracking(o)
        elif o.timestamp - self._progress_time > self.stall_timeout:
            self._path, self.progress, self._last_plan = np.empty((0, 2)), None, -np.inf
            self._progress_time = o.timestamp
        if self.goal is None:
            self.status = NavigationStatus.NO_PATH
            return MotionRequest(MotionPolicy.STOP, o.timestamp)
        if not at_position or (not self.command_space.can_spin and not at_heading):
            if not len(self._path) and o.timestamp - self._last_plan >= self.retry_interval:
                self._plan(o)
        if at_position and at_heading:
            stopped = max(abs(o.chassis.velocity.linear), abs(o.chassis.velocity.angular)) < 1e-3
            self.status = NavigationStatus.ARRIVED if stopped else NavigationStatus.BRAKING
            return MotionRequest(MotionPolicy.STOP, o.timestamp)
        if at_position and self.command_space.can_spin:
            local_path = Path(path_poses(planar_state(self.goal)[None, :2], o.frame, o.chassis.pose.ego_frame))
            progress = None
            self.status = NavigationStatus.ALIGNING
        elif len(self._path):
            local_path, progress = self.path, self.progress
            self.status = NavigationStatus.NAVIGATING
        else:
            self.status = NavigationStatus.NO_PATH
            return MotionRequest(MotionPolicy.STOP, o.timestamp)
        return MotionRequest(MotionPolicy.NAVIGATION, o.timestamp, path=local_path, goal=self.goal,
                             local_target=progress.target if progress else None,
                             remaining_path=Path(progress.remaining) if progress else None,
                             remaining_distance=progress.remaining_distance if progress else None)

    def navigation_output(self, observation, motion, *, elapsed=0.):
        self.last_motion = motion
        status = self.status
        d, p = motion.diagnostics, self.progress
        if not motion.safe:
            status = NavigationStatus.EMERGENCY_STOP
        elif d.feasible_count == 0 and status in (NavigationStatus.NAVIGATING, NavigationStatus.ALIGNING):
            status = NavigationStatus.BLOCKED
        trajectory = motion.trajectory.poses if motion.trajectory else path_poses([], observation.frame)
        return NavigationOutput(observation.timestamp, motion.command, status, motion.safe, self.path.poses, trajectory,
              {'elapsed': elapsed, 'replanned': self._replanned, 'planner_status': self._planner_status,
               'clearance': d.clearance_m, 'candidates': d.candidate_count, 'feasible_candidates': d.feasible_count,
               'planner': type(self.planner).__name__, 'planning': self._planning,
               'controller_mode': d.extra.get('controller_mode', 'braking'),
               'progress_m': p.distance if p else None, 'remaining_m': p.remaining_distance if p else None,
               'target': p.target if p else self.goal, 'segment': p.segment if p else None,
               'obstacle_points': len(self.index.get(observation.obstacles).xy),
               'motion_status': motion.status, 'safety_assessment': motion.safety_assessment,
               'safety_action': motion.safety_action, 'reason': d.reason, **d.extra})

    def step(self, observation: NavigationInput) -> NavigationOutput:
        started = perf_counter()
        o = observation
        state = o.chassis
        if state is None:
            if not self.command_space.can_spin:
                raise ValueError('Ackermann requires measured ChassisState, including steering at rest')
            state = ChassisState(o.pose, o.velocity, timestamp=o.timestamp)
        if state.pose.wrd_frame != o.pose.wrd_frame or not np.allclose(state.pose.matrix, o.pose.matrix) or state.velocity != o.velocity:
            raise ValueError('ChassisState must match observed pose and velocity')
        if abs(state.timestamp - o.timestamp) > 1e-8:
            raise ValueError('chassis timestamp must match observation')
        current = RobotObservation(state, o.obstacles, o.bounds, o.dt, o.timestamp)
        request = self.make_motion_request(current, goal=o.goal)
        motion = self.optimizer.optimize(current, request)
        motion = self.safety.enforce(current, motion, request)
        return self.navigation_output(current, motion, elapsed=perf_counter()-started)
