"""One current observation in, one current control decision out."""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from graphmap.pose import Pose, PoseArray

from .control import DWAConfig, DynamicWindow
from .control.dwa.path import PathTracker
from .dynamics import ChassisState, DynamicsConfig
from .kinematics import Ackermann, DifferentialDrive, ModelCommand
from .planning import RRTConnect, RRTConnectConfig, VoronoiConfig, VoronoiPlanner
from .pointcloud import PointCloud, PointObstacles
from .types import Bounds2D, Footprint, Twist2D, path_poses, planar_state, pose2d, positive, wrap_angle


@dataclass(frozen=True)
class NavigationInput:
    timestamp: float
    pose: Pose
    velocity: Twist2D
    goal: Pose
    obstacles: PointCloud
    bounds: Bounds2D
    dt: float = 0.1
    chassis: ChassisState | None = None


@dataclass(frozen=True)
class NavigationOutput:
    timestamp: float
    command: ModelCommand
    status: str
    safe: bool
    path: PoseArray
    trajectory: PoseArray
    diagnostics: dict = field(default_factory=dict)


class Navigator:
    """Stateful global planning + DWA. No global state, I/O, plotting or threads.

    History is bounded to a path, one roadmap, turn/timing/progress state and RNG.
    A PointCloud is a complete current obstacle snapshot, not a map update/delta.
    """
    def __init__(self, *, footprint: Footprint | None = None,
                 planner: RRTConnectConfig | VoronoiConfig | None = None,
                 controller: DWAConfig | None = None, point_radius: float = 0.03,
                 model: DifferentialDrive | Ackermann | None = None, dynamics: DynamicsConfig | None = None,
                 z_range: tuple[float, float] | None = None, max_cloud_age: float = 0.3,
                 retry_interval: float = 0.5, stall_timeout: float = 4.0):
        self.footprint = footprint or Footprint()
        self.controller = DynamicWindow(controller, self.footprint, model=model, dynamics=dynamics)
        self.model = self.controller.model
        if planner is None and isinstance(self.model, Ackermann):
            planner = RRTConnectConfig(clearance=.12 + .4*self.model.minimum_turn_radius, time_budget=.2)
        if isinstance(planner, VoronoiConfig):
            self.planner = VoronoiPlanner(planner, self.footprint)
        elif planner is None or isinstance(planner, RRTConnectConfig):
            self.planner = RRTConnect(planner, self.footprint)
        else:
            raise TypeError("planner must be RRTConnectConfig or VoronoiConfig")
        positive(point_radius, "point_radius", zero=True)
        positive(max_cloud_age, "max_cloud_age", zero=True)
        positive(retry_interval, "retry_interval")
        positive(stall_timeout, "stall_timeout")
        self.point_radius, self.z_range = point_radius, z_range
        self.max_cloud_age, self.retry_interval, self.stall_timeout = max_cloud_age, retry_interval, stall_timeout
        self.reset()

    def reset(self):
        self.planner.reset()
        self.controller.reset()
        self._path = np.empty((0, 2))
        self._goal = None
        self._last_timestamp = None
        self._last_plan = -np.inf
        self._progress_position = None
        self._progress_time = None
        self.tracker = PathTracker()

    def step(self, observation: NavigationInput) -> NavigationOutput:
        started = perf_counter()
        o = observation
        if not np.isfinite(o.timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and o.timestamp <= self._last_timestamp:
            raise ValueError("timestamps must strictly increase; call reset() before replay")
        current, goal = planar_state(o.pose), planar_state(o.goal)
        if o.pose.wrd_frame != o.obstacles.frame or o.goal.wrd_frame != o.obstacles.frame:
            raise ValueError("robot, goal and cloud must share an explicit world frame")
        if o.obstacles.timestamp is not None:
            age = o.timestamp - o.obstacles.timestamp
            if age < -1e-9 or age > self.max_cloud_age + 1e-9:
                raise ValueError("cloud timestamp is in the future or exceeds max_cloud_age")
        obstacles = PointObstacles(o.obstacles, point_radius=self.point_radius, z_range=self.z_range)
        self.controller._validate(o.pose, o.velocity, obstacles, o.dt, o.chassis)
        if o.chassis is not None and abs(o.chassis.timestamp - o.timestamp) > 1e-8:
            raise ValueError("chassis timestamp must match observation")
        self._last_timestamp = o.timestamp
        cfg = self.controller.config
        at_position = np.linalg.norm(current[:2] - goal[:2]) <= cfg.goal_tolerance
        at_heading = abs(float(wrap_angle(current[2] - goal[2]))) <= cfg.yaw_tolerance
        signature = (*goal, o.obstacles.frame, o.bounds)
        changed_goal = self._goal != signature
        if changed_goal:
            self.planner.reset()
            self.controller.reset()
            self._path = np.empty((0, 2))
            self._goal = signature
            self._last_plan = -np.inf
            self._progress_position, self._progress_time = current[:2].copy(), o.timestamp
            self.tracker.reset()
        if np.linalg.norm(current[:2] - self._progress_position) > 0.15 or at_position:
            self._progress_position, self._progress_time = current[:2].copy(), o.timestamp
        elif o.timestamp - self._progress_time > self.stall_timeout:
            self._path = np.empty((0, 2))
            self._last_plan = -np.inf
            self._progress_time = o.timestamp

        replanned = False
        planner_status = "cached"
        planner_diagnostics = {}
        progress = None
        diff = isinstance(self.model, DifferentialDrive)
        if not at_position or (not diff and not at_heading):
            if len(self._path):
                progress = self.tracker.update(o.pose, cfg.lookahead)
                if not obstacles.path_free(progress.remaining.position[:, :2], self.footprint,
                                           o.bounds, goal_yaw=goal[2]):
                    self._path = np.empty((0, 2))
                    progress = None
                    self._last_plan = -np.inf
            if len(self._path) == 0 and o.timestamp - self._last_plan >= self.retry_interval:
                planning_goal = o.goal
                approach = None
                if not diff:
                    # Reserve a straight final approach so yaw is attained in
                    # motion. The global XY route is still not kinodynamic.
                    direction = np.array([np.cos(goal[2]), np.sin(goal[2])])
                    distance = max(cfg.lookahead, self.model.wheelbase / np.tan(self.model.max_steering) * 1.5)
                    staging = goal[:2] - distance * direction
                    offset = current[:2] - goal[:2]
                    on_approach = (-distance <= offset @ direction <= 0 and
                                   abs(direction[0]*offset[1] - direction[1]*offset[0]) < cfg.goal_tolerance and
                                   abs(float(wrap_angle(current[2] - goal[2]))) < cfg.yaw_tolerance)
                    if not on_approach:
                        approach = staging
                        planning_goal = pose2d(*staging, goal[2], frame=o.pose.wrd_frame)
                if approach is not None and not obstacles.rectangle_segment_free(
                        approach, goal[:2], self.planner.safety_footprint, o.bounds):
                    plan = None
                else:
                    plan = self.planner.plan(o.pose, planning_goal, obstacles, o.bounds)
                self._last_plan = o.timestamp
                replanned, planner_status = True, plan.status if plan else "invalid_goal_approach"
                if plan is not None:
                    planner_diagnostics = {**plan.diagnostics, "elapsed": plan.elapsed, "extensions": plan.extensions}
                    self._path = plan.path.position[:, :2].copy()
                    if plan.success and approach is not None:
                        self._path = np.vstack((self._path, goal[:2]))
                if len(self._path):
                    self.tracker.set_path(path_poses(self._path, o.pose.wrd_frame, o.pose.ego_frame))
                    progress = self.tracker.update(o.pose, cfg.lookahead)

        path = path_poses(self._path, o.pose.wrd_frame, o.pose.ego_frame)
        if at_position and at_heading:
            local = self.controller.stop(o.pose, o.velocity, obstacles, o.bounds, o.dt, chassis=o.chassis)
            stopped = abs(o.velocity.linear) < 1e-3 and abs(o.velocity.angular) < 1e-3
            status = "arrived" if stopped else "braking"
        elif (at_position and diff) or len(self._path):
            local_path = path
            if at_position and diff:
                local_path = path_poses(goal[None, :2], o.pose.wrd_frame, o.pose.ego_frame)
                progress = None
            local = self.controller.step(o.pose, o.velocity, o.goal, local_path, obstacles, o.bounds, o.dt,
                                         chassis=o.chassis, progress=progress)
            status = "aligning" if at_position else "navigating"
            if local.feasible_candidates == 0:
                status = "blocked"
        else:
            local = self.controller.stop(o.pose, o.velocity, obstacles, o.bounds, o.dt, chassis=o.chassis)
            status = "no_path"
        if not local.safe:
            status = "emergency_stop"
        return NavigationOutput(o.timestamp, local.command, status, local.safe, path, local.trajectory,
                                {"elapsed": perf_counter() - started, "replanned": replanned,
                                 "planner_status": planner_status, "clearance": local.clearance,
                                 "candidates": local.candidates, "feasible_candidates": local.feasible_candidates,
                                 "planner": type(self.planner).__name__, "planning": planner_diagnostics,
                                 "controller_mode": local.mode,
                                 "progress_m": progress.distance if progress else None,
                                 "remaining_m": progress.remaining_distance if progress else None,
                                 "target": progress.target if progress else o.goal,
                                 "segment": progress.segment if progress else None,
                                 "obstacle_points": len(obstacles.xy)})
