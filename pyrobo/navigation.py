"""One current observation in, one current control decision out."""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from graphmap.pose import Pose, PoseArray

from .control import DWAConfig, DynamicWindow
from .planning import RRTConnect, RRTConnectConfig
from .pointcloud import PointCloud, PointObstacles
from .types import Bounds2D, Footprint, Twist2D, path_poses, planar_state, positive, wrap_angle


@dataclass(frozen=True)
class NavigationInput:
    timestamp: float
    pose: Pose
    velocity: Twist2D
    goal: Pose
    obstacles: PointCloud
    bounds: Bounds2D
    dt: float = 0.1


@dataclass(frozen=True)
class NavigationOutput:
    timestamp: float
    command: Twist2D
    status: str
    safe: bool
    path: PoseArray
    trajectory: PoseArray
    diagnostics: dict = field(default_factory=dict)


class Navigator:
    """Stateful RRT-Connect + DWA black box. No global state, I/O, plotting or threads.

    History is bounded to a cached path, goal, timing/progress bookkeeping and RNG.
    A PointCloud is a complete current obstacle snapshot, not a map update/delta.
    """
    def __init__(self, *, footprint: Footprint | None = None, planner: RRTConnectConfig | None = None,
                 controller: DWAConfig | None = None, point_radius: float = 0.03,
                 z_range: tuple[float, float] | None = None, max_cloud_age: float = 0.3,
                 retry_interval: float = 0.5, stall_timeout: float = 4.0):
        self.footprint = footprint or Footprint()
        self.planner = RRTConnect(planner, self.footprint)
        self.controller = DynamicWindow(controller, self.footprint)
        positive(point_radius, "point_radius", zero=True)
        positive(max_cloud_age, "max_cloud_age", zero=True)
        positive(retry_interval, "retry_interval")
        positive(stall_timeout, "stall_timeout")
        self.point_radius, self.z_range = point_radius, z_range
        self.max_cloud_age, self.retry_interval, self.stall_timeout = max_cloud_age, retry_interval, stall_timeout
        self.reset()

    def reset(self):
        self.planner.reset()
        self._path = np.empty((0, 2))
        self._goal = None
        self._last_timestamp = None
        self._last_plan = -np.inf
        self._progress_position = None
        self._progress_time = None

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
        self.controller._validate(o.pose, o.velocity, obstacles, o.dt)
        self._last_timestamp = o.timestamp
        cfg = self.controller.config
        at_position = np.linalg.norm(current[:2] - goal[:2]) <= cfg.goal_tolerance
        at_heading = abs(float(wrap_angle(current[2] - goal[2]))) <= cfg.yaw_tolerance
        signature = (*goal, o.obstacles.frame, o.bounds)
        changed_goal = self._goal != signature
        if changed_goal:
            self._path = np.empty((0, 2))
            self._goal = signature
            self._last_plan = -np.inf
            self._progress_position, self._progress_time = current[:2].copy(), o.timestamp
        if np.linalg.norm(current[:2] - self._progress_position) > 0.15 or at_position:
            self._progress_position, self._progress_time = current[:2].copy(), o.timestamp
        elif o.timestamp - self._progress_time > self.stall_timeout:
            self._path = np.empty((0, 2))
            self._last_plan = -np.inf
            self._progress_time = o.timestamp

        replanned = False
        planner_status = "cached"
        if not at_position:
            if len(self._path):
                nearest = int(np.argmin(np.linalg.norm(self._path - current[:2], axis=1)))
                rest = self._path[nearest:]
                candidate = np.vstack((current[:2], rest))
                if all(obstacles.segment_free(a, b, self.footprint.radius, o.bounds)
                       for a, b in zip(candidate[:-1], candidate[1:])):
                    self._path = rest
                else:
                    self._path = np.empty((0, 2))
                    self._last_plan = -np.inf
            if len(self._path) == 0 and o.timestamp - self._last_plan >= self.retry_interval:
                plan = self.planner.plan(o.pose, o.goal, obstacles, o.bounds)
                self._last_plan = o.timestamp
                replanned, planner_status = True, plan.status
                self._path = plan.path.position[:, :2].copy()

        path = path_poses(self._path, o.pose.wrd_frame, o.pose.ego_frame)
        if at_position and at_heading:
            local = self.controller.stop(o.pose, o.velocity, obstacles, o.bounds, o.dt)
            stopped = abs(o.velocity.linear) < 1e-3 and abs(o.velocity.angular) < 1e-3
            status = "arrived" if stopped else "braking"
        elif at_position or len(self._path):
            if at_position:
                path = path_poses(goal[None, :2], o.pose.wrd_frame, o.pose.ego_frame)
            local = self.controller.step(o.pose, o.velocity, o.goal, path, obstacles, o.bounds, o.dt)
            status = "aligning" if at_position else "navigating"
            if local.feasible_candidates == 0:
                status = "blocked"
        else:
            local = self.controller.stop(o.pose, o.velocity, obstacles, o.bounds, o.dt)
            status = "no_path"
        if not local.safe:
            status = "emergency_stop"
        return NavigationOutput(o.timestamp, local.command, status, local.safe, path, local.trajectory,
                                {"elapsed": perf_counter() - started, "replanned": replanned,
                                 "planner_status": planner_status, "clearance": local.clearance,
                                 "candidates": local.candidates, "feasible_candidates": local.feasible_candidates,
                                 "obstacle_points": len(obstacles.xy)})
