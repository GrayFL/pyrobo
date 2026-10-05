"""Deterministic 2D geometry, ideal lidar and acceleration-limited robot dynamics."""
from dataclasses import dataclass

import numpy as np
from graphmap.pose import Pose

from ..control import DWAConfig
from ..motion import ramp_step
from ..pointcloud import PointCloud
from ..types import Bounds2D, Footprint, Twist2D, planar_state, pose2d, positive


@dataclass(frozen=True)
class Box:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self):
        Bounds2D(self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def corners(self):
        return np.array([[self.xmin, self.ymin], [self.xmax, self.ymin],
                         [self.xmax, self.ymax], [self.xmin, self.ymax]])


class World2D:
    def __init__(self, bounds: Bounds2D, boxes=(), *, frame: str = "world"):
        if not isinstance(frame, str) or not frame:
            raise ValueError("world frame must be nonempty")
        self.bounds, self.boxes, self.frame = bounds, tuple(boxes), frame

    def segments(self):
        b = self.bounds
        polygons = [Box(b.xmin, b.xmax, b.ymin, b.ymax).corners, *(box.corners for box in self.boxes)]
        return np.concatenate([np.stack((p, np.roll(p, -1, axis=0)), axis=1) for p in polygons])

    def point_cloud(self, spacing: float = 0.04, *, timestamp: float | None = None) -> PointCloud:
        """Known map: densely sampled boundaries, with no grid or filled raster."""
        positive(spacing, "point spacing")
        samples = []
        for a, b in self.segments():
            count = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
            samples.append(np.linspace(a, b, count, endpoint=False))
        return PointCloud(np.concatenate(samples), self.frame, timestamp)

    def lidar(self, pose: Pose, *, timestamp: float, beams: int = 360, max_range: float = 8.0) -> PointCloud:
        """Exact ray/segment first returns, in the robot frame, with occlusion.

        Missing returns are omitted; they do not certify free space for the planner.
        This is ideal noiseless 2D lidar, separate from the known map used in demos.
        """
        state = planar_state(pose)
        if pose.wrd_frame != self.frame or not pose.ego_frame:
            raise ValueError("lidar pose must label the world and robot frames")
        if not isinstance(beams, int) or beams < 1:
            raise ValueError("beams must be a positive integer")
        positive(max_range, "max_range")
        positive(timestamp, "timestamp", zero=True)
        angles = np.linspace(-np.pi, np.pi, beams, endpoint=False)
        direction = np.column_stack((np.cos(angles + state[2]), np.sin(angles + state[2])))
        segments = self.segments()
        edge = segments[:, 1] - segments[:, 0]
        offset = segments[:, 0] - state[:2]
        cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        denominator = cross(direction[:, None], edge[None])
        with np.errstate(divide="ignore", invalid="ignore"):
            distance = cross(offset, edge)[None] / denominator
            fraction = cross(offset[None], direction[:, None]) / denominator
        valid = ((np.abs(denominator) > 1e-12) & (distance >= 0) & (fraction >= 0) & (fraction <= 1))
        ranges = np.where(valid, distance, np.inf).min(axis=1)
        hit = ranges <= max_range
        points = ranges[hit, None] * np.column_stack((np.cos(angles[hit]), np.sin(angles[hit])))
        return PointCloud(points, pose.ego_frame, timestamp)

    def collides(self, pose: Pose, footprint: Footprint, *, padding: float = 0.0) -> bool:
        """Independent solid-geometry truth using the separating-axis theorem."""
        state = planar_state(pose)
        if pose.wrd_frame != self.frame:
            raise ValueError("pose frame must match world")
        positive(padding, "padding", zero=True)
        robot = Footprint(footprint.length + 2 * padding, footprint.width + 2 * padding, 0).corners(pose)
        if not np.all(self.bounds.contains(robot)):
            return True
        c, s = np.cos(state[2]), np.sin(state[2])
        axes = np.array([[1, 0], [0, 1], [c, s], [-s, c]])
        p = robot @ axes.T
        for box in self.boxes:
            q = box.corners @ axes.T
            if np.all((p.max(axis=0) >= q.min(axis=0)) & (q.max(axis=0) >= p.min(axis=0))):
                return True
        return False


@dataclass(frozen=True)
class RobotState:
    timestamp: float
    pose: Pose
    velocity: Twist2D


class Simulator2D:
    """No rendering dependency. Each step consumes a velocity setpoint."""
    def __init__(self, world: World2D, start: Pose, *, footprint: Footprint | None = None,
                 dynamics: DWAConfig | None = None, timestamp: float = 0.0):
        self.world, self.footprint = world, footprint or Footprint()
        self.dynamics = dynamics or DWAConfig()
        positive(timestamp, "timestamp", zero=True)
        if world.collides(start, self.footprint):
            raise ValueError("initial robot intersects solid geometry")
        self.state = RobotState(timestamp, start.copy(), Twist2D())
        self.collided = False

    def step(self, command: Twist2D, dt: float = 0.1) -> RobotState:
        positive(dt, "dt")
        cfg = self.dynamics
        if not 0 <= command.linear <= cfg.max_speed + 1e-9 or abs(command.angular) > cfg.max_yaw_rate + 1e-9:
            raise ValueError("command outside simulator limits")
        if self.collided:
            raise RuntimeError("simulation terminated after collision")
        state = planar_state(self.state.pose)[None]
        velocity = np.array([[self.state.velocity.linear, self.state.velocity.angular]])
        target = np.array([[command.linear, command.angular]])
        count = max(1, int(np.ceil(dt / 0.02)))
        h = dt / count
        old_pose = self.state.pose
        for _ in range(count):
            candidate, new_velocity = ramp_step(state, velocity, target, cfg.linear_acceleration, cfg.angular_acceleration, h)
            pose = pose2d(*candidate[0], frame=self.world.frame, body=old_pose.ego_frame)
            padding = (max(velocity[0, 0], new_velocity[0, 0]) + self.footprint.radius *
                       max(abs(velocity[0, 1]), abs(new_velocity[0, 1]))) * h / 2
            if (self.world.collides(pose, self.footprint, padding=padding)
                    or self.world.collides(old_pose, self.footprint, padding=padding)):
                self.collided = True
                velocity[:] = 0
                break
            state, velocity, old_pose = candidate, new_velocity, pose
        self.state = RobotState(self.state.timestamp + dt, old_pose, Twist2D(*velocity[0]))
        return self.state


def demo_world():
    """A reproducible detour with space for a 0.8 m square to turn."""
    world = World2D(Bounds2D(0, 12, 0, 9), [Box(3.4, 4.4, 0.0, 5.7),
                                          Box(7.0, 8.0, 3.4, 9.0)])
    return world, pose2d(1.4, 1.5), pose2d(10.5, 7.2, np.pi / 2, body="goal")

