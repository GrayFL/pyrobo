"""Independent solid geometry and force-driven model-native simulation."""
from dataclasses import dataclass, replace

import numpy as np
from graphmap.pose import Pose

from pyrobo.models.dynamics.chassis import ChassisDynamics
from pyrobo.core.state import ChassisState
from pyrobo.models.dynamics.config import DynamicsConfig
from pyrobo.models.kinematics.ackermann import Ackermann
from pyrobo.models.kinematics.differential import DifferentialDrive
from pyrobo.core.commands import ModelCommand
from pyrobo.core.obstacles import PointCloud
from pyrobo.core.obstacles import Bounds2D
from pyrobo.core.footprint import Footprint
from pyrobo.core.state import Twist2D
from pyrobo.core.geometry import planar_state, pose2d, positive


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
        robot = Footprint(footprint.length + 2 * padding, footprint.width + 2 * padding, 0, footprint.offset_x).corners(pose)
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


# Backward-compatible import name; pose still uses graphmap.Pose.
RobotState = ChassisState


class Simulator2D:
    """Manual and autonomous commands run through exactly the same plant.

    The public pose is the kinematic reference point. For an Ackermann rear
    axle, Footprint.offset_x can place the body center ahead of that point.
    """
    def __init__(self, world: World2D, start: Pose, *, footprint: Footprint | None = None,
                 model: DifferentialDrive | Ackermann | None = None,
                 dynamics: DynamicsConfig | None = None, timestamp: float = 0.0):
        self.world, self.footprint = world, footprint or Footprint()
        self.plant = ChassisDynamics(model, dynamics)
        self.model, self.dynamics = self.plant.model, self.plant.config
        positive(timestamp, "timestamp", zero=True)
        if world.collides(start, self.footprint):
            raise ValueError("initial robot intersects solid geometry")
        self.state = ChassisState(start.copy(), timestamp=timestamp)
        self.collided = False
        self.last_dynamics_result = None

    def step(self, command: ModelCommand | Twist2D, dt: float = 0.1, *, command_age=0.0) -> ChassisState:
        positive(dt, "dt")
        positive(command_age, "command_age", zero=True)
        if self.collided:
            raise RuntimeError("simulation terminated after collision")
        if isinstance(command, Twist2D):
            command = self.model.from_twist(command)
        self.model.encode_command(command)
        count = max(1, int(np.ceil(dt / self.dynamics.substep - 1e-10)))
        h = dt / count
        end_time = self.state.timestamp + dt
        for i in range(count):
            result = self.plant.step(self.state, command, h, command_age=command_age + i*h)
            self.last_dynamics_result = result
            candidate = result.state
            mid_v = (self.state.velocity.linear + candidate.velocity.linear) / 2
            mid_w = (self.state.velocity.angular + candidate.velocity.angular) / 2
            padding = (abs(mid_v) + self.footprint.radius * abs(mid_w)) * h / 2
            if (self.world.collides(candidate.pose, self.footprint, padding=padding)
                    or self.world.collides(self.state.pose, self.footprint, padding=padding)):
                self.collided = True
                self.state = replace(self.state, velocity=Twist2D(), steering_rate=0., acceleration=0.,
                                     yaw_acceleration=0., steering_acceleration=0.)
                break
            self.state = candidate
        self.state = replace(self.state, timestamp=end_time)
        return self.state


def demo_world():
    """A reproducible staggered-wall detour for both models and body presets."""
    world = World2D(Bounds2D(0, 12, 0, 9), [Box(3.4, 4.4, 0.0, 5.7),
                                          Box(7.0, 8.0, 3.4, 9.0)])
    return world, pose2d(1.4, 1.5), pose2d(10.5, 7.2, np.pi / 2, body="goal")

