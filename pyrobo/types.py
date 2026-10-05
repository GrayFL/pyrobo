"""Shared SI-unit types. Public poses are always graphmap poses."""
from dataclasses import dataclass

import numpy as np
from graphmap.pose import Pose, PoseArray


def positive(value: float, name: str, *, zero: bool = False) -> None:
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


def planar_state(pose: Pose) -> np.ndarray:
    """Validate a rigid, z-up planar pose; return an internal [x, y, yaw] vector."""
    if not isinstance(pose, Pose):
        raise TypeError("pose must be graphmap.pose.Pose")
    if not np.isfinite(pose.matrix).all() or not np.isclose(pose.scale, 1.0, atol=1e-9, rtol=0):
        raise ValueError("2D control requires a finite rigid Pose with scale=1")
    if not np.allclose(pose.rot_mat[:, 2], [0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("2D control requires zero roll/pitch and z pointing up")
    if not np.isclose(pose.translation[2], 0.0, atol=1e-8, rtol=0):
        raise ValueError("2D robot/goal poses must lie in the z=0 plane")
    return np.array([*pose.translation[:2], np.arctan2(pose.rot_mat[1, 0], pose.rot_mat[0, 0])])


def pose2d(x: float, y: float, yaw: float = 0.0, *, frame: str = "world", body: str = "base") -> Pose:
    """Construct a graphmap Pose using radians (graphmap itself defaults to degrees)."""
    if not np.isfinite([x, y, yaw]).all():
        raise ValueError("pose coordinates must be finite")
    return Pose(translation=[x, y, 0], rotation=[0, 0, yaw], degrees=False,
                wrd_frame=frame, ego_frame=body)


def pose_array(states: np.ndarray, frame: str = "world", body: str = "base") -> PoseArray:
    states = np.asarray(states, dtype=float).reshape(-1, 3)
    position = np.column_stack((states[:, :2], np.zeros(len(states))))
    rotation = np.column_stack((np.zeros((len(states), 2)), states[:, 2]))
    return PoseArray(position=position, rotation=rotation, degrees=False,
                     wrd_frame=frame, ego_frame=body)


def path_poses(xy: np.ndarray, frame: str = "world", body: str = "base") -> PoseArray:
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    yaw = np.zeros(len(xy))
    if len(xy) > 1:
        delta = np.diff(xy, axis=0)
        yaw[:-1] = np.arctan2(delta[:, 1], delta[:, 0])
        yaw[-1] = yaw[-2]
    return pose_array(np.column_stack((xy, yaw)), frame, body)


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class Twist2D:
    """Forward speed in m/s and counterclockwise yaw rate in rad/s."""
    linear: float = 0.0
    angular: float = 0.0

    def __post_init__(self):
        if not np.isfinite([self.linear, self.angular]).all():
            raise ValueError("velocity must be finite")


@dataclass(frozen=True)
class Footprint:
    length: float = 0.8
    width: float = 0.8
    margin: float = 0.08

    def __post_init__(self):
        positive(self.length, "length")
        positive(self.width, "width")
        positive(self.margin, "margin", zero=True)

    @property
    def radius(self) -> float:
        """Circumscribed radius including the safety margin."""
        return float(np.hypot(self.length / 2, self.width / 2) + self.margin)

    def corners(self, pose: Pose) -> np.ndarray:
        planar_state(pose)
        xy = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * [self.length / 2, self.width / 2]
        return pose.apply(np.column_stack((xy, np.zeros(4))))[:, :2]


@dataclass(frozen=True)
class Bounds2D:
    """Physical workspace boundary, not just sampling limits."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self):
        if not np.isfinite([self.xmin, self.xmax, self.ymin, self.ymax]).all():
            raise ValueError("bounds must be finite")
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("bounds must have positive area")

    def contains(self, xy, inset: float = 0.0):
        xy = np.asarray(xy)
        return ((xy[..., 0] >= self.xmin + inset) & (xy[..., 0] <= self.xmax - inset)
                & (xy[..., 1] >= self.ymin + inset) & (xy[..., 1] <= self.ymax - inset))
