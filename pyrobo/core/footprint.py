"""Rectangular collision footprint about the kinematic reference point."""
from dataclasses import dataclass
import numpy as np
from graphmap.pose import Pose
from .geometry import planar_state, positive

@dataclass(frozen=True)
class Footprint:
    """Rectangle in the robot frame; offset_x locates its center from the axle."""
    length: float = 0.5
    width: float = 0.5
    margin: float = 0.08
    offset_x: float = 0.0

    def __post_init__(self):
        positive(self.length, "length")
        positive(self.width, "width")
        positive(self.margin, "margin", zero=True)
        if not np.isfinite(self.offset_x):
            raise ValueError("offset_x must be finite")

    @classmethod
    def compact(cls, **kwargs):
        return cls(0.5, 0.5, **kwargs)

    @classmethod
    def long(cls, **kwargs):
        return cls(1.0, 0.5, **kwargs)

    @property
    def radius(self) -> float:
        """Circumscribed radius including the safety margin."""
        return float(np.hypot(self.length / 2 + abs(self.offset_x), self.width / 2) + self.margin)

    def corners(self, pose: Pose) -> np.ndarray:
        planar_state(pose)
        xy = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]]) * [self.length / 2, self.width / 2]
        xy[:, 0] += self.offset_x
        return pose.apply(np.column_stack((xy, np.zeros(4))))[:, :2]

