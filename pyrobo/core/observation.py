"""One explicit observation shared by navigation, control and runtime."""
from dataclasses import dataclass

import numpy as np

from .state import ChassisState
from .obstacles import Bounds2D, PointCloud


@dataclass(frozen=True)
class RobotObservation:
    chassis: 'ChassisState'
    obstacles: 'PointCloud'
    bounds: 'Bounds2D'
    dt: float = .1
    timestamp: float | None = None

    def __post_init__(self):
        timestamp = self.chassis.timestamp if self.timestamp is None else self.timestamp
        if not np.isfinite(timestamp) or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('observation requires finite timestamp and positive dt')
        if self.chassis.pose.wrd_frame != self.obstacles.frame:
            raise ValueError('chassis and obstacles must share an explicit world frame')
        object.__setattr__(self, 'timestamp', float(timestamp))

    @property
    def frame(self):
        return self.obstacles.frame
