from typing import Protocol
import numpy as np


class CollisionWorld(Protocol):
    cloud: object

    def clearances(self, states: np.ndarray, footprint, bounds) -> np.ndarray: ...
    def rectangle_segment_free(self, a, b, footprint, bounds) -> bool: ...
    def path_free(self, xy, footprint, bounds, *, start_yaw=None, goal_yaw=None) -> bool: ...
