"""Measured, complete chassis state shared across all control modes."""
from dataclasses import dataclass, field
import numpy as np
from graphmap.pose import Pose
from .geometry import planar_state

@dataclass(frozen=True)
class Twist2D:
    """Forward speed in m/s and counterclockwise yaw rate in rad/s."""
    linear: float = 0.0
    angular: float = 0.0

    def __post_init__(self):
        if not np.isfinite([self.linear, self.angular]).all():
            raise ValueError("velocity must be finite")

@dataclass(frozen=True)
class ChassisState:
    """Measured plant state; pose remains graphmap.Pose, never a second pose type."""
    pose: Pose
    velocity: Twist2D = field(default_factory=Twist2D)
    steering: float = 0.0
    steering_rate: float = 0.0
    acceleration: float = 0.0
    yaw_acceleration: float = 0.0
    steering_acceleration: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self):
        planar_state(self.pose)
        if not np.isfinite([self.steering, self.steering_rate, self.acceleration, self.yaw_acceleration,
                            self.steering_acceleration, self.timestamp]).all():
            raise ValueError("chassis state must be finite")

