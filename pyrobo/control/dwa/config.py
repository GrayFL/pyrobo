"""Local optimizer settings; physical limits remain owned by the model."""
from dataclasses import dataclass
import numpy as np
from pyrobo.core.geometry import positive


@dataclass(frozen=True)
class DWAConfig:
    max_speed: float = 0.9
    max_yaw_rate: float = 1.5
    linear_acceleration: float = 0.9
    angular_acceleration: float = 2.5
    prediction_time: float = 1.8
    integration_dt: float = 0.08
    linear_samples: int = 7
    angular_samples: int = 17
    lookahead: float = 1.3
    goal_tolerance: float = 0.18
    yaw_tolerance: float = 0.10
    position_weight: float = 3.0
    path_weight: float = 2.0
    heading_weight: float = 0.65
    clearance_weight: float = 0.08
    speed_weight: float = 0.15
    heading_slow_angle: float = np.pi / 3
    turn_in_place_angle: float = np.pi / 3
    turn_exit_angle: float = np.pi / 9
    turn_gain: float = 1.0
    min_turn_yaw_rate: float = 0.3
    yaw_reversal_weight: float = 0.2
    braking_timeout: float = 4.0
    allow_reverse: bool = True
    batch_size: int = 0
    collect_batch_diagnostics: bool = False
    reject_lateral_violation: bool = False

    def __post_init__(self):
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool) or self.batch_size < 0:
            raise ValueError("batch_size must be a nonnegative integer")
        for name in ("max_speed", "max_yaw_rate", "linear_acceleration", "angular_acceleration",
                     "prediction_time", "integration_dt", "lookahead", "goal_tolerance", "yaw_tolerance",
                     "heading_slow_angle", "turn_in_place_angle", "turn_exit_angle", "turn_gain", "braking_timeout"):
            positive(getattr(self, name), name)
        for name in ("position_weight", "path_weight", "heading_weight", "clearance_weight", "speed_weight",
                     "min_turn_yaw_rate", "yaw_reversal_weight"):
            positive(getattr(self, name), name, zero=True)
        if not self.turn_exit_angle < self.turn_in_place_angle <= np.pi or self.heading_slow_angle > np.pi:
            raise ValueError("turn angles require 0 < exit < enter <= pi and heading_slow_angle <= pi")
        for name in ("linear_samples", "angular_samples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 2:
                raise ValueError(f"{name} must be an integer >= 2")

