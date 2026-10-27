"""Intent is separate from the final executable command."""
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from graphmap.pose import Pose

from .paths import Path
from .state import Twist2D
from .geometry import planar_state


class MotionPolicy(StrEnum):
    NAVIGATION = 'navigation'
    TELEOP_STOP = 'teleop_stop'
    TELEOP_AVOID = 'teleop_avoid'
    STOP = 'stop'


@dataclass(frozen=True)
class MotionRequest:
    policy: MotionPolicy
    timestamp: float
    desired: 'Twist2D | None' = None
    steering: float | None = None
    path: Path | None = None
    goal: Pose | None = None
    local_target: Pose | None = None
    remaining_path: Path | None = None
    remaining_distance: float | None = None
    ttl: float = .3
    max_turn_deviation: float = .8
    max_lateral_deviation: float = .8
    max_heading_deviation: float = 1.2
    stop_mode: str = 'estop'
    source: str = 'navigation'

    def __post_init__(self):
        object.__setattr__(self, 'policy', MotionPolicy(self.policy))
        if not np.isfinite(self.timestamp) or not np.isfinite(self.ttl) or self.ttl <= 0:
            raise ValueError('request timestamp must be finite and ttl positive')
        for name in ('max_turn_deviation', 'max_lateral_deviation', 'max_heading_deviation'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.steering is not None and not np.isfinite(self.steering):
            raise ValueError('steering must be finite')
        if self.policy == MotionPolicy.NAVIGATION:
            if self.path is None or not len(self.path) or self.goal is None:
                raise ValueError('navigation requires a nonempty path and goal')
            if self.goal.wrd_frame != self.path.frame:
                raise ValueError('goal and path frames must match')
            planar_state(self.goal)
            if self.local_target is not None:
                planar_state(self.local_target)
                if self.local_target.wrd_frame != self.path.frame:
                    raise ValueError('local target frame must match the path')
            if self.remaining_path is not None and self.remaining_path.frame != self.path.frame:
                raise ValueError('remaining path frame must match the path')
            if self.remaining_distance is not None and (not np.isfinite(self.remaining_distance) or self.remaining_distance < 0):
                raise ValueError('remaining distance must be finite and nonnegative')
        if self.policy in (MotionPolicy.TELEOP_STOP, MotionPolicy.TELEOP_AVOID) and self.desired is None:
            raise ValueError('teleop requires a desired velocity')
        if self.desired is not None and not isinstance(self.desired, Twist2D):
            raise TypeError('desired velocity must be Twist2D')
        if self.stop_mode not in ('soft_stop', 'estop'):
            raise ValueError('invalid stop mode')
        if not self.source:
            raise ValueError('request source must be named')

    def fresh(self, timestamp):
        return -1e-9 <= timestamp - self.timestamp <= self.ttl + 1e-9
