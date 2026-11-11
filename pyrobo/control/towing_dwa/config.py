"""Classic DWA weights and numerical resolution; vehicle limits are injected."""
from dataclasses import dataclass

from ...core.geometry import positive


@dataclass(frozen=True)
class TowingDWAConfig:
    prediction_time: float = 1.8
    integration_dt: float = .08
    braking_timeout: float = 4.
    linear_samples: int = 11
    angular_samples: int = 21
    lookahead: float = 1.3
    goal_tolerance: float = .18
    yaw_tolerance: float = .10
    heading_weight: float = 1.
    clearance_weight: float = .2
    speed_weight: float = .2
    clearance_scale: float = 1.
    allow_reverse: bool = False

    def __post_init__(self):
        for name in ('prediction_time', 'integration_dt', 'braking_timeout', 'lookahead',
                     'goal_tolerance', 'yaw_tolerance', 'clearance_scale'):
            positive(getattr(self, name), name)
        for name in ('heading_weight', 'clearance_weight', 'speed_weight'):
            positive(getattr(self, name), name, zero=True)
        for name in ('linear_samples', 'angular_samples'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f'{name} must be an integer >= 2')
        if not isinstance(self.allow_reverse, bool):
            raise ValueError('allow_reverse must be boolean')
