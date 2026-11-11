"""Bounded Timed Elastic Band settings; vehicle limits belong to the model."""
from dataclasses import dataclass

from ...core.geometry import positive


@dataclass(frozen=True)
class TowingTEBConfig:
    prediction_time: float = 2.4
    integration_dt: float = .08
    braking_timeout: float = 4.
    lookahead: float = 1.3
    goal_tolerance: float = .18
    yaw_tolerance: float = .10
    allow_reverse: bool = True
    dt_ref: float = .3
    dt_min: float = .06
    dt_max: float = 1.5
    min_samples: int = 5
    max_samples: int = 16
    max_iterations: int = 30
    time_budget: float | None = None
    obstacle_clearance: float = .12
    time_weight: float = 1.
    obstacle_weight: float = 80.
    kinematics_weight: float = 300.
    velocity_weight: float = 30.
    acceleration_weight: float = 10.
    turning_weight: float = 100.

    def __post_init__(self):
        for name in ('prediction_time', 'integration_dt', 'braking_timeout', 'lookahead',
                     'goal_tolerance', 'yaw_tolerance', 'dt_ref', 'dt_min', 'dt_max'):
            positive(getattr(self, name), name)
        for name in ('obstacle_clearance', 'time_weight', 'obstacle_weight',
                     'kinematics_weight', 'velocity_weight', 'acceleration_weight', 'turning_weight'):
            positive(getattr(self, name), name, zero=True)
        for name, minimum in (('min_samples', 3), ('max_samples', 3), ('max_iterations', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.max_samples < self.min_samples:
            raise ValueError('max_samples must be >= min_samples')
        if not self.dt_min <= self.dt_ref <= self.dt_max or self.dt_min == self.dt_max:
            raise ValueError('require dt_min <= dt_ref <= dt_max and dt_min < dt_max')
        if self.time_budget is not None:
            positive(self.time_budget, 'time_budget')
        for name in ('allow_reverse',):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')
