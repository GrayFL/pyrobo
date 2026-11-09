"""Independent geometric repair and kinodynamic tracking settings."""
from dataclasses import dataclass

import numpy as np

from ...core.geometry import positive


@dataclass(frozen=True)
class EBDWAConfig:
    # Output coverage, free optimization range and tracking preview are distinct.
    max_coverage_m: float = 12.
    min_coverage_m: float = 4.
    coverage_margin_m: float = .8
    rejoin_validation_m: float = .8
    rejoin_tail_m: float = 1.
    active_padding_m: float = 1.2
    node_spacing_m: float = .45
    obstacle_node_spacing_m: float = .2
    reference_spacing_m: float = .08
    max_nodes: int = 96
    preferred_clearance: float = .16
    minimum_clearance: float = 0.
    max_tube_width_m: float = .5
    sharp_corner_angle: float = .25
    preferred_curvature: float = 1.2
    preferred_curvature_rate: float = 2.
    max_deformation_m: float = 2.5
    smooth_weight: float = 1.
    curvature_weight: float = .08
    curvature_rate_weight: float = .02
    curvature_limit_weight: float = 100.
    obstacle_weight: float = 80.
    hard_clearance_weight: float = 2000.
    global_anchor_weight: float = .3
    max_iterations: int = 24
    max_evaluations: int = 350
    max_refinement_cycles: int = 8
    time_budget_s: float | None = None
    validation_depth: int = 12
    max_validation_intervals: int = 8192
    warm_start: bool = True
    topology_attempts: int = 3
    topology_offset_m: float = 1.2
    fallback_enabled: bool = True
    prediction_time: float = 1.8
    integration_dt: float = .08
    braking_timeout: float = 4.
    linear_samples: int = 7
    angular_samples: int = 17
    batch_size: int = 0
    track_lookahead_m: float = 1.3
    track_lookahead_time_s: float = .5
    goal_tolerance: float = .18
    yaw_tolerance: float = .10
    cross_track_weight: float = 2.
    heading_weight: float = .5
    progress_weight: float = 2.5
    backward_weight: float = 4.
    tube_weight: float = 4.
    clearance_weight: float = .08
    speed_weight: float = 6.
    control_weight: float = .08
    endpoint_weight: float = 3.
    allow_reverse: bool = True
    reject_lateral_violation: bool = False
    speed_profile: bool = True
    speed_profile_samples: int = 9
    speed_response_time_s: float = .5
    speed_lateral_fraction: float = .3

    @property
    def lookahead(self):
        """Navigation's existing composition interface, in metres."""
        return self.track_lookahead_m

    def __post_init__(self):
        integers = {'max_nodes': 4, 'max_iterations': 1, 'max_evaluations': 1, 'max_refinement_cycles': 1,
                    'validation_depth': 1, 'max_validation_intervals': 4,
                    'topology_attempts': 1, 'linear_samples': 2,
                    'angular_samples': 2, 'batch_size': 0, 'speed_profile_samples': 3}
        flags = ('warm_start', 'fallback_enabled', 'allow_reverse',
                 'reject_lateral_violation', 'speed_profile')
        for name, value in vars(self).items():
            if name in flags:
                if not isinstance(value, bool):
                    raise ValueError(f'{name} must be boolean')
            elif name in integers:
                if type(value) is not int or value < integers[name]:
                    raise ValueError(f'{name} must be an integer >= {integers[name]}')
            elif name == 'time_budget_s' and value is None:
                continue
            else:
                if isinstance(value, bool):
                    raise ValueError(f'{name} must be a number')
                positive(value, name, zero=name.endswith('_weight') or name == 'minimum_clearance')
        if self.minimum_clearance > self.preferred_clearance:
            raise ValueError('minimum_clearance must not exceed preferred_clearance')
        if self.min_coverage_m > self.max_coverage_m:
            raise ValueError('min_coverage_m must not exceed max_coverage_m')
        if self.obstacle_node_spacing_m > self.node_spacing_m:
            raise ValueError('obstacle spacing must not exceed ordinary node spacing')
        if self.sharp_corner_angle > np.pi or self.topology_attempts > 5:
            raise ValueError('corner angle must be <= pi and topology_attempts <= 5')
        if self.speed_lateral_fraction > 1:
            raise ValueError('lateral fraction must be <= 1')
