"""Composition configuration shared by local and process deployments."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

from pyrobo import (ChassisDynamics, DWAConfig, DWAEngine, DifferentialDrive, Footprint, NavigationConfig,
                    SafetyConfig, SafetyGuard)
from pyrobo.geometry.cache import SnapshotIndex
from pyrobo.models.dynamics.config import DynamicsConfig
from pyrobo.models.kinematics import command_space_for
from pyrobo.planning import AdaptiveClearancePlanner, RRTConnect, RRTConnectConfig, VoronoiConfig, VoronoiPlanner


class LimitedCommandSpace:
    """Restrict reference commands without changing measured-state/model limits."""
    def __init__(self, space, speed_limit=None, yaw_limit=None):
        self._space, self.speed_limit, self.yaw_limit = space, speed_limit, yaw_limit

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_space'), name)

    def bounds(self):
        lo, hi = self._space.bounds()
        if self.speed_limit is not None:
            lo[0], hi[0] = max(lo[0], -self.speed_limit), min(hi[0], self.speed_limit)
        if self.yaw_limit is not None:
            lo[1], hi[1] = max(lo[1], -self.yaw_limit), min(hi[1], self.yaw_limit)
        return lo, hi

    def to_native_batch(self, search):
        native, valid = self._space.to_native_batch(search)
        lo, hi = self.bounds()
        return native, valid & ((search >= lo-1e-10) & (search <= hi+1e-10)).all(axis=1)


@dataclass(frozen=True)
class StackConfig:
    model: object = field(default_factory=DifferentialDrive)
    footprint: Footprint = field(default_factory=Footprint)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    controller: DWAConfig | None = None
    planner: object = field(default_factory=lambda: RRTConnectConfig(time_budget=1., clearance=.08))
    navigation: NavigationConfig | None = None
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    control_period: float = .1
    execution_period: float = .02
    command_horizon: float = .3
    max_result_age: float = .25
    adaptive_clearance: bool = True
    clock: str = 'host:monotonic'
    command_speed_limit: float | None = None
    command_yaw_limit: float | None = None

    def __post_init__(self):
        if not isinstance(self.clock, str) or not self.clock:
            raise ValueError('clock must explicitly identify the environment time domain')
        for name in ('command_speed_limit', 'command_yaw_limit'):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive when supplied')
        if self.command_yaw_limit is not None and not command_space_for(self.model).can_spin:
            raise ValueError('command_yaw_limit requires a differential-drive model')
        if self.controller is None:
            car = not command_space_for(self.model).can_spin
            object.__setattr__(self, 'controller', DWAConfig(prediction_time=2.4 if car else 1.8,
                                                            lookahead=1.6 if car else 1.3))
        if self.navigation is None:
            object.__setattr__(self, 'navigation', NavigationConfig(lookahead=self.controller.lookahead,
                goal_tolerance=self.controller.goal_tolerance, yaw_tolerance=self.controller.yaw_tolerance))
        if not isinstance(self.adaptive_clearance, bool):
            raise ValueError('adaptive_clearance must be boolean')
        if not all(math.isfinite(v) for v in (self.control_period, self.execution_period,
                                            self.command_horizon, self.max_result_age)):
            raise ValueError('timing parameters must be finite')
        if not 0 < self.execution_period <= self.control_period <= self.command_horizon:
            raise ValueError('execution period <= control period <= command horizon required')
        if not 0 < self.max_result_age <= self.command_horizon <= self.dynamics.timeout:
            raise ValueError('command age / horizon must fit the chassis watchdog')
        if self.command_horizon > self.safety.max_control_dt or self.safety.expected_dt is not None:
            raise ValueError('asynchronous safety must allow variable hold intervals up to command_horizon')
        if max(2*self.execution_period, self.control_period) > min(self.safety.max_control_dt, self.dynamics.timeout):
            raise ValueError('downstream actuator lease exceeds safety or chassis watchdog')
        if self.control_period > self.controller.prediction_time:
            raise ValueError('control period exceeds the optimizer prediction horizon')

    def global_planner(self):
        if isinstance(self.planner, RRTConnectConfig):
            factory = RRTConnect
        elif isinstance(self.planner, VoronoiConfig):
            factory = VoronoiPlanner
        else:
            raise TypeError('use a supported planner config or inject a planner factory')
        if self.adaptive_clearance:
            return AdaptiveClearancePlanner(factory, self.planner, self.footprint)
        return factory(self.planner, self.footprint)

    def fingerprint(self):
        data = {'configuration': asdict(self),
                'model_type': f'{type(self.model).__module__}.{type(self.model).__qualname__}',
                'planner_type': f'{type(self.planner).__module__}.{type(self.planner).__qualname__}'}
        return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False,
                                        separators=(',', ':')).encode()).hexdigest()

    def control(self):
        plant, index = ChassisDynamics(self.model, self.dynamics), SnapshotIndex()
        space = command_space_for(self.model)
        if self.command_speed_limit is not None or self.command_yaw_limit is not None:
            space = LimitedCommandSpace(space, self.command_speed_limit, self.command_yaw_limit)
        optimizer = DWAEngine(self.controller, self.footprint, model=self.model, dynamics=plant,
                              command_space=space, index=index, input_limits=self.safety)
        guard = SafetyGuard(plant, optimizer.command_space, self.footprint, index=index, config=self.safety)
        return plant, index, optimizer, guard
