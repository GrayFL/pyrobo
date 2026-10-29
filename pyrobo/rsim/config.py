"""Composition configuration shared by local and process deployments."""
from dataclasses import dataclass, field
import math

from pyrobo import (ChassisDynamics, DWAConfig, DWAEngine, DifferentialDrive, Footprint, NavigationConfig,
                    SafetyConfig, SafetyGuard)
from pyrobo.geometry.cache import SnapshotIndex
from pyrobo.models.dynamics.config import DynamicsConfig
from pyrobo.models.kinematics import command_space_for
from pyrobo.planning import AdaptiveClearancePlanner, RRTConnect, RRTConnectConfig, VoronoiConfig, VoronoiPlanner


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

    def __post_init__(self):
        if not isinstance(self.clock, str) or not self.clock:
            raise ValueError('clock must explicitly identify the environment time domain')
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

    def control(self):
        plant, index = ChassisDynamics(self.model, self.dynamics), SnapshotIndex()
        optimizer = DWAEngine(self.controller, self.footprint, model=self.model, dynamics=plant,
                              index=index, input_limits=self.safety)
        guard = SafetyGuard(plant, optimizer.command_space, self.footprint, index=index, config=self.safety)
        return plant, index, optimizer, guard
