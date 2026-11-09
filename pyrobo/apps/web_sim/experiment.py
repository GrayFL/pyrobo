"""Portable, bounded experiments shared by Web and headless SceneSession."""
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyrobo.core import PlanResult, PlanStatus
from pyrobo.core.geometry import path_poses, planar_state


class SceneEvent(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    time_s: float = Field(ge=0, le=3600)
    kind: Literal['add_boxes', 'freeze_cloud', 'resume_cloud']
    boxes: list[tuple[float, float, float, float]] = Field(default_factory=list, max_length=32)

    @model_validator(mode='after')
    def valid_boxes(self):
        if (self.kind == 'add_boxes') != bool(self.boxes):
            raise ValueError('only add_boxes events require nonempty boxes')
        if any(x1 >= x2 or y1 >= y2 for x1, x2, y1, y2 in self.boxes):
            raise ValueError('event boxes must have positive area')
        return self


class InitialState(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    linear: float = Field(default=0., ge=-5, le=5)
    angular: float = Field(default=0., ge=-10, le=10)
    steering: float = Field(default=0., ge=-1.5, le=1.5)
    steering_rate: float = Field(default=0., ge=-10, le=10)
    acceleration: float = Field(default=0., ge=-20, le=20)
    yaw_acceleration: float = Field(default=0., ge=-50, le=50)
    steering_acceleration: float = Field(default=0., ge=-50, le=50)


class Experiment(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    reference_path: list[tuple[float, float]] | None = Field(default=None, min_length=2, max_length=512)
    events: list[SceneEvent] = Field(default_factory=list, max_length=64)
    initial_state: InitialState = Field(default_factory=InitialState)
    expected: Literal['arrived', 'replan', 'stopped', 'infeasible', 'emergency_stop'] = 'arrived'
    max_steps: int = Field(default=800, ge=1, le=10000)

    @model_validator(mode='after')
    def ordered(self):
        if any(a.time_s > b.time_s for a, b in zip(self.events, self.events[1:])):
            raise ValueError('experiment events must be ordered by time_s')
        if self.reference_path is not None and np.any(np.linalg.norm(np.diff(self.reference_path, axis=0), axis=1) < 1e-8):
            raise ValueError('reference_path contains duplicate adjacent vertices')
        if sum(len(event.boxes) for event in self.events) > 128:
            raise ValueError('experiment exceeds 128 added boxes')
        return self


class InitialRoutePlanner:
    """Offer the authored rough route once, then use the real global planner.

    The first route deliberately need not be collision-free: that is the EB
    repair experiment. It is never an execution certificate. Changing a goal
    or requesting a replan uses the configured ordinary global planner.
    """
    def __init__(self, planner, xy, goal):
        self.planner, self.xy, self.goal = planner, np.asarray(xy), goal
        self.used = False
        self.safety_footprint = planner.safety_footprint

    def reset(self):
        self.planner.reset()

    def plan(self, start, goal, obstacles, bounds):
        if not self.used and np.allclose(planar_state(goal), planar_state(self.goal), rtol=0, atol=1e-8):
            self.used = True
            return PlanResult(path_poses(self.xy, start.wrd_frame), PlanStatus.SUCCESS, 0, 0.,
                              {'experiment_reference': True})
        self.used = True
        return self.planner.plan(start, goal, obstacles, bounds)
