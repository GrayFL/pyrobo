"""Stable string-compatible status and selected-motion results."""
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .trajectories import Trajectory
from .paths import Path
from graphmap.pose import PoseArray
from .commands import ModelCommand


class PlanStatus(StrEnum):
    SUCCESS = 'success'
    INVALID_START = 'invalid_start'
    INVALID_GOAL = 'invalid_goal'
    INVALID_GOAL_APPROACH = 'invalid_goal_approach'
    NO_PATH = 'no_path'
    BUDGET_EXHAUSTED = 'budget_exhausted'
    DEGENERATE_GEOMETRY = 'degenerate_geometry'


class MotionStatus(StrEnum):
    TRACKING = 'tracking'
    CONSTRAINED = 'constrained'
    OBSTACLE_AVOIDING = 'obstacle_avoiding'
    BRAKING = 'braking'
    INFEASIBLE = 'infeasible'
    EMERGENCY_STOP = 'emergency_stop'


class NavigationStatus(StrEnum):
    NAVIGATING = 'navigating'
    ALIGNING = 'aligning'
    BRAKING = 'braking'
    ARRIVED = 'arrived'
    NO_PATH = 'no_path'
    BLOCKED = 'blocked'
    EMERGENCY_STOP = 'emergency_stop'


class SafetyAssessment(StrEnum):
    SAFE_UNDER_MODEL = 'safe_under_model'
    NO_FEASIBLE_CANDIDATE = 'no_feasible_candidate'
    BRAKING_UNVERIFIED = 'braking_unverified'
    INPUT_STALE = 'input_stale'
    MODEL_INVALID = 'model_invalid'


class SafetyAction(StrEnum):
    NONE = 'none'
    LIMIT = 'limit'
    SOFT_STOP = 'soft_stop'
    ESTOP = 'estop'
    UNVERIFIED = 'unverified'


@dataclass(frozen=True)
class PlanResult:
    # Preserve the original PoseArray .path boundary; .route is the typed path.
    path: PoseArray
    status: PlanStatus
    extensions: int
    elapsed: float
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.path, Path):
            object.__setattr__(self, 'path', self.path.poses)
        Path(self.path)
        object.__setattr__(self, 'status', PlanStatus(self.status))

    @property
    def route(self) -> Path:
        return Path(self.path)

    @property
    def success(self):
        return self.status == PlanStatus.SUCCESS


@dataclass(frozen=True)
class Diagnostics:
    elapsed_s: float = 0.
    candidate_count: int = 0
    feasible_count: int = 0
    clearance_m: float = float('-inf')
    requested_command: tuple[float, ...] | None = None
    selected_command: tuple[float, ...] | None = None
    cost_terms: dict[str, float] = field(default_factory=dict)
    reason: str = ''
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionResult:
    command: 'ModelCommand'
    status: MotionStatus
    safety_assessment: SafetyAssessment
    trajectory: Trajectory | None
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    safety_action: SafetyAction = SafetyAction.NONE

    def __post_init__(self):
        object.__setattr__(self, 'status', MotionStatus(self.status))
        object.__setattr__(self, 'safety_assessment', SafetyAssessment(self.safety_assessment))
        object.__setattr__(self, 'safety_action', SafetyAction(self.safety_action))

    @property
    def safe(self):
        return self.safety_assessment == SafetyAssessment.SAFE_UNDER_MODEL
