"""Compatibility API over the shared local motion optimizer."""
from dataclasses import dataclass, replace

import numpy as np
from graphmap.pose import Pose, PoseArray

from ...core import MotionPolicy, MotionRequest, Path, RobotObservation
from pyrobo.core.state import ChassisState
from pyrobo.core.commands import ModelCommand
from pyrobo.core.obstacles import Bounds2D
from pyrobo.core.state import Twist2D
from pyrobo.core.geometry import positive
from ...geometry.sweep import swept_clearances
from ...safety.fallback import verified_stop
from .config import DWAConfig
from .engine import DWAEngine
from .path import PathProgress


@dataclass(frozen=True)
class DWAResult:
    command: ModelCommand
    trajectory: PoseArray
    safe: bool
    clearance: float
    candidates: int
    feasible_candidates: int
    mode: str = 'following'


class DynamicWindow(DWAEngine):
    """Legacy step/stop signatures; optimize(observation, request) is the new API."""
    def _chassis(self, pose, velocity, chassis=None):
        if chassis is None:
            if not self.command_space.can_spin:
                raise ValueError('Ackermann requires measured ChassisState, including steering at rest')
            chassis = ChassisState(pose, velocity)
        if (chassis.pose.wrd_frame != pose.wrd_frame or not np.allclose(chassis.pose.matrix, pose.matrix)
                or chassis.velocity != velocity):
            raise ValueError('ChassisState must match observed pose and velocity')
        return chassis

    def _validate(self, pose, velocity, obstacles, dt, chassis=None):
        state = self._chassis(pose, velocity, chassis)
        positive(dt, 'control dt')
        if dt > self.config.prediction_time:
            raise ValueError('control dt must not exceed prediction_time')
        if pose.wrd_frame != obstacles.cloud.frame:
            raise ValueError('pose and obstacles must have the same explicit frame')
        return self.plant.codec.pack(state)

    def _rollout(self, state, commands, dt, *, stop_only=False, braking=False):
        batch = self.predictor.predict(state, commands, dt, stop_only=stop_only, mode='estop' if braking else 'track')
        return batch.states, batch.sweep, batch.prediction_end, batch.stopped

    def _clearance(self, trajectories, sweep, obstacles, bounds):
        poses = trajectories if trajectories.shape[-1] == 3 else self.plant.codec.poses(trajectories)
        return swept_clearances(poses, sweep, obstacles, self.footprint, bounds)

    @staticmethod
    def _legacy(result, mode=None):
        diag = result.diagnostics
        return DWAResult(result.command, result.trajectory.poses, result.safe, diag.clearance_m,
                         diag.candidate_count, diag.feasible_count, mode or diag.extra.get('controller_mode', 'following'))

    def stop(self, pose: Pose, velocity: Twist2D, obstacles, bounds: Bounds2D, dt: float,
             *, chassis: ChassisState | None = None) -> DWAResult:
        self._validate(pose, velocity, obstacles, dt, chassis)
        state = self._chassis(pose, velocity, chassis)
        if chassis is None and obstacles.cloud.timestamp is not None:
            state = replace(state, timestamp=obstacles.cloud.timestamp)
        observation = RobotObservation(state, obstacles.cloud, bounds, dt)
        result = verified_stop(observation, self.predictor, self.command_space, obstacles)
        self.navigation_policy.turn_active = False
        return self._legacy(result, 'braking')

    def step(self, pose: Pose, velocity: Twist2D, goal: Pose, path: PoseArray, obstacles,
             bounds: Bounds2D, dt: float, *, chassis: ChassisState | None = None,
             progress: PathProgress | None = None) -> DWAResult:
        self._validate(pose, velocity, obstacles, dt, chassis)
        state = self._chassis(pose, velocity, chassis)
        if chassis is None and obstacles.cloud.timestamp is not None:
            state = replace(state, timestamp=obstacles.cloud.timestamp)
        observation = RobotObservation(state, obstacles.cloud, bounds, dt)
        request = MotionRequest(MotionPolicy.NAVIGATION, state.timestamp, path=Path(path), goal=goal,
                                local_target=progress.target if progress else None,
                                remaining_path=Path(progress.remaining) if progress else None,
                                remaining_distance=progress.remaining_distance if progress else None)
        return self._legacy(self.optimize(observation, request, collision_world=obstacles))
