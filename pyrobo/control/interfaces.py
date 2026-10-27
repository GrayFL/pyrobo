from typing import Protocol
from ..core import MotionRequest, MotionResult, RobotObservation


class LocalMotionOptimizer(Protocol):
    def optimize(self, observation: RobotObservation, request: MotionRequest) -> MotionResult: ...
