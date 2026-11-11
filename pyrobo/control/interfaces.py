from typing import Protocol
from ..core import MotionRequest, MotionResult, RobotObservation


class LocalMotionOptimizer(Protocol):
    """Only the per-cycle input/output contract; no algorithm base class.

    optimize() proposes a native command and reports its assessment; it must not
    advance the real chassis. Navigator/Supervisor independently gate execution.
    Sampling, trajectory optimization, validation and diagnostics are private.

    Optional lifecycle hook: reset() clears local caches on navigation reset or
    relocation. Navigator discovers it with getattr; stateless implementations
    need not provide it. See control/README.md for built-in factory arguments and
    composition attributes, which are separate from this callable protocol.
    """

    def optimize(self, observation: RobotObservation, request: MotionRequest) -> MotionResult: ...
