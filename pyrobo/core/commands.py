"""Executable model-native commands, separate from a desired motion request."""
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
import numpy as np


class CommandMode(StrEnum):
    TRACK = 'track'
    COAST = 'coast'
    SOFT_STOP = 'soft_stop'
    ESTOP = 'estop'


Mode = Literal['track', 'coast', 'soft_stop', 'estop']

def _command(values, mode):
    if not np.isfinite(values).all():
        raise ValueError("command must be finite")
    if mode not in ("track", "coast", "soft_stop", "estop"):
        raise ValueError("unknown command mode")

@dataclass(frozen=True)
class WheelCommand:
    """Left/right wheel ground speeds, m/s; also valid for equivalent wheel pairs."""
    left: float = 0.0
    right: float = 0.0
    mode: Mode = "track"

    def __post_init__(self):
        _command((self.left, self.right), self.mode)
        object.__setattr__(self, 'mode', CommandMode(self.mode))

@dataclass(frozen=True)
class AckermannCommand:
    """Rear axle speed (m/s) and equivalent front wheel steering angle (rad)."""
    speed: float = 0.0
    steering: float = 0.0
    mode: Mode = "track"

    def __post_init__(self):
        _command((self.speed, self.steering), self.mode)
        object.__setattr__(self, 'mode', CommandMode(self.mode))

ModelCommand = WheelCommand | AckermannCommand
