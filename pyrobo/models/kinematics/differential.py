"""DifferentialDrive model and native command conversion."""
from dataclasses import dataclass
import numpy as np
from ...core.commands import WheelCommand, Mode
from ...core.state import Twist2D
from ...core.geometry import positive

@dataclass(frozen=True)
class DifferentialDrive:
    track: float = 0.4
    max_speed: float = 0.9
    max_reverse_speed: float = 0.5
    max_yaw_rate: float = 1.5
    max_wheel_speed: float = 1.4

    def __post_init__(self):
        for name, value in vars(self).items():
            positive(value, name, zero=name == "max_reverse_speed")

    def command(self, speed=0.0, turn=0.0, *, mode: Mode = "track") -> WheelCommand:
        return WheelCommand(speed - self.track * turn / 2, speed + self.track * turn / 2, mode)

    def from_twist(self, twist: Twist2D, *, mode: Mode = "track") -> WheelCommand:
        return self.command(twist.linear, twist.angular, mode=mode)

    def encode_command(self, command: WheelCommand) -> np.ndarray:
        if not isinstance(command, WheelCommand):
            raise TypeError("DifferentialDrive requires WheelCommand")
        return np.array([command.left, command.right])

    def decode_command(self, values, mode: Mode = "track") -> WheelCommand:
        return WheelCommand(*values, mode=mode)

    def references(self, commands):
        values = np.asarray(commands, dtype=float)
        wheel = np.clip(values, -self.max_wheel_speed, self.max_wheel_speed)
        v = np.clip(wheel[..., :2].mean(axis=-1), -self.max_reverse_speed, self.max_speed)
        w = np.clip((wheel[..., 1] - wheel[..., 0]) / self.track, -self.max_yaw_rate, self.max_yaw_rate)
        return v, w

