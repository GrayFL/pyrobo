"""Ackermann model and native command conversion."""
from dataclasses import dataclass
import numpy as np
from ...core.commands import AckermannCommand, Mode
from ...core.state import Twist2D
from ...core.geometry import positive

@dataclass(frozen=True)
class Ackermann:
    """Bicycle model about the rear axle midpoint, including reverse motion."""
    wheelbase: float = 0.5
    max_steering: float = np.deg2rad(35)
    max_steering_rate: float = 1.2
    max_speed: float = 0.9
    max_reverse_speed: float = 0.5
    max_yaw_rate: float = 1.5

    def __post_init__(self):
        for name, value in vars(self).items():
            positive(value, name, zero=name == "max_reverse_speed")
        if self.max_steering >= np.pi / 2:
            raise ValueError("max_steering must be less than pi/2")

    def command(self, speed=0.0, turn=0.0, *, mode: Mode = "track") -> AckermannCommand:
        return AckermannCommand(speed, turn, mode)

    @property
    def minimum_turn_radius(self):
        return self.wheelbase / np.tan(self.max_steering)

    def from_twist(self, twist: Twist2D, *, mode: Mode = "track") -> AckermannCommand:
        if abs(twist.linear) < 1e-12:
            if abs(twist.angular) > 1e-12:
                raise ValueError("Ackermann cannot rotate in place; use a steering command at rest")
            return self.command(mode=mode)
        return self.command(twist.linear, np.arctan(self.wheelbase * twist.angular / twist.linear), mode=mode)

    def yaw_rate(self, speed, steering):
        return np.asarray(speed) * np.tan(steering) / self.wheelbase

    def encode_command(self, command: AckermannCommand) -> np.ndarray:
        if not isinstance(command, AckermannCommand):
            raise TypeError("Ackermann requires AckermannCommand")
        return np.array([command.speed, command.steering])

    def decode_command(self, values, mode: Mode = "track") -> AckermannCommand:
        return AckermannCommand(*values, mode=mode)

    def references(self, commands):
        values = np.asarray(commands, dtype=float)
        return (np.clip(values[..., 0], -self.max_reverse_speed, self.max_speed),
                np.clip(values[..., 1], -self.max_steering, self.max_steering))

