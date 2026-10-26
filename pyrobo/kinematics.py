"""Model-native commands and low-speed planar kinematics (SI units)."""
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .types import Twist2D, positive

Mode = Literal["track", "coast", "soft_stop", "estop"]


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


@dataclass(frozen=True)
class AckermannCommand:
    """Rear axle speed (m/s) and equivalent front wheel steering angle (rad)."""
    speed: float = 0.0
    steering: float = 0.0
    mode: Mode = "track"

    def __post_init__(self):
        _command((self.speed, self.steering), self.mode)


ModelCommand = WheelCommand | AckermannCommand


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
