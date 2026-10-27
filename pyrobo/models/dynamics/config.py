"""Shared motor, inertia, passive force and servo parameters."""
from dataclasses import dataclass, field
import numpy as np
from ...core.geometry import positive

@dataclass(frozen=True)
class Drag:
    coulomb: float = 2.0
    linear: float = 8.0
    quadratic: float = 1.0
    epsilon: float = 0.02

    def __post_init__(self):
        for name, value in vars(self).items():
            positive(value, name, zero=name != "epsilon")

    def __call__(self, speed):
        speed = np.asarray(speed)
        return (self.coulomb * np.tanh(speed / self.epsilon) + self.linear * speed
                + self.quadratic * np.abs(speed) * speed)

    def derivative(self, speed):
        return (self.coulomb / self.epsilon * (1 - np.tanh(speed / self.epsilon) ** 2)
                + self.linear + 2 * self.quadratic * np.abs(speed))

@dataclass(frozen=True)
class DynamicsConfig:
    mass: float = 20.0
    yaw_inertia: float = 1.2
    steering_inertia: float = 0.05
    drive_force: float = 50.0
    brake_force: float = 70.0
    wheel_force: float = 35.0
    yaw_torque: float = 3.0
    steering_torque: float = 1.0
    longitudinal_drag: Drag = field(default_factory=Drag)
    yaw_drag: Drag = field(default_factory=lambda: Drag(.08, .5, .1))
    steering_drag: Drag = field(default_factory=lambda: Drag(.02, .3, 0))
    steering_spring: float = .8
    steering_speed_spring: float = .6
    velocity_time_constant: float = .3
    yaw_time_constant: float = .25
    steering_frequency: float = 6.0
    steering_damping: float = 1.0
    drive_acceleration: float = .9
    brake_acceleration: float = 1.2
    yaw_acceleration: float = 2.5
    steering_acceleration: float = 6.0
    linear_jerk: float = 3.0
    yaw_jerk: float = 8.0
    steering_jerk: float = 40.0
    friction: float = .6
    gravity: float = 9.81
    lateral_acceleration: float = 1.2
    substep: float = .02
    timeout: float = .3
    stop_epsilon: float = 1e-4

    def __post_init__(self):
        nonnegative = {"drive_force", "brake_force", "wheel_force", "yaw_torque", "steering_torque",
                       "steering_spring", "steering_speed_spring", "friction"}
        for name, value in vars(self).items():
            if not isinstance(value, Drag):
                positive(value, name, zero=name in nonnegative)

