"""Shared acceleration-limited unicycle model, independent of plotting/simulation."""
import numpy as np


def advance(states, linear, angular, dt):
    """Exact constant-twist arc; supports NumPy broadcasting over candidate states."""
    states = np.asarray(states, dtype=float)
    result = states.copy()
    turn = np.asarray(angular) * dt
    distance = np.asarray(linear) * dt * np.sinc(turn / (2 * np.pi))
    result[..., 0] += distance * np.cos(states[..., 2] + turn / 2)
    result[..., 1] += distance * np.sin(states[..., 2] + turn / 2)
    result[..., 2] += turn
    return result


def approach(current, target, acceleration, dt):
    return np.asarray(current) + np.clip(np.asarray(target) - current, -acceleration * dt, acceleration * dt)


def ramp_step(states, velocity, target, linear_acceleration, angular_acceleration, dt):
    """Midpoint integration of a slew-limited velocity setpoint over a small step."""
    new = np.column_stack((approach(velocity[..., 0], target[..., 0], linear_acceleration, dt),
                           approach(velocity[..., 1], target[..., 1], angular_acceleration, dt)))
    mid = (velocity + new) / 2
    return advance(states, mid[..., 0], mid[..., 1], dt), new

