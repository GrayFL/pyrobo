"""Pure numerical axes and planar integration; shared by prediction and execution."""
import numpy as np
from ...core.geometry import positive
from .config import Drag

def implicit_axis(position, speed, effort, mass, spring, drag: Drag, dt):
    """Backward Euler of M*qddot = effort - K*q - drag(qdot).

    The monotone scalar solve is vectorized across all candidates. Bracketed
    Newton steps preserve dissipativity even for large steps and stiff drag.
    """
    positive(dt, "dt")
    if (not all(np.isfinite(value).all() for value in (position, speed, effort, mass, spring))
            or np.any(np.asarray(mass) <= 0) or np.any(np.asarray(spring) < 0)):
        raise ValueError("axis inputs must be finite, mass positive and spring nonnegative")
    effective = mass + spring * dt * dt
    free = (mass * speed + dt * (effort - spring * position)) / effective
    lo, hi = np.minimum(0., free), np.maximum(0., free)
    value = (lo + hi) / 2
    scale = dt / effective
    for _ in range(32):
        residual = value + scale * drag(value) - free
        if np.all(np.abs(residual) < 1e-12):
            break
        lo, hi = np.where(residual < 0, value, lo), np.where(residual > 0, value, hi)
        trial = value - residual / (1 + scale * drag.derivative(value))
        value = np.where((trial >= lo) & (trial <= hi), trial, (lo + hi) / 2)
    return position + dt * value, value

def _effort(position, speed, acceleration, mass, spring, drag, dt):
    next_speed = speed + dt * acceleration
    return mass * acceleration + spring * (position + dt * next_speed) + drag(next_speed)

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

