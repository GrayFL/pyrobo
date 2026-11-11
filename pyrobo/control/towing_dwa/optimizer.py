"""Editable DWA mathematics: acceleration window and three-term objective.

Fox/Burgard/Thrun (1997): search V_s ∩ V_d ∩ V_a, prefer heading,
obstacle distance and forward velocity. See README for the rectangle/plant
adaptation. No route-distance, recovery, turn latch or previous-command cost.
"""
import numpy as np

from ...core.geometry import wrap_angle


def velocity_window(current, lower, upper, acceleration, dt, samples):
    """Sample V_s ∩ V_d; admissibility V_a is checked by physical rollout.

    All arguments are numerical arrays in search coordinates (v, w), or
    (v, steering) for the optional car adapter. Returns an (N, 2) array.
    """
    lo = np.maximum(lower, current-acceleration*dt)
    hi = np.minimum(upper, current+acceleration*dt)
    if np.any(lo > hi):
        return np.empty((0, 2))
    axes = [np.unique(np.r_[np.linspace(a, b, n), np.clip(0., a, b)])
            for a, b, n in zip(lo, hi, samples)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)


def objective(terminal, search, clearance, target, max_speed, config, *, goal_yaw=None):
    """Three normalized costs, lower is better; no hidden constraints.

    terminal: (N,3) predicted x/y/yaw; search: (N,2) commands;
    clearance: (N,) rectangle clearance; target: (2,) local goal.
    goal_yaw optionally replaces the heading target for final pose alignment.
    """
    direction = np.arctan2(target[1]-terminal[:, 1], target[0]-terminal[:, 0])
    heading = np.abs(wrap_angle((direction if goal_yaw is None else goal_yaw)-terminal[:, 2]))/np.pi
    obstacle = 1.-np.clip(clearance/config.clearance_scale, 0., 1.)
    velocity = 1.-search[:, 0]/max_speed
    terms = {'heading': config.heading_weight*heading,
             'clearance': config.clearance_weight*obstacle,
             'velocity': config.speed_weight*velocity}
    return sum(terms.values()), terms
