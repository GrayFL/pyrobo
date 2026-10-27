"""Continuous rectangular sweep reduction, independent of a control policy."""
import numpy as np


def swept_clearances(poses, sweep, world, footprint, bounds, valid_time_mask=None):
    count = len(poses)
    if not count:
        return np.empty(0)
    clear = world.clearances(poses, footprint, bounds)
    valid = np.ones_like(clear, dtype=bool) if valid_time_mask is None else valid_time_mask
    # The first stopped endpoint stays valid; only repeated padding is ignored.
    intervals = np.minimum(clear[:, :-1], clear[:, 1:]) - sweep[:, 1:]
    intervals = np.where(valid[:, 1:], intervals, np.inf)
    return np.minimum(np.where(valid, clear, np.inf).min(axis=1), intervals.min(axis=1))
