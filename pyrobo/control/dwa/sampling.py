"""Array-only sampling; arbitrary search dimension and stable candidate order."""
import numpy as np


def sample_window(window, counts):
    dim = len(window.lower)
    if len(counts) != dim or any(n < 2 for n in counts):
        raise ValueError('one sample count >=2 is required per search axis')
    if np.any(window.lower > window.upper):
        return np.empty((0, dim))
    axes = [np.unique(np.r_[np.linspace(lo, hi, count), np.clip(0., lo, hi)])
            for lo, hi, count in zip(window.lower, window.upper, counts)]
    grids = np.meshgrid(*axes, indexing='ij')
    grid = np.column_stack([axis.ravel() for axis in grids])
    anchors = np.asarray(window.anchors).reshape(-1, dim)
    valid = np.all((anchors >= window.lower - 1e-10) & (anchors <= window.upper + 1e-10), axis=1)
    return np.unique(np.vstack((grid, anchors[valid])), axis=0)
