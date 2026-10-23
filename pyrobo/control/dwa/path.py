"""Polyline geometry for the historical DWA arc-length lookahead policy."""
import numpy as np


def path_lookahead(xy, position, distance):
    """Project onto the remaining path, then interpolate an exact arc distance.

    Input is an internal XY polyline. Duplicate points and one-point paths are
    supported. Projection makes the result independent of waypoint density.
    """
    xy = np.asarray(xy, dtype=float)
    if len(xy) == 1:
        return xy[0].copy(), xy.copy()
    vectors = np.diff(xy, axis=0)
    squared = np.einsum("ij,ij->i", vectors, vectors)
    fraction = np.clip(np.einsum("ij,ij->i", position - xy[:-1], vectors) /
                       np.maximum(squared, 1e-30), 0, 1)
    projected = xy[:-1] + fraction[:, None] * vectors
    nearest = int(np.argmin(np.linalg.norm(projected - position, axis=1)))
    remaining = np.vstack((projected[nearest], xy[nearest + 1:]))
    lengths = np.linalg.norm(np.diff(remaining, axis=0), axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    if distance >= cumulative[-1]:
        return remaining[-1].copy(), remaining
    index = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(lengths) - 1)
    ratio = (distance - cumulative[index]) / max(lengths[index], 1e-30)
    return remaining[index] + ratio * (remaining[index + 1] - remaining[index]), remaining


def polyline_distances(points, xy):
    """Exact distance to segments, including sparse paths and repeated vertices."""
    if len(xy) == 1:
        return np.linalg.norm(points - xy[0], axis=1)
    vectors = np.diff(xy, axis=0)
    offsets = points[:, None, :] - xy[None, :-1, :]
    squared = np.einsum("ij,ij->i", vectors, vectors)
    fraction = np.clip(np.einsum("mij,ij->mi", offsets, vectors) / np.maximum(squared, 1e-30), 0, 1)
    return np.linalg.norm(offsets - fraction[..., None] * vectors, axis=2).min(axis=1)
