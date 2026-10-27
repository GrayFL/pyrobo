"""Reversible arc-length progress without deleting historical waypoints."""
from dataclasses import dataclass

import numpy as np
from graphmap.pose import Pose, PoseArray

from pyrobo.core.geometry import path_poses, planar_state, pose2d, positive


@dataclass(frozen=True)
class PathProgress:
    target: Pose
    remaining: PoseArray
    distance: float
    remaining_distance: float
    segment: int
    target_segment: int


class PathTracker:
    """Project locally in path order, with symmetric forward/backward recovery.

    At intersections, the previous arc position breaks spatial ties. The search
    interval grows with actual displacement, so a reverse move restores earlier
    segments while a nearby crossing cannot jump to a distant future waypoint.
    Call reset after discontinuous relocalization; set_path when replanning.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.xy = np.empty((0, 2))
        self.cumulative = np.zeros(0)
        self.distance = None
        self.position = None

    def set_path(self, path: PoseArray):
        xy = np.asarray(path.position)[:, :2]
        if xy.ndim != 2 or not len(xy) or not np.isfinite(xy).all():
            raise ValueError("path must be a nonempty planar PoseArray")
        if not path.wrd_frame:
            raise ValueError("path must have an explicit world frame")
        self.reset()
        self.xy = xy.copy()
        self.frame, self.body = path.wrd_frame, path.ego_frame
        self.lengths = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        self.cumulative = np.r_[0., np.cumsum(self.lengths)]

    def sample(self, distance):
        if len(self.xy) == 1:
            return self.xy[0].copy(), 0
        index = min(max(0, int(np.searchsorted(self.cumulative, distance, side="right")) - 1), len(self.lengths) - 1)
        fraction = np.clip((distance - self.cumulative[index]) / max(self.lengths[index], 1e-30), 0, 1)
        return self.xy[index] + fraction * (self.xy[index + 1] - self.xy[index]), index

    def update(self, pose: Pose, lookahead: float) -> PathProgress:
        positive(lookahead, "lookahead", zero=True)
        position = planar_state(pose)[:2]
        if not len(self.xy):
            raise ValueError("set_path before tracking")
        if pose.wrd_frame != self.frame:
            raise ValueError("pose and path frames must match")
        if len(self.xy) == 1:
            progress = 0.
        else:
            vectors = np.diff(self.xy, axis=0)
            fraction = np.clip(np.einsum("ij,ij->i", position - self.xy[:-1], vectors)
                               / np.maximum(self.lengths**2, 1e-30), 0, 1)
            arcs = self.cumulative[:-1] + fraction * self.lengths
            if self.distance is not None:
                window = 2 * np.linalg.norm(position - self.position) + .25
                lo, hi = max(0., self.distance - window), min(self.cumulative[-1], self.distance + window)
                arcs = np.maximum(self.cumulative[:-1], np.minimum(self.cumulative[1:], np.clip(arcs, lo, hi)))
                valid = (arcs >= lo - 1e-10) & (arcs <= hi + 1e-10)
                fraction = (arcs - self.cumulative[:-1]) / np.maximum(self.lengths, 1e-30)
            else:
                valid = np.ones(len(arcs), dtype=bool)
            projected = self.xy[:-1] + fraction[:, None] * vectors
            errors = np.linalg.norm(projected - position, axis=1)
            errors[~valid] = np.inf
            # Small hysteresis favors the current branch at equal-distance crossings.
            if self.distance is not None:
                errors += .01 * np.abs(arcs - self.distance)
            progress = float(arcs[np.argmin(errors)])
        self.distance, self.position = progress, position.copy()
        projection, segment = self.sample(progress)
        target, target_segment = self.sample(min(self.cumulative[-1], progress + lookahead))
        direction = self.xy[min(target_segment + 1, len(self.xy) - 1)] - self.xy[target_segment]
        yaw = float(np.arctan2(direction[1], direction[0]))
        remaining = np.vstack((projection, self.xy[segment + 1:]))
        return PathProgress(pose2d(*target, yaw, frame=self.frame, body=self.body),
                            path_poses(remaining, self.frame, self.body), progress,
                            float(self.cumulative[-1] - progress), segment, target_segment)


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
