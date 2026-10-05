"""Raw point observations and optional, explicit point selection. No occupancy grid."""
from dataclasses import dataclass

import numpy as np
from graphmap.pose import Pose
from scipy.spatial import cKDTree

from .types import Bounds2D, Footprint, positive


@dataclass(frozen=True)
class PointCloud:
    """A snapshot of obstacle returns, with explicit frame and optional acquisition time.

    Nx2 is lifted to z=0; Nx3 is preserved. Invalid points raise instead of silently
    turning sensor failure into free space. Empty (0, 2)/(0, 3) is a valid observation.
    """
    points: np.ndarray
    frame: str = "world"
    timestamp: float | None = None

    def __post_init__(self):
        p = np.asarray(self.points, dtype=float)
        if p.ndim != 2 or p.shape[1] not in (2, 3) or not np.isfinite(p).all():
            raise ValueError("points must be a finite (N, 2) or (N, 3) array")
        if not isinstance(self.frame, str) or not self.frame:
            raise ValueError("point frame must be a nonempty string")
        if self.timestamp is not None and not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        p = np.column_stack((p, np.zeros(len(p)))) if p.shape[1] == 2 else p.copy()
        p.setflags(write=False)
        object.__setattr__(self, "points", p)

    def transformed(self, transform: Pose) -> "PointCloud":
        """Transform with T_destination_source, honoring graphmap's frame labels."""
        if not isinstance(transform, Pose) or transform.ego_frame != self.frame or not transform.wrd_frame:
            raise ValueError("transform must name this source frame and a destination frame")
        if not np.isclose(transform.scale, 1.0, atol=1e-9, rtol=0) or not np.isfinite(transform.matrix).all():
            raise ValueError("sensor transform must be finite and rigid")
        return PointCloud(transform.apply(self.points), transform.wrd_frame, self.timestamp)

    def select_voxels(self, size: float) -> "PointCloud":
        """Opt-in decimation: keep the first original point per 3D voxel, never a centroid.

        The original cloud is unchanged. Increase obstacle padding for discarded
        coverage (up to sqrt(3)*size), or retain the original for collision checks.
        """
        positive(size, "voxel size")
        _, indices = np.unique(np.floor(self.points / size), axis=0, return_index=True)
        return PointCloud(self.points[np.sort(indices)], self.frame, self.timestamp)


class PointObstacles:
    """KD-tree index of a cloud's XY projection; the source XYZ cloud is retained."""
    def __init__(self, cloud: PointCloud, *, point_radius: float = 0.03,
                 z_range: tuple[float, float] | None = None):
        positive(point_radius, "point_radius", zero=True)
        self.cloud = cloud
        self.point_radius = point_radius
        mask = np.ones(len(cloud.points), dtype=bool)
        if z_range is not None:
            if len(z_range) != 2 or not np.isfinite(z_range).all() or z_range[0] > z_range[1]:
                raise ValueError("z_range must be a finite ordered pair")
            mask = (cloud.points[:, 2] >= z_range[0]) & (cloud.points[:, 2] <= z_range[1])
        self.xy = np.array(cloud.points[mask, :2], copy=True)
        self.xy.setflags(write=False)
        self.tree = cKDTree(self.xy)

    def segment_free(self, a, b, radius: float, bounds: Bounds2D) -> bool:
        """Exact segment-to-point distances for a swept disk (no edge sampling gaps)."""
        a, b = np.asarray(a), np.asarray(b)
        if not bounds.contains(a, radius) or not bounds.contains(b, radius):
            return False
        radius = radius + self.point_radius
        d = b - a
        length2 = float(d @ d)
        ids = self.tree.query_ball_point((a + b) / 2, np.sqrt(length2) / 2 + radius)
        if not ids:
            return True
        q = self.xy[ids] - a
        t = np.clip(q @ d / length2, 0, 1) if length2 > 1e-20 else np.zeros(len(q))
        return bool(np.all(np.linalg.norm(q - t[:, None] * d, axis=1) > radius))

    def clearances(self, states: np.ndarray, footprint: Footprint, bounds: Bounds2D) -> np.ndarray:
        """Signed distance from every oriented rectangle to points and workspace edges.

        Query a circumscribed neighborhood only where the nearest-point lower bound
        is small. Farther-away values are conservative lower bounds, sufficient for
        both collision rejection and the clearance cost.
        """
        states = np.asarray(states, dtype=float)
        shape = states.shape[:-1]
        s = states.reshape(-1, 3)
        xy, yaw = s[:, :2], s[:, 2]
        c, sn = np.cos(yaw), np.sin(yaw)
        hx, hy = footprint.length / 2, footprint.width / 2
        ex, ey = np.abs(c) * hx + np.abs(sn) * hy, np.abs(sn) * hx + np.abs(c) * hy
        bound = np.minimum.reduce([xy[:, 0] - bounds.xmin - ex, bounds.xmax - xy[:, 0] - ex,
                                   xy[:, 1] - bounds.ymin - ey, bounds.ymax - xy[:, 1] - ey])
        radius = np.hypot(hx, hy)
        nearest = self.tree.query(xy)[0]
        clearance = nearest - radius - self.point_radius
        near = np.flatnonzero(clearance < 0.25 + footprint.margin)
        if len(near):
            neighbors = self.tree.query_ball_point(xy[near], radius + self.point_radius + footprint.margin + 0.25)
            lengths = np.fromiter(map(len, neighbors), dtype=int, count=len(near))
            if lengths.sum():
                rows = np.repeat(near, lengths)
                ids = np.concatenate(neighbors).astype(int)
                q = self.xy[ids] - xy[rows]
                local = np.column_stack((c[rows] * q[:, 0] + sn[rows] * q[:, 1],
                                         -sn[rows] * q[:, 0] + c[rows] * q[:, 1]))
                delta = np.abs(local) - [hx, hy]
                signed = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(axis=1), 0)
                # Outside this search ball clearance is >= margin + 0.25.
                clearance[near] = footprint.margin + 0.25
                np.minimum.at(clearance, rows, signed - self.point_radius)
        return (np.minimum(clearance, bound) - footprint.margin).reshape(shape)

