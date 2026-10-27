"""Immutable raw obstacle snapshots and physical workspace boundaries."""
from dataclasses import dataclass
import numpy as np
from graphmap.pose import Pose
from .geometry import positive

@dataclass(frozen=True)
class Bounds2D:
    """Physical workspace boundary, not just sampling limits."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self):
        if not np.isfinite([self.xmin, self.xmax, self.ymin, self.ymax]).all():
            raise ValueError("bounds must be finite")
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("bounds must have positive area")

    def contains(self, xy, inset: float = 0.0):
        xy = np.asarray(xy)
        return ((xy[..., 0] >= self.xmin + inset) & (xy[..., 0] <= self.xmax - inset)
                & (xy[..., 1] >= self.ymin + inset) & (xy[..., 1] <= self.ymax - inset))

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

