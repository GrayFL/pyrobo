"""Editable polygon solids; planning still receives surface point samples."""
import numpy as np

from pyrobo.core import Footprint
from pyrobo.simulation import World2D


def cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def edges_intersect(a, b, c, d):
    """Broadcast segment intersection, including touching and collinear overlap."""
    ab, cd = b-a, d-c
    side1, side2 = cross(ab, c-a), cross(ab, d-a)
    side3, side4 = cross(cd, a-c), cross(cd, b-c)
    overlap = np.all((np.maximum(a, b) >= np.minimum(c, d)-1e-10) &
                     (np.maximum(c, d) >= np.minimum(a, b)-1e-10), axis=-1)
    return overlap & (side1*side2 <= 1e-12) & (side3*side4 <= 1e-12)


def contains(polygon, points):
    """Odd/even rule supports concave simple polygons; edges count as inside."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    p = np.asarray(points)[:, None, :]
    edge = b-a
    relative = p-a
    on_edge = ((np.abs(cross(edge, relative)) <= 1e-10) &
               (np.sum(relative*edge, axis=-1) >= -1e-10) &
               (np.sum(relative*edge, axis=-1) <= np.sum(edge*edge, axis=-1)+1e-10)).any(axis=1)
    y = p[..., 1]
    crosses = (a[:, 1] > y) != (b[:, 1] > y)
    intersection = a[:, 0] + (y-a[:, 1])*edge[:, 0]/np.where(np.abs(edge[:, 1]) > 1e-12, edge[:, 1], 1.)
    return on_edge | (np.sum(crosses & (p[..., 0] < intersection), axis=1) % 2 == 1)


def validate_polygon(points):
    polygon = np.asarray(points, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not 3 <= len(polygon) <= 256:
        raise ValueError('each polygon needs 3–256 XY vertices')
    if not np.isfinite(polygon).all():
        raise ValueError('polygon coordinates must be finite')
    ends = np.roll(polygon, -1, axis=0)
    if np.any(np.linalg.norm(ends-polygon, axis=1) < 1e-6):
        raise ValueError('polygon has duplicate adjacent vertices')
    if abs(np.sum(cross(polygon, ends))) < 1e-6:
        raise ValueError('polygon has zero area')
    hits = edges_intersect(polygon[:, None], ends[:, None], polygon[None], ends[None])
    i, j = np.indices(hits.shape)
    non_adjacent = (i != j) & ((i-j) % len(polygon) != 1) & ((j-i) % len(polygon) != 1)
    if np.any(hits & non_adjacent):
        raise ValueError('polygon self-intersects; split it into simple polygons')
    return polygon


def point_segment_distance2(points, a, b):
    edge = b-a
    t = np.clip(np.sum((points-a)*edge, axis=-1)/np.sum(edge*edge, axis=-1), 0., 1.)
    return np.sum((points-a-t[..., None]*edge)**2, axis=-1)


def capsule_outline(a, b, radius):
    """Surface approximation for point sampling; collision truth remains analytic."""
    yaw = np.arctan2(*(b-a)[::-1])
    first = yaw+np.linspace(np.pi/2, 3*np.pi/2, 17)
    second = yaw+np.linspace(-np.pi/2, np.pi/2, 17)
    return np.concatenate((a+radius*np.column_stack((np.cos(first), np.sin(first))),
                           b+radius*np.column_stack((np.cos(second), np.sin(second)))))


class PolygonWorld(World2D):
    def __init__(self, world, polygons, polylines=()):
        super().__init__(world.bounds, world.boxes, frame=world.frame)
        self.polygons = tuple(validate_polygon(p) for p in polygons)
        self.polylines = tuple((np.asarray(line.points, dtype=float), line.width/2) for line in polylines)
        self.wall_outlines = tuple(capsule_outline(a, b, radius) for points, radius in self.polylines
                                   for a, b in zip(points[:-1], points[1:]))

    def segments(self):
        segments = [super().segments()]
        segments.extend(np.stack((p, np.roll(p, -1, axis=0)), axis=1)
                        for p in (*self.polygons, *self.wall_outlines))
        return np.concatenate(segments)

    def collides(self, pose, footprint, *, padding=0.):
        if super().collides(pose, footprint, padding=padding):
            return True
        robot = Footprint(footprint.length+2*padding, footprint.width+2*padding, 0., footprint.offset_x).corners(pose)
        ends = np.roll(robot, -1, axis=0)
        for polygon in self.polygons:
            if np.any(robot.max(axis=0) < polygon.min(axis=0)) or np.any(polygon.max(axis=0) < robot.min(axis=0)):
                continue
            if (contains(polygon, robot).any() or contains(robot, polygon).any() or
                    edges_intersect(robot[:, None], ends[:, None], polygon[None],
                                    np.roll(polygon, -1, axis=0)[None]).any()):
                return True
        for points, radius in self.polylines:
            if (np.any(robot.max(axis=0) < points.min(axis=0)-radius) or
                    np.any(points.max(axis=0)+radius < robot.min(axis=0))):
                continue
            a, b = points[:-1], points[1:]
            if contains(robot, points).any() or edges_intersect(
                    robot[:, None], ends[:, None], a[None], b[None]).any():
                return True
            # Exact distance from each thick wall segment (capsule) to rectangle.
            if (np.any(point_segment_distance2(robot[:, None], a[None], b[None]) <= radius**2) or
                    np.any(point_segment_distance2(points[:, None], robot[None], ends[None]) <= radius**2)):
                return True
        return False
