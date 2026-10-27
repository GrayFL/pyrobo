"""One immutable observation snapshot owns one cached spatial index."""
from pyrobo.geometry.collision import PointObstacles


class SnapshotIndex:
    def __init__(self, *, point_radius=.03, z_range=None):
        self.point_radius, self.z_range = point_radius, z_range
        self._world = None

    def get(self, cloud):
        if self._world is None or self._world.cloud is not cloud:
            self._world = PointObstacles(cloud, point_radius=self.point_radius, z_range=self.z_range)
        return self._world

    def reset(self):
        self._world = None
