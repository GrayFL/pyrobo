"""Compatibility imports for raw observations and geometric collision queries."""
from .core.obstacles import PointCloud
from .geometry.collision import PointObstacles

__all__ = ['PointCloud', 'PointObstacles']
