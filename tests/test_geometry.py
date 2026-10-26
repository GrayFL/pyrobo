import numpy as np
import pytest
from graphmap.pose import Pose

from pyrobo import Bounds2D, Footprint, PointCloud, PointObstacles, pose2d
from pyrobo.types import planar_state
from pyrobo.simulation import Box, World2D


def test_frames_radians_and_original_points():
    raw = np.array([[1.0, 0, 0.2], [1.01, 0, 0.3]])
    cloud = PointCloud(raw, "lidar", 2.0)
    raw[:] = 99
    transform = Pose(translation=[2, 3, 0], rotation=[0, 0, np.pi / 2], degrees=False,
                     wrd_frame="world", ego_frame="lidar")
    np.testing.assert_allclose(cloud.transformed(transform).points[0], [2, 4, .2], atol=1e-10)
    assert cloud.timestamp == 2.0 and not cloud.points.flags.writeable
    np.testing.assert_array_equal(cloud.select_voxels(1).points, [[1, 0, .2]])
    assert len(cloud.points) == 2
    with pytest.raises(ValueError, match="source frame"):
        cloud.transformed(pose2d(0, 0))


@pytest.mark.parametrize("raw", [[[np.nan, 0]], [[np.inf, 0]], [1, 2], [[1]], np.empty((0, 4))])
def test_invalid_cloud_is_not_free_space(raw):
    with pytest.raises(ValueError):
        PointCloud(raw)


@pytest.mark.parametrize("pose", [Pose(scale=2), Pose(rotation=[1, 0, 0]), Pose(z=.2)])
def test_nonplanar_or_scaled_robot_rejected(pose):
    with pytest.raises(ValueError):
        planar_state(pose)


def test_exact_swept_disk_detects_thin_obstacle():
    obstacles = PointObstacles(PointCloud([[.123, 0]]), point_radius=0)
    bounds = Bounds2D(-3, 3, -3, 3)
    assert not obstacles.segment_free([-1, 0], [1, 0], .001, bounds)
    assert obstacles.segment_free([-1, .01], [1, .01], .001, bounds)
    assert not obstacles.segment_free([-2.9, 0], [-2, 0], .2, bounds)


def test_rectangle_corners_rotation_and_height_filter():
    bounds = Bounds2D(-4, 4, -4, 4)
    footprint = Footprint(.8, .8, 0)
    cloud = PointCloud([[.55, 0, 0], [0, 0, 3]])
    obstacles = PointObstacles(cloud, point_radius=0, z_range=(-.1, 1))
    clearance = obstacles.clearances([[0, 0, 0], [0, 0, np.pi / 4]], footprint, bounds)
    assert clearance[0] > 0 and clearance[1] < 0
    assert len(obstacles.cloud.points) == 2


def test_clearance_never_overestimates_bruteforce_geometry():
    rng = np.random.default_rng(10)
    points = rng.uniform(-3, 3, (250, 2))
    states = rng.uniform([-2, -2, -np.pi], [2, 2, np.pi], (50, 3))
    footprint = Footprint(.8, .6, .08)
    obstacles = PointObstacles(PointCloud(points), point_radius=.03)
    actual = obstacles.clearances(states, footprint, Bounds2D(-5, 5, -5, 5))
    for s, value in zip(states, actual):
        q = points - s[:2]
        c, sn = np.cos(s[2]), np.sin(s[2])
        local = q @ np.array([[c, -sn], [sn, c]])
        delta = np.abs(local) - [.4, .3]
        exact = (np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(delta.max(axis=1), 0)).min() - .11
        assert value <= exact + 1e-12
        if exact < 0:
            assert value == pytest.approx(exact)


def test_lidar_first_returns_and_sensor_to_world():
    world = World2D(Bounds2D(-5, 5, -5, 5), [Box(1, 2, -1, 1)])
    pose = pose2d(0, 0, np.pi / 2)
    scan = world.lidar(pose, timestamp=1, beams=4, max_range=6)
    world_scan = scan.transformed(pose)
    assert scan.frame == "base" and world_scan.frame == "world"
    assert np.linalg.norm(world_scan.points - [1, 0, 0], axis=1).min() < 1e-9
    assert not np.any(np.all(np.isclose(world_scan.points, [5, 0, 0]), axis=1))
    assert world.collides(pose2d(1.5, 0), Footprint())
    assert not world.collides(pose, Footprint())


@pytest.mark.parametrize("length,width", [(0.5, 0.5), (1., .5), (1.7, .32)])
def test_arbitrary_rectangles_and_axle_offset(length, width):
    footprint = Footprint(length, width, 0., .2)
    world = World2D(Bounds2D(-4, 4, -4, 4), [Box(.2 + length/2 - .01, 2., -.01, .01)])
    assert world.collides(pose2d(0, 0), footprint)
    obstacle = PointObstacles(PointCloud([[.2 + length/2 - .005, 0]]), point_radius=0)
    assert obstacle.clearances([[0, 0, 0]], footprint, world.bounds)[0] < 0


def test_rectangle_translation_and_rotation_sweep_between_samples():
    footprint = Footprint.long(margin=0)
    bounds = Bounds2D(-4, 4, -4, 4)
    obstacle = PointObstacles(PointCloud([[.123, 0]]), point_radius=0)
    assert not obstacle.rectangle_segment_free([-1, 0], [1, 0], footprint, bounds)
    # End orientations are free but the middle of the turn hits the long corner.
    point = PointObstacles(PointCloud([[.5, .2]]), point_radius=.01)
    assert point.clearances([[0, 0, -np.pi/4], [0, 0, np.pi/2]], footprint, bounds).min() > 0
    assert not point.rotation_free([0, 0], -np.pi/4, np.pi/2, footprint, bounds)


@pytest.mark.parametrize("planner_name", ["rrt", "voronoi"])
def test_long_rectangle_uses_narrow_corridor_rejected_by_outer_circle(planner_name):
    from pyrobo.planning import RRTConnect, RRTConnectConfig, VoronoiConfig, VoronoiPlanner
    footprint = Footprint.long(margin=.02)
    bounds = Bounds2D(-3, 3, -.4, .4)
    obstacles = PointObstacles(World2D(bounds).point_cloud())
    planner = (RRTConnect(RRTConnectConfig(clearance=0), footprint) if planner_name == "rrt"
               else VoronoiPlanner(VoronoiConfig(clearance=0), footprint))
    assert not obstacles.segment_free([-2, 0], [2, 0], footprint.radius, bounds)
    result = planner.plan(pose2d(-2, 0), pose2d(2, 0), obstacles, bounds)
    assert result.success
    assert planner.plan(pose2d(-2, 0, np.pi/2), pose2d(2, 0), obstacles, bounds).status == "invalid_start"

