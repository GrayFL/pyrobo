import numpy as np
import pytest

from pyrobo import Bounds2D, Footprint, PointCloud, PointObstacles, Twist2D, pose2d
from pyrobo.control import DWAConfig, DynamicWindow
from pyrobo.control.dwa.path import path_lookahead, polyline_distances
from pyrobo.planning import RRTConnect, RRTConnectConfig
from pyrobo.types import path_poses
from pyrobo.simulation import Box, World2D, demo_world


def test_rrt_finds_detour_and_all_edges_are_valid():
    world, start, goal = demo_world()
    obstacles = PointObstacles(world.point_cloud())
    planner = RRTConnect(RRTConnectConfig(time_budget=2.0))
    result = planner.plan(start, goal, obstacles, world.bounds)
    assert result.success
    xy = result.path.position[:, :2]
    np.testing.assert_allclose(xy[0], start.translation[:2])
    np.testing.assert_allclose(xy[-1], goal.translation[:2])
    assert len(xy) > 2
    assert all(obstacles.segment_free(a, b, planner.radius, world.bounds) for a, b in zip(xy[:-1], xy[1:]))
    planner.reset()
    replay = planner.plan(start, goal, obstacles, world.bounds)
    np.testing.assert_allclose(replay.path.position, result.path.position)


def test_rrt_bounded_search_does_not_report_unreachability():
    world = World2D(Bounds2D(0, 8, 0, 6), [Box(3.8, 4.2, 0, 6)])
    planner = RRTConnect(RRTConnectConfig(max_extensions=30, time_budget=1))
    result = planner.plan(pose2d(1, 3), pose2d(7, 3), PointObstacles(world.point_cloud()), world.bounds)
    assert result.status == "budget_exhausted"
    assert result.extensions <= 30 and result.path.shape == (0,)


def test_rrt_invalid_start_and_empty_cloud():
    obstacles = PointObstacles(PointCloud(np.empty((0, 2))))
    planner = RRTConnect()
    bounds = Bounds2D(-5, 5, -5, 5)
    assert planner.plan(pose2d(-5, 0), pose2d(2, 0), obstacles, bounds).status == "invalid_start"
    assert planner.plan(pose2d(0, 0), pose2d(2, 0), obstacles, bounds).success


def test_dwa_acceleration_limits_and_braking_tail():
    bounds = Bounds2D(-5, 5, -5, 5)
    obstacles = PointObstacles(PointCloud(np.empty((0, 2))))
    dwa = DynamicWindow()
    v = Twist2D(.6, .5)
    result = dwa.step(pose2d(0, 0), v, pose2d(3, 0), path_poses([[0, 0], [3, 0]]), obstacles, bounds, .1)
    assert result.safe
    assert abs(result.command.linear - v.linear) <= dwa.config.linear_acceleration * .1 + 1e-12
    assert abs(result.command.angular - v.angular) <= dwa.config.angular_acceleration * .1 + 1e-12
    assert len(result.trajectory.position) > int(dwa.config.prediction_time / dwa.config.integration_dt)
    stopped = dwa.stop(pose2d(0, 0), v, obstacles, bounds, .1)
    assert stopped.command.linear == pytest.approx(.51)
    assert stopped.command.angular == pytest.approx(.25)
    assert stopped.safe


def test_unavoidable_collision_is_explicit_not_a_false_safe_stop():
    obstacle = PointObstacles(PointCloud([[.55, 0]]), point_radius=.03)
    dwa = DynamicWindow()
    result = dwa.stop(pose2d(0, 0), Twist2D(.9, 0), obstacle, Bounds2D(-5, 5, -5, 5), .1)
    assert not result.safe
    assert result.command.linear == pytest.approx(.81)
    assert result.clearance < 0


def test_sweep_rejects_an_obstacle_between_coarse_samples():
    dwa = DynamicWindow(DWAConfig(integration_dt=.5), Footprint(.02, .02, 0))
    trajectories = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=float)
    obstacles = PointObstacles(PointCloud([[.5, 0]]), point_radius=.01)
    bounds = Bounds2D(-3, 3, -3, 3)
    assert np.all(obstacles.clearances(trajectories, dwa.footprint, bounds) > 0)
    assert dwa._clearance(trajectories, np.array([[0, .5]]), obstacles, bounds)[0] < 0


def test_lookahead_interpolates_sparse_corners_and_ignores_duplicate_points():
    path = np.array([[0, 0], [0, 0], [2, 0], [2, 4]], dtype=float)
    target, remaining = path_lookahead(path, np.array([1., .2]), 2.)
    np.testing.assert_allclose(target, [2, 1])
    np.testing.assert_allclose(remaining[0], [1, 0])
    target, _ = path_lookahead(np.array([[2., 1.], [2., 1.]]), np.zeros(2), 1.)
    np.testing.assert_allclose(target, [2, 1])
    np.testing.assert_allclose(polyline_distances(np.array([[1., .2], [2., 3.]]), path), [.2, 0])


def test_turn_hysteresis_brakes_before_release_and_reset_is_reproducible():
    dwa = DynamicWindow()
    bounds = Bounds2D(-5, 5, -5, 5)
    obstacles = PointObstacles(PointCloud(np.empty((0, 2))))
    path, goal = path_poses([[0, 0], [3, 0]]), pose2d(3, 0)
    first = dwa.step(pose2d(0, 0, np.pi), Twist2D(), goal, path, obstacles, bounds, .1)
    assert first.mode == "turning" and first.safe and first.command.linear == 0
    assert abs(first.command.angular) > 0
    turning = dwa.step(pose2d(0, 0, -.5), Twist2D(0, .3), goal, path, obstacles, bounds, .1)
    assert turning.mode == "turning" and turning.command.linear == 0
    braking = dwa.step(pose2d(0, 0, -.1), Twist2D(0, .3), goal, path, obstacles, bounds, .1)
    assert braking.mode == "turn_braking" and braking.command.angular == pytest.approx(.05)
    following = dwa.step(pose2d(0, 0, -.1), Twist2D(), goal, path, obstacles, bounds, .1)
    assert following.mode == "following" and following.command.linear > 0
    dwa.reset()
    assert dwa.step(pose2d(0, 0, np.pi), Twist2D(), goal, path, obstacles, bounds, .1).command == first.command


def test_heading_speed_cap_is_acceleration_limited():
    dwa = DynamicWindow()
    result = dwa.step(pose2d(0, 0, np.pi), Twist2D(.8, 0), pose2d(3, 0), path_poses([[0, 0], [3, 0]]),
                      PointObstacles(PointCloud(np.empty((0, 2)))), Bounds2D(-5, 5, -5, 5), .1)
    assert result.safe and result.command.linear == pytest.approx(.71)


def test_turning_cannot_bypass_rectangle_or_braking_checks():
    dwa = DynamicWindow(footprint=Footprint(.8, .8, 0))
    obstacles = PointObstacles(PointCloud([[.55, 0], [-.55, 0]]), point_radius=0)
    bounds = Bounds2D(-5, 5, -5, 5)
    result = dwa.step(pose2d(0, 0), Twist2D(), pose2d(-2, 0), path_poses([[0, 0], [-2, 0]]),
                      obstacles, bounds, .1)
    assert result.safe
    assert np.all(obstacles.clearances(
        np.column_stack((result.trajectory.position[:, :2],
                         np.arctan2(result.trajectory.rot_mat[:, 1, 0], result.trajectory.rot_mat[:, 0, 0]))),
        dwa.footprint, bounds) > 0)
    assert abs(result.command.angular) <= dwa.config.angular_acceleration * .1

