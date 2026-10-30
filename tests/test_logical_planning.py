"""Logical RRT budgets do not use host duration as a search termination rule."""
from dataclasses import replace
import importlib
import numpy as np
import pytest

from pyrobo import Bounds2D, Footprint, PointObstacles, pose2d
from pyrobo.planning import AdaptiveClearancePlanner, RRTConnect, RRTConnectConfig
from pyrobo.simulation import Box, World2D, demo_world


def test_logical_rrt_replays_detour_under_different_wall_timings(monkeypatch):
    module = importlib.import_module('pyrobo.planning.rrt_connect.planner')
    world, start, goal = demo_world()
    obstacles = PointObstacles(world.point_cloud())
    config = RRTConnectConfig(time_budget=None, max_extensions=1800, seed=7)
    planner = RRTConnect(config)
    result = planner.plan(start, goal, obstacles, world.bounds)
    assert result.success and len(result.path) > 2
    ticks = iter([0., 1_000_000.])
    monkeypatch.setattr(module, 'perf_counter', lambda: next(ticks))
    planner.reset()
    repeated = planner.plan(start, goal, obstacles, world.bounds)
    np.testing.assert_array_equal(result.path.position, repeated.path.position)
    assert repeated.extensions == result.extensions
    assert repeated.elapsed == 1_000_000.


def test_logical_rrt_exhausts_exact_extension_budget_and_keeps_wall_default(monkeypatch):
    world = World2D(Bounds2D(0, 8, 0, 6), [Box(3.8, 4.2, 0, 6)])
    obstacles = PointObstacles(world.point_cloud())
    config = RRTConnectConfig(time_budget=None, max_extensions=31)
    result = RRTConnect(config).plan(pose2d(1, 3), pose2d(7, 3), obstacles, world.bounds)
    assert result.status == 'budget_exhausted' and result.extensions == 31
    assert RRTConnectConfig().time_budget == .045
    with pytest.raises(ValueError, match='wall time_budget'):
        AdaptiveClearancePlanner(RRTConnect, config, Footprint())
    with pytest.raises(ValueError):
        replace(config, time_budget=0.)
