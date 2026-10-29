"""Preferred clearance may adapt; physical margins and explicit floors may not."""
import numpy as np
import pytest

from pyrobo import Bounds2D, Footprint, PointCloud, pose2d
from pyrobo.geometry.collision import PointObstacles
from pyrobo.planning import AdaptiveClearancePlanner, RRTConnect, RRTConnectConfig


def test_adaptive_clearance_opens_a_narrow_route_without_shrinking_body():
    body = Footprint(.5, .5, .08)
    cfg = RRTConnectConfig(clearance=1., time_budget=.5)
    bounds, cloud = Bounds2D(-2, 8, -1, 1), PointObstacles(PointCloud(np.empty((0, 2))))
    start, goal = pose2d(-1, 0), pose2d(7, 0)
    assert RRTConnect(cfg, body).plan(start, goal, cloud, bounds).status == 'invalid_start'
    planner = AdaptiveClearancePlanner(RRTConnect, cfg, body)
    result = planner.plan(start, goal, cloud, bounds)
    assert result.success and planner.footprint == body
    assert 0 < result.diagnostics['effective_clearance'] < cfg.clearance
    assert cloud.path_free(result.path.position[:, :2], body, bounds, start_yaw=0., goal_yaw=0.)


def test_explicit_clearance_floor_cannot_be_relaxed_even_if_that_prevents_a_route():
    body = Footprint(.5, .5, .08)
    planner = AdaptiveClearancePlanner(RRTConnect, RRTConnectConfig(clearance=1., time_budget=.5),
                                       body, minimum_clearance=.7)
    result = planner.plan(pose2d(-1, 0), pose2d(7, 0), PointObstacles(PointCloud(np.empty((0, 2)))),
                          Bounds2D(-2, 8, -1, 1))
    assert result.status == 'invalid_start'
    assert all(a['clearance'] >= .7 for a in result.diagnostics['clearance_attempts'])
    assert planner.safety_footprint.margin == body.margin+.7


@pytest.mark.parametrize('seed', [0, 7, 19])
def test_bridge_sampling_narrow_route_has_valid_sweeps_and_replays_after_reset(seed):
    from examples.navigation_benchmark.scenarios import scenarios
    from examples.navigation_benchmark.planning import solid_path_valid
    case = scenarios()['narrow_ackermann']
    body = Footprint.compact()
    planner = RRTConnect(RRTConnectConfig(seed=seed, clearance=.08, time_budget=2.), body)
    world = PointObstacles(case.world.point_cloud(.04))
    # This checks sampling itself without the navigator's terminal entry segment.
    result = planner.plan(case.start, case.goal, world, case.world.bounds)
    assert result.success and result.diagnostics['bridge_samples'] > 0
    assert solid_path_valid(case, result.path.position[:, :2], body)
    planner.reset()
    replayed = planner.plan(case.start, case.goal, world, case.world.bounds)
    np.testing.assert_array_equal(result.path.position, replayed.path.position)
