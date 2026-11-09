"""Independent EB geometry, using the same saved worlds as the Web experiments."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from pyrobo import (Ackermann, Bounds2D, ChassisState, Footprint, Path as Route,
                    PointCloud, RobotObservation, pose2d)
from pyrobo.apps.web_sim.scene import Scene, SceneSession
from pyrobo.control.eb_dwa import EBDWAConfig, EBPathRepairer
from pyrobo.control.eb_dwa.reference import GeometricCurve, validate_curve
from pyrobo.core.geometry import path_poses
from pyrobo.geometry.collision import PointObstacles
from pyrobo.simulation import Box, World2D


CASES = Path(__file__).resolve().parents[1]/'pyrobo/apps/scenes'


def case(name, **settings):
    data = json.loads((CASES/f'eb_dwa_{name}.json').read_text())
    # Phase 1 tests geometry independently. Phase 3 installs the same explicit
    # experiment fields in SceneSession for browser/CLI execution.
    experiment = data.pop('experiment')
    session = SceneSession(Scene.model_validate(data))
    cfg = EBDWAConfig(time_budget_s=None, min_coverage_m=12., max_evaluations=900, **settings)
    repairer = EBPathRepairer(cfg, session.footprint, model=session.model)
    observation = RobotObservation(session.sim.state, session.cloud, session.world.bounds)
    route = Route(path_poses(experiment['reference_path']))
    return session, repairer, observation, route


def solid_clear(session, reference):
    curve = reference.curve
    u = curve.samples(.02)
    xy, v = curve.evaluate(u), curve.evaluate(u, 1)
    yaw = np.arctan2(v[:, 1], v[:, 0])+(np.pi if reference.gear < 0 else 0.)
    assert not any(session.world.collides(pose2d(*p, a), session.footprint,
                                         padding=session.footprint.margin) for p, a in zip(xy, yaw))


@pytest.mark.parametrize('settings', [
    {'minimum_clearance': .3}, {'max_evaluations': 0}, {'max_nodes': True},
    {'warm_start': 1}, {'topology_attempts': 6}, {'time_budget_s': 0},
    {'node_spacing_m': float('nan')}, {'min_coverage_m': 20.}, {'batch_size': -1},
])
def test_configuration_rejects_unbounded_or_inconsistent_settings(settings):
    with pytest.raises(ValueError):
        EBDWAConfig(**settings)


def test_straight_reference_skips_optimization_and_reuses_verified_cache():
    _, repairer, o, route = case('straight')
    first = repairer.update(o, route)
    assert first.success and not first.diagnostics['eb_optimized']
    np.testing.assert_allclose(first.reference.xy[:, 1], 0., atol=1e-12)
    moved = replace(o, chassis=ChassisState(pose2d(.1, 0), timestamp=.1), timestamp=.1)
    second = repairer.update(moved, route)
    assert second.success and second.diagnostics['eb_cache_hit']
    assert second.diagnostics['eb_evaluations'] == 0
    np.testing.assert_allclose(second.reference.xy[0], [.1, 0])
    assert not second.reference.xy.flags.writeable


@pytest.mark.parametrize('name', ['corner', 'obtuse', 'zigzag'])
def test_empty_rough_path_is_actively_smoothed_into_a_continuous_tangent_curve(name):
    session, repairer, o, route = case(name)
    result = repairer.update(o, route)
    assert result.success, result.diagnostics
    assert result.diagnostics['eb_optimized']
    assert result.diagnostics['eb_active_nodes'] < result.diagnostics['eb_nodes']
    reference = result.reference
    curve = reference.curve
    nodes = curve.parameter[1:-1]
    before, after = curve.evaluate(nodes-1e-8, 1), curve.evaluate(nodes+1e-8, 1)
    np.testing.assert_allclose(before, after, atol=1e-5)
    np.testing.assert_array_equal(reference.xy[[0, -1]], route.xy[[0, -1]])
    assert np.max(abs(reference.curvature)) < 1.5
    if name == 'zigzag':
        assert np.max(abs(reference.xy[:, 1])) < .08
    solid_clear(session, reference)


@pytest.mark.parametrize('name,wall_end', [('single_obstacle', 4.), ('long_wall', 6.5), ('two_obstacles', 6.)])
def test_conflict_repair_rejoins_after_all_obstacles_with_verified_tail(name, wall_end):
    session, repairer, o, route = case(name)
    result = repairer.update(o, route)
    assert result.success, result.diagnostics
    assert result.diagnostics['eb_conflict_count'] >= 1
    assert result.diagnostics['eb_rejoin_s'] > wall_end+session.footprint.length/2
    assert result.reference.global_s[-1]-result.diagnostics['eb_rejoin_s'] >= repairer.config.rejoin_validation_m
    assert result.diagnostics['eb_evaluations'] <= repairer.config.max_evaluations
    assert any(a['topology'] != 'nominal' for a in result.diagnostics['eb_attempts'])
    solid_clear(session, result.reference)


def test_clearance_only_defect_moves_path_away_from_wall():
    session, repairer, o, route = case('wall_clearance')
    result = repairer.update(o, route)
    assert result.success, result.diagnostics
    assert result.diagnostics['eb_conflict_count'] == 0
    assert result.diagnostics['eb_triggers']['insufficient_clearance'] > 0
    middle = (result.reference.xy[:, 0] > 2) & (result.reference.xy[:, 0] < 6)
    assert result.reference.xy[middle, 1].mean() > .06
    solid_clear(session, result.reference)


def test_smoothing_a_wall_corner_must_not_cut_into_the_solid():
    session, repairer, o, route = case('corner_wall')
    result = repairer.update(o, route)
    assert result.success, result.diagnostics
    assert result.diagnostics['eb_verified_clearance_m'] > 0.
    solid_clear(session, result.reference)


def test_offset_ackermann_corner_respects_minimum_radius_over_entire_curve():
    session, repairer, o, route = case('ackermann_corner')
    result = repairer.update(o, route)
    assert result.success, result.diagnostics
    limit = 1/session.model.minimum_turn_radius
    assert result.diagnostics['eb_curvature_bound'] <= limit
    assert np.max(abs(result.reference.curvature)) <= limit
    solid_clear(session, result.reference)


def test_ackermann_reverse_keeps_geometric_tangent_separate_from_body_heading():
    session, repairer, o, route = case('ackermann_reverse')
    result = repairer.update(o, route)
    assert result.success and result.reference.gear == -1, result.diagnostics
    assert np.all(np.cos(result.reference.yaw) < -.99)
    assert np.max(abs(result.reference.curvature)) < 1e-9
    solid_clear(session, result.reference)


def test_too_narrow_ackermann_turn_and_hard_clearance_fail_explicitly():
    _, repairer, o, route = case('ackermann_impossible')
    result = repairer.update(o, route)
    assert not result.success and result.reference is None
    world = World2D(Bounds2D(-2, 10, -2, 2), [Box(-1, 9, .45, 2), Box(-1, 9, -2, -.45)])
    o = RobotObservation(ChassisState(pose2d(0, 0)), world.point_cloud(), world.bounds)
    cfg = EBDWAConfig(time_budget_s=None, minimum_clearance=.12)
    result = EBPathRepairer(cfg).update(o, Route(path_poses([[0, 0], [8, 0]])))
    assert not result.success and result.status == 'invalid_start'


def test_no_safe_rejoin_does_not_return_first_isolated_clear_point():
    world = World2D(Bounds2D(-2, 8, -3, 3), [Box(2, 6.5, -.5, .5)])
    o = RobotObservation(ChassisState(pose2d(0, 0)), world.point_cloud(), world.bounds)
    result = EBPathRepairer(EBDWAConfig(time_budget_s=None)).update(o, Route(path_poses([[0, 0], [7, 0]])))
    assert not result.success and result.status == 'no_safe_rejoin'


def test_new_cloud_invalidates_old_path_and_repairs_without_reusing_unsafe_reference():
    session, repairer, o, route = case('straight')
    old = repairer.update(o, route)
    world = World2D(session.world.bounds, [Box(3, 4, -.5, .5)])
    changed = replace(o, obstacles=world.point_cloud(), timestamp=.1,
                      chassis=replace(o.chassis, timestamp=.1))
    result = repairer.update(changed, route)
    assert old.success and result.success, result.diagnostics
    assert not result.diagnostics['eb_cache_hit']
    assert result.reference.xy[:, 1].max() > .7
    assert result.diagnostics['eb_verified_clearance_m'] > 0


def test_budget_exhaustion_never_exposes_an_unverified_curve():
    _, repairer, o, route = case('single_obstacle')
    repairer.config = replace(repairer.config, time_budget_s=1e-12, max_evaluations=1)
    result = repairer.update(o, route)
    assert not result.success and result.reference is None
    assert result.diagnostics['eb_evaluations'] <= 1


def test_intermediate_collision_is_detected_between_safe_waypoints():
    world = PointObstacles(PointCloud([[1., 0.]]))
    curve = GeometricCurve([[0, 0], [2, 0]])
    body, bounds = Footprint(.2, .2), Bounds2D(-1, 3, -2, 2)
    assert np.all(world.clearances([[0, 0, 0], [2, 0, 0]], body, bounds) > 0)
    validation = validate_curve(curve, world, body, bounds)
    assert not validation.valid and validation.reason == 'collision'


def test_crossing_progress_and_real_reverse_keep_global_branch_memory():
    route = Route(path_poses([[-2, 0], [2, 0], [2, 2], [0, 2], [0, -2]]))
    repairer = EBPathRepairer(EBDWAConfig(time_budget_s=1e-12))
    cloud, bounds = PointCloud(np.empty((0, 2))), Bounds2D(-4, 4, -4, 4)
    for tick, x in enumerate(np.r_[np.linspace(-2, .1, 15), np.linspace(.1, -1, 10)]):
        o = RobotObservation(ChassisState(pose2d(x, .01), timestamp=tick*.1), cloud, bounds)
        repairer.update(o, route)
        assert repairer.tracker.distance < 2.2
    assert repairer.tracker.distance < 1.1
    np.testing.assert_array_equal(repairer.tracker.xy, route.xy)


def test_reset_goal_change_and_time_reversal_clear_incompatible_cache():
    _, repairer, o, route = case('straight')
    repairer.update(o, route, goal=pose2d(9, 0))
    changed = repairer.update(o, route, goal=pose2d(9, 0, .2))
    assert not changed.diagnostics['eb_warm_started']
    later = replace(o, chassis=replace(o.chassis, timestamp=.2), timestamp=.2)
    repairer.update(later, route, goal=pose2d(9, 0, .2))
    earlier = replace(o, chassis=replace(o.chassis, timestamp=.1), timestamp=.1)
    rewound = repairer.update(earlier, route, goal=pose2d(9, 0, .2))
    assert not rewound.diagnostics['eb_warm_started']
    repairer.reset()
    assert repairer._reference is None and not len(repairer.tracker.xy)


def test_short_deterministic_budgets_continue_without_publishing_unsafe_iterates():
    session, repairer, o, route = case('ackermann_corner')
    repairer.config = replace(repairer.config, max_evaluations=90, max_refinement_cycles=12)
    results = []
    for tick in range(13):
        current = replace(o, chassis=replace(o.chassis, timestamp=tick*.1), timestamp=tick*.1)
        result = repairer.update(current, route)
        results.append(result)
        assert result.diagnostics['eb_evaluations'] <= 90
        if result.success:
            assert result.diagnostics['eb_verified_clearance_m'] > 0.
            assert np.max(abs(result.reference.curvature)) < 1/session.model.minimum_turn_radius
        if result.success and not result.diagnostics['eb_solver_pending']:
            break
    assert not results[0].success
    assert results[0].diagnostics['eb_solver_pending']
    assert results[-1].success and not results[-1].diagnostics['eb_solver_pending']
    assert results[-1].diagnostics['eb_warm_started']


def test_fixed_nodes_outside_active_region_and_rolling_warm_start_keep_route_memory():
    _, repairer, o, route = case('corner')
    result = repairer.update(o, route)
    curve = result.reference.curve
    regions = result.diagnostics['eb_active_intervals']
    fixed = np.array([not any(a <= s <= b for a, b in regions) for s in curve.parameter])
    expected = np.column_stack((np.minimum(curve.parameter, 5.), np.maximum(curve.parameter-5., 0.)))
    np.testing.assert_allclose(curve.xy[fixed], expected[fixed], atol=1e-12)
    moved = replace(o, chassis=ChassisState(pose2d(.1, 0), timestamp=.1), timestamp=.1)
    rolled = repairer.update(moved, route)
    assert rolled.success and rolled.diagnostics['eb_warm_started']
    np.testing.assert_array_equal(repairer.tracker.xy, route.xy)
    np.testing.assert_array_equal(rolled.reference.xy[0], [.1, 0])


def test_duplicate_waypoints_and_resampling_do_not_create_spurious_defects():
    _, repairer, o, route = case('straight')
    dense = Route(path_poses([[0, 0], [0, 0], [.15, 0], [.2, 0], [.2, 0], [.4, 0], [9, 0]]))
    result = repairer.update(o, dense)
    assert result.success and not result.diagnostics['eb_optimized']
    np.testing.assert_allclose(result.reference.xy[:, 1], 0.)
