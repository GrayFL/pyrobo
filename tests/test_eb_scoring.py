"""Analytical spatial-integral oracles and real finite-force rollout contracts."""
from dataclasses import replace

import numpy as np
import pytest

from pyrobo import (Ackermann, ChassisState, DWAConfig, DWAEngine, DynamicsConfig,
                    Footprint, MotionPolicy, MotionRequest, Path, PointCloud,
                    RobotObservation, Twist2D, Bounds2D, pose2d)
from pyrobo.control.eb_dwa import EBDWAConfig, EBReference
from pyrobo.control.eb_dwa.reference import polyline
from pyrobo.control.eb_dwa.scoring import project_trajectories, score_rollout, trajectory_scores
from pyrobo.core.geometry import path_poses


def reference(xy=((0, 0), (5, 0)), *, gear=1, width=.5):
    xy, s = polyline(xy)
    delta = np.diff(xy, axis=0)
    yaw = np.unwrap(np.r_[np.arctan2(delta[:, 1], delta[:, 0]), np.arctan2(*delta[-1, ::-1])])
    return EBReference(xy, s, yaw, np.zeros(len(s)), np.full(len(s), width),
                       np.full(len(s), width), s, gear)


def scores(ref, xy, *, heading=None, times=None, speed=None, config=None, **kwargs):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim == 2:
        xy = xy[None]
    count, steps = xy.shape[:2]
    yaw = np.zeros((count, steps)) if heading is None else np.broadcast_to(heading, (count, steps))
    poses = np.concatenate((xy, yaw[..., None]), axis=2)
    times = np.linspace(0, 2, steps) if times is None else times
    speed = np.full((count, steps), .5) if speed is None else np.broadcast_to(speed, (count, steps))
    return trajectory_scores(ref, poses, speed, times, config or EBDWAConfig(), desired_speed=.5, **kwargs)


def test_same_spatial_motion_at_different_times_has_identical_geometric_cost():
    ref = reference()
    x = np.linspace(0, 3, 61)
    xy = np.column_stack((x, .15*np.sin(x)))
    slow = scores(ref, xy, times=np.linspace(0, 6, len(x)), speed=.5)
    fast = scores(ref, xy, times=np.linspace(0, 3, len(x)), speed=1.)
    for name in ('cross_track', 'heading', 'tube', 'progress', 'backward', 'endpoint'):
        np.testing.assert_allclose(slow.terms[name], fast.terms[name], atol=1e-12)
    assert fast.terms['speed'][0] > slow.terms['speed'][0]


def test_spatial_quadrature_matches_known_integral_across_nonuniform_sampling():
    ref = reference()
    cfg = replace(EBDWAConfig(), cross_track_weight=1.)
    expected = .1**2*3**3/3  # integral of y(s)^2, y=.1*s over [0,3]
    for samples in (31, 101, 501):
        x = 3*np.linspace(0, 1, samples)**1.8
        actual = scores(ref, np.column_stack((x, .1*x)), config=cfg)
        assert actual.terms['cross_track'][0] == pytest.approx(expected, rel=.004)


def test_middle_weaving_is_penalized_even_with_identical_endpoints():
    x = np.linspace(0, 3, 101)
    smooth = np.column_stack((x, np.zeros_like(x)))
    weaving = np.column_stack((x, .35*np.sin(4*np.pi*x/3)))
    result = scores(reference(), [smooth, weaving])
    assert result.costs[1] > result.costs[0]+.3
    assert result.terms['cross_track'][1] > result.terms['cross_track'][0]
    assert result.terms['progress'][0] == result.terms['progress'][1]


def test_standing_still_has_no_progress_reward_and_retreat_is_separately_penalized():
    x = np.linspace(0, 1, 41)
    zero = np.zeros((len(x), 2))
    forward = np.column_stack((x, np.zeros_like(x)))
    backward = np.column_stack((-x, np.zeros_like(x)))
    result = scores(reference(), [zero, forward, backward], speed=0.)
    assert result.terms['progress'][0] == 0
    assert result.terms['progress'][1] < 0 < result.terms['progress'][2]
    assert result.terms['backward'][2] > 0
    assert result.costs[1] < result.costs[0] < result.costs[2]


def test_crossing_projection_stays_on_previous_branch_and_can_move_backwards():
    ref = reference([[-2, 0], [2, 0], [2, 2], [0, 2], [0, -2]])
    x = np.r_[np.linspace(-.5, .1, 30), np.linspace(.1, -.6, 30)]
    points = np.column_stack((x, np.full_like(x, .01)))[None]
    p = project_trajectories(ref, points, start_s=1.5)
    assert np.max(p.s) < 2.2
    assert p.s[0, -1] < p.s[0, 0]
    np.testing.assert_allclose(p.xy[0, :, 1], 0., atol=1e-12)


@pytest.mark.parametrize('gear,heading,speed', [(1, 0., .5), (-1, np.pi, -.5)])
def test_reverse_vehicle_heading_is_not_the_geometric_tangent(gear, heading, speed):
    x = np.linspace(0, 2, 31)
    xy = np.column_stack((x, np.zeros_like(x)))
    good = scores(reference(gear=gear), xy, heading=heading, speed=speed)
    wrong = scores(reference(gear=gear), xy, heading=heading+np.pi, speed=speed)
    assert good.terms['heading'][0] == pytest.approx(0., abs=1e-12)
    assert wrong.terms['heading'][0] > 1.
    assert good.terms['progress'][0] < 0 and good.terms['backward'][0] == 0


def test_narrow_soft_tube_increases_penalty_without_declaring_collision():
    x = np.linspace(0, 2, 31)
    xy = np.column_stack((x, np.full_like(x, .2)))
    wide = scores(reference(width=.4), xy)
    narrow = scores(reference(width=.1), xy)
    assert narrow.terms['tube'][0] > wide.terms['tube'][0] == 0
    assert np.isfinite(narrow.costs).all()  # feasibility belongs to the physical evaluator


def test_control_smoothing_uses_explicit_execution_feedback():
    x = np.linspace(0, 1, 21)
    xy = np.tile(np.column_stack((x, np.zeros_like(x)))[None], (2, 1, 1))
    search = np.array([[.5, -.3], [.5, .3]])
    left = scores(reference(), xy, search=search, previous_command=search[0], command_scale=[1., 1.])
    right = scores(reference(), xy, search=search, previous_command=search[1], command_scale=[1., 1.])
    assert np.argmin(left.costs) == 0 and np.argmin(right.costs) == 1
    np.testing.assert_allclose(left.costs, sum(left.terms.values()))


def test_endpoint_overshoot_cannot_keep_earning_progress():
    x = np.linspace(0, 6, 101)
    result = scores(reference(), np.column_stack((x, np.zeros_like(x))))
    assert result.projection.s[0, -1] == 5.
    assert result.terms['endpoint'][0] == pytest.approx(3.)


def test_invalid_padded_tail_does_not_change_any_cost():
    x = np.linspace(0, 2, 31)
    base = np.column_stack((x, np.zeros_like(x)))
    a = scores(reference(), base)
    padded = np.vstack((base, [[3., .5], [4., .5]]))
    times = np.r_[np.linspace(0, 2, 31), 2.1, 2.2]
    mask = np.array([[True]*31+[False]*2])
    b = scores(reference(), padded, times=times, valid_time_mask=mask)
    for name in a.terms:
        np.testing.assert_allclose(a.terms[name], b.terms[name], atol=1e-12)


def test_duplicate_reference_vertices_are_normalized_and_single_endpoint_is_finite():
    ref = reference([[0, 0], [0, 0], [1, 0], [1, 0], [3, 0]])
    x = np.linspace(0, 2, 30)
    result = scores(ref, np.column_stack((x, np.zeros_like(x))))
    assert np.isfinite(result.costs).all()
    one = EBReference([[0, 0]], [0], [0], [0], [.1], [.1], [0])
    result = scores(one, np.zeros((4, 2)))
    assert np.isfinite(result.costs).all()


def test_real_rollout_retains_collisions_braking_masks_and_native_first_cycle():
    model = Ackermann()
    engine = DWAEngine(DWAConfig(prediction_time=1.), Footprint.long(offset_x=.2), model=model)
    state = ChassisState(pose2d(0, 0), Twist2D(.2, model.yaw_rate(.2, .1)),
                         steering=.1, steering_rate=.03, acceleration=.2)
    cloud = PointCloud(np.column_stack((np.full(61, 1.4), np.linspace(-1, 1, 61))))
    o = RobotObservation(state, cloud, Bounds2D(-3, 5, -3, 3))
    route = Path(path_poses([[0, 0], [4, 0]]))
    request = MotionRequest(MotionPolicy.NAVIGATION, 0., path=route, goal=pose2d(4, 0))
    search = np.array([[0., 0.], [.9, 0.], [.9, .5]])
    batch = engine.evaluate_candidates(o, request, search)
    assert not batch.feasible.all() and batch.feasible.any()
    cfg = EBDWAConfig()
    result = score_rollout(reference(), batch.prediction, engine.plant.codec, cfg, desired_speed=.5)
    costs = np.where(batch.feasible, result.costs, np.inf)
    assert np.all(np.isinf(costs[~batch.feasible]))
    assert 'stop_reachable' in batch.constraint_masks
    assert batch.prediction.states.shape[1] > batch.prediction.prediction_end+1
    first = int(np.flatnonzero(np.isclose(batch.prediction.times, o.dt))[0])
    for row in range(len(search)):
        command = engine.command_space.decode_one(batch.native_commands[row])
        actual = engine.plant.step(o.chassis, command, o.dt).state
        np.testing.assert_allclose(batch.prediction.states[row, first], engine.plant.codec.pack(actual), atol=1e-12)
    np.testing.assert_allclose(batch.prediction.states[..., 4],
                               model.yaw_rate(batch.prediction.states[..., 3], batch.prediction.states[..., 5]), atol=1e-12)


def test_unfinished_braking_is_rejected_even_when_tracking_score_is_good():
    engine = DWAEngine(DWAConfig(braking_timeout=.001), dynamics=DynamicsConfig(brake_force=0.))
    o = RobotObservation(ChassisState(pose2d(0, 0), Twist2D(.7, 0.)),
                         PointCloud(np.empty((0, 2))), Bounds2D(-5, 10, -4, 4))
    request = MotionRequest(MotionPolicy.NAVIGATION, 0.,
                            path=Path(path_poses([[0, 0], [5, 0]])), goal=pose2d(5, 0))
    batch = engine.evaluate_candidates(o, request, np.array([[.8, 0.]]))
    result = score_rollout(reference(), batch.prediction, engine.plant.codec, EBDWAConfig(), desired_speed=.8)
    assert np.isfinite(result.costs).all()
    assert not batch.constraint_masks['stop_reachable'][0]
    assert not batch.feasible[0]
