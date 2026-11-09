"""Pose/time optimization, real vehicle geometry and executable-motion contracts."""
from dataclasses import replace

import numpy as np
import pytest

from pyrobo import (Ackermann, AckermannCommand, Bounds2D, ChassisState, DifferentialDrive,
    DynamicsConfig, Drag, Footprint, MotionPolicy, MotionRequest, Navigator, Path, PointCloud,
    RobotObservation, SafetyAssessment, TEBConfig, TimedElasticBand, Twist2D, WheelCommand, pose2d)
from pyrobo.core.geometry import path_poses, planar_state
from pyrobo.control.teb.band import segment_motion
from pyrobo.simulation import Box, Simulator2D, World2D


def observation(points=(), *, state=None, bounds=None):
    return RobotObservation(state or ChassisState(pose2d(0, 0)),
        PointCloud(np.asarray(points).reshape(-1, 2)), bounds or Bounds2D(-5, 5, -4, 4), .1)


def request(timestamp=0., *, xy=((0, 0), (3, 0)), goal=None):
    return MotionRequest(MotionPolicy.NAVIGATION, timestamp, path=Path(path_poses(xy, 'world')),
                         goal=goal or pose2d(*xy[-1]))


@pytest.mark.parametrize('values', [
    {'dt_min': 0}, {'dt_min': .4}, {'dt_max': .2}, {'min_samples': 2},
    {'max_samples': 3}, {'max_iterations': True}, {'max_iterations': 0},
    {'time_budget': 0}, {'path_weight': -1}, {'obstacle_weight': float('nan')}, {'allow_reverse': 1},
])
def test_invalid_config(values):
    with pytest.raises(ValueError):
        TEBConfig(**values)


@pytest.mark.parametrize('model,command_type', [(DifferentialDrive(), WheelCommand), (Ackermann(), AckermannCommand)])
def test_band_optimizes_space_and_time_and_only_returns_real_plant_trajectory(model, command_type):
    engine = TimedElasticBand(TEBConfig(time_budget=None, max_iterations=18), model=model,
                             footprint=Footprint.long(offset_x=.2))
    o = observation()
    result = engine.optimize(o, request(xy=((0, 0), (1, .25), (3, .25))))
    band = engine.last_band
    assert result.safe and isinstance(result.command, command_type)
    assert band.cost < band.initial_cost
    assert not np.allclose(band.intervals, engine.config.dt_ref)
    assert np.any(np.abs(band.poses[1:-1, 1]) > 1e-5)
    assert engine.config.min_samples <= len(band.poses) <= engine.config.max_samples
    assert np.all((band.intervals >= engine.config.dt_min) & (band.intervals <= engine.config.dt_max))
    assert result.diagnostics.candidate_count <= 16
    assert not band.poses.flags.writeable and not band.intervals.flags.writeable
    np.testing.assert_array_equal(band.poses[0], planar_state(o.chassis.pose))
    executed = engine.plant.step(o.chassis, result.command, o.dt).state
    index = np.flatnonzero(np.isclose(result.trajectory.times, o.dt))[0]
    np.testing.assert_allclose(result.trajectory.states[index], engine.plant.codec.pack(executed), atol=1e-12)
    assert result.trajectory.times[-1] > result.trajectory.times[result.trajectory.prediction_end]
    assert engine.plant.codec.stopped(result.trajectory.states[-1:])[0]
    if isinstance(model, Ackermann):
        states = result.trajectory.states
        np.testing.assert_allclose(states[:, 4], model.yaw_rate(states[:, 3], states[:, 5]), atol=1e-12)


def test_exact_arc_reverse_steering_and_spin():
    angle, radius, duration = -.4, 1.2, .8
    # Reverse rightward yaw at positive steering about the rear axle.
    poses = np.array([[0, 0, 0], [radius*np.sin(angle), radius*(1-np.cos(angle)), angle]])
    v, w, lateral = segment_motion(poses, np.array([duration]))
    assert v[0] == pytest.approx(radius*angle/duration)
    assert w[0] == pytest.approx(angle/duration)
    assert lateral[0] == pytest.approx(0., abs=1e-12)
    assert np.arctan(.5*w[0]/v[0]) > 0
    v, w, _ = segment_motion(np.array([[0, 0, 3.1], [0, 0, -3.1]]), np.array([.1]))
    assert v[0] == 0 and 0 < w[0] < 1


def test_car_reverse_initialization_and_explicit_reverse_disable():
    r = request(xy=((0, 0), (-3, 0)), goal=pose2d(-3, 0, 0))
    reverse = TimedElasticBand(TEBConfig(time_budget=None), model=Ackermann())
    result = reverse.optimize(observation(), r)
    assert result.safe and result.command.speed < 0
    assert abs(result.command.steering) < .01
    forward = TimedElasticBand(TEBConfig(time_budget=None, allow_reverse=False), model=Ackermann())
    assert forward.optimize(observation(), r).command.speed >= 0


def test_obstacle_penalty_moves_the_band_using_the_oriented_footprint():
    engine = TimedElasticBand(TEBConfig(time_budget=None, max_iterations=25))
    engine.optimize(observation([[1., .24], [1.05, .24]]), request())
    assert engine.last_band.cost < engine.last_band.initial_cost
    assert engine.last_band.poses[:, 1].min() < -.08


def test_warm_start_reset_goal_change_and_teleop_handoff():
    engine = TimedElasticBand(TEBConfig(time_budget=None))
    first = engine.optimize(observation(), request())
    state = engine.plant.step(observation().chassis, first.command, .1).state
    o = observation(state=state)
    engine.optimize(o, request(.1))
    assert engine.last_band.warm_started
    engine.optimize(o, request(.1, xy=((0, 0), (3, .2))))
    assert not engine.last_band.warm_started
    manual = MotionRequest(MotionPolicy.TELEOP_STOP, .1, desired=Twist2D(-.3, .1))
    world = engine.index.get(o.obstacles)
    result = engine.optimize(o, manual)
    assert engine.index.get(o.obstacles) is world
    chosen = engine.command_space.from_native_batch(engine.model.encode_command(result.command)[None])[0]
    assert -.3 <= chosen[0] <= 0 and chosen[1] == pytest.approx(.1)
    assert result.diagnostics.extra['optimizer'] == 'dwa' and engine.last_band is None
    engine.optimize(o, request(.1))
    assert not engine.last_band.warm_started
    engine.reset()
    assert engine.last_band is None and engine._last_path is None


def test_soft_budget_expiry_still_certifies_command_and_frame_errors_are_rejected():
    engine = TimedElasticBand(TEBConfig(time_budget=1e-12))
    result = engine.optimize(observation(), request())
    assert result.safe and result.diagnostics.extra['teb_status'] == 'time_budget'
    assert result.diagnostics.extra['teb_evaluations'] == 1
    invalid = replace(request(), path=Path(path_poses([[0, 0], [3, 0]], 'map')), goal=pose2d(3, 0, frame='map'))
    with pytest.raises(ValueError, match='frames'):
        engine.optimize(observation(), invalid)


def test_current_cloud_and_offset_rectangle_are_used_even_with_cached_band():
    engine = TimedElasticBand(TEBConfig(time_budget=None), footprint=Footprint.long(offset_x=.35))
    engine.optimize(observation(), request())
    # Inside the offset front but outside the same rectangle centered on the axle.
    obstacle = [[.8, 0]]
    result = engine.optimize(observation(obstacle, state=ChassisState(pose2d(0, 0), timestamp=.1)), request(.1))
    assert result.command.mode == 'estop' and not result.safe
    assert result.diagnostics.clearance_m < 0
    centered = TimedElasticBand(TEBConfig(time_budget=None), footprint=Footprint.long())
    stop = centered.optimize(observation(obstacle), MotionRequest(MotionPolicy.STOP, 0.))
    assert stop.safe


def test_stale_input_and_unstoppable_chassis_cannot_report_safe():
    engine = TimedElasticBand(TEBConfig(time_budget=None), dynamics=DynamicsConfig(
        brake_force=0, longitudinal_drag=Drag(0, 0, 0)))
    o = observation(state=ChassisState(pose2d(0, 0), Twist2D(.7, 0)))
    result = engine.optimize(o, MotionRequest(MotionPolicy.STOP, 0.))
    assert not result.safe and result.safety_assessment == SafetyAssessment.BRAKING_UNVERIFIED
    stale = engine.optimize(observation(), request(-1.))
    assert not stale.safe and stale.safety_assessment == SafetyAssessment.INPUT_STALE


@pytest.mark.parametrize('model', [DifferentialDrive(), Ackermann()])
@pytest.mark.parametrize('planner', ['rrt_connect', 'voronoi'])
def test_obstacle_detour_arrives_with_offset_long_body_and_full_path(model, planner):
    from pyrobo import NavigationInput, RRTConnectConfig, VoronoiConfig
    world = World2D(Bounds2D(-5, 5, -4, 4), [Box(0., .4, -4., .4)])
    footprint = Footprint.long(offset_x=.2)
    config = RRTConnectConfig(time_budget=1., clearance=.2) if planner == 'rrt_connect' else VoronoiConfig(time_budget=2., clearance=.2)
    nav = Navigator(model=model, footprint=footprint, controller=TEBConfig(time_budget=None), planner=config)
    sim = Simulator2D(world, pose2d(-3, -1), model=model, footprint=footprint, dynamics=nav.dynamics.config)
    cloud, goal, path = world.point_cloud(.08), pose2d(3, 1), None
    for _ in range(350):
        s = sim.state
        out = nav.step(NavigationInput(s.timestamp, s.pose, s.velocity, goal, cloud, world.bounds, .1, s))
        assert out.safe and not sim.collided
        if path is None and len(out.path):
            path = out.path.position.copy()
        if out.status == 'arrived':
            break
        sim.step(out.command, .1)
    assert out.status == 'arrived'
    assert not sim.collided and abs(sim.state.velocity.linear) < 1e-3
    assert np.linalg.norm(sim.state.pose.translation[:2]-goal.translation[:2]) <= nav.settings.goal_tolerance
    np.testing.assert_array_equal(out.path.position, path)
    assert np.max(path[:, 1]) > .8
