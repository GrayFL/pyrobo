"""Numerical contracts, vectorized mappings and request-specific hard constraints."""
from dataclasses import replace

import numpy as np
import pytest

from pyrobo import (Ackermann, Bounds2D, ChassisDynamics, ChassisState, DifferentialDrive,
                    Drag, DWAConfig, DynamicsConfig, Footprint, PointCloud, Twist2D, pose2d)
from pyrobo.core import MotionPolicy, MotionRequest, RobotObservation, SafetyAssessment
from pyrobo.control.dwa import DWAEngine
from pyrobo.models.kinematics import command_space_for
from pyrobo.models.dynamics.reachability import ReachabilityEstimator
from pyrobo.teleop import TeleopAdapter
from pyrobo.adapters import TwistCommandAdapter


def observation(*, model=None, state=None, points=(), timestamp=0., bounds=None):
    return RobotObservation(state or ChassisState(pose2d(0, 0), timestamp=timestamp),
                            PointCloud(np.asarray(points).reshape(-1, 2)),
                            bounds or Bounds2D(-5, 5, -5, 5))


def request(v=.6, turn=0., *, policy=MotionPolicy.TELEOP_STOP, steering=None, timestamp=0.):
    return MotionRequest(policy, timestamp, desired=Twist2D(v, turn), steering=steering)


@pytest.mark.parametrize('model', [DifferentialDrive(), Ackermann()])
@pytest.mark.parametrize('count', [0, 1, 1024])
def test_mapping_and_integrator_shapes_and_roundtrip(model, count):
    space, plant = command_space_for(model), ChassisDynamics(model)
    rng = np.random.default_rng(19)
    lo, hi = space.bounds()
    search = rng.uniform(lo, hi, size=(count, 2))
    native, valid = space.to_native_batch(search)
    assert native.shape == (count, 2) and valid.shape == (count,)
    np.testing.assert_allclose(space.from_native_batch(native), search, atol=1e-15)
    states = np.tile(plant.pack(ChassisState(pose2d(0, 0))), (count, 1))
    result, diagnostics = plant.integrate(states, native, .1, diagnostics=True)
    assert result.shape == (count, 10)
    assert all(mask.shape == (count,) for mask in diagnostics.flags.values())
    np.testing.assert_array_equal(result, plant.integrate(states, native, .1))
    if count:
        np.testing.assert_array_equal(result[-1:], plant.integrate(states[-1:], native[-1:], .1))


def test_mapping_limits_spin_and_car_zero_speed_intent():
    diff = command_space_for(DifferentialDrive(max_wheel_speed=.5))
    native, valid = diff.to_native_batch(np.array([[0, 1.], [.5, 1.], [-.3, -.2]]))
    assert valid.tolist() == [True, False, True]
    assert native[0, 0] == -native[0, 1]
    car = command_space_for(Ackermann())
    adapter = TeleopAdapter(car)
    with pytest.raises(ValueError, match='steering intent'):
        adapter.submit(Twist2D(0, .5), 0.)
    r = adapter.submit(Twist2D(.5, .2), 0.)
    stopped = adapter.submit(Twist2D(0, 0), .1)
    assert stopped.steering == r.steering != 0
    reverse = adapter.submit(Twist2D(-.5, .2), .2)
    assert reverse.steering < 0
    with pytest.raises(ValueError):
        TwistCommandAdapter(car).convert(car.model.command(0, .3))
    with pytest.raises(ValueError):
        TwistCommandAdapter(diff).convert(diff.model.command(0, 0, mode='estop'))


@pytest.mark.parametrize('model', [DifferentialDrive(), Ackermann()])
@pytest.mark.parametrize('v', [.65, -.4])
def test_reachability_uses_actual_track_response_and_does_not_truncate_references(model, v):
    cfg = DynamicsConfig(longitudinal_drag=Drag(150., 80., 40.), drive_force=80., brake_force=150.)
    plant, space = ChassisDynamics(model, cfg), command_space_for(model)
    state = ChassisState(pose2d(0, 0), Twist2D(v, 0), acceleration=.4)
    packed = plant.pack(state)
    estimator = ReachabilityEstimator(plant, space)
    window = estimator.command_window(packed, request(v), .1)
    assert window.envelope is not None and not window.envelope.strict_outer_bound
    np.testing.assert_array_equal(window.lower, space.bounds()[0])
    np.testing.assert_array_equal(window.upper, space.bounds()[1])
    env = window.envelope
    expected = plant.integrate(np.tile(packed, (len(env.native_probes), 1)), env.native_probes, .1, mode='track')
    np.testing.assert_array_equal(env.next_states, expected)
    estop = estimator.state_envelope(packed, .1, 'estop')
    assert np.max(np.abs(estop.next_states[:, 3] - env.next_states[0, 3])) > 1e-5
    for command, actual in zip(env.native_probes, env.next_states):
        executed = plant.step(state, model.decode_command(command), .1).state
        np.testing.assert_allclose(actual, plant.pack(executed), atol=1e-12)


def test_estimator_failure_expands_to_full_allowed_domain(monkeypatch):
    plant, space = ChassisDynamics(), command_space_for(DifferentialDrive())
    estimator = ReachabilityEstimator(plant, space)
    def fail(*args, **kwargs):
        raise FloatingPointError('uncertain estimate')
    monkeypatch.setattr(estimator, 'state_envelope', fail)
    lower, upper = np.array([0., -.2]), np.array([.6, .2])
    window = estimator.command_window(plant.pack(ChassisState(pose2d(0, 0))), request(), .1, lower=lower, upper=upper)
    assert window.envelope is None and window.fallback_reason == 'FloatingPointError'
    np.testing.assert_array_equal(window.lower, lower)
    np.testing.assert_array_equal(window.upper, upper)


def test_per_row_diagnostics_do_not_poison_other_candidates():
    plant = ChassisDynamics(DifferentialDrive())
    states = np.tile(plant.pack(ChassisState(pose2d(0, 0))), (3, 1))
    commands = np.array([[0, 0], [8, 8], [.1, .1]])
    result, report = plant.integrate(states, commands, .1, diagnostics=True)
    assert report.flags['reference_limited'].tolist() == [False, True, False]
    for i in range(3):
        one, flags = plant.integrate(states[i:i+1], commands[i:i+1], .1, diagnostics=True)
        np.testing.assert_array_equal(result[i], one[0])
        assert all(mask[i] == flags.flags[name][0] for name, mask in report.flags.items())


@pytest.mark.parametrize('model', [DifferentialDrive(), Ackermann()])
@pytest.mark.parametrize('policy', [MotionPolicy.TELEOP_STOP, MotionPolicy.TELEOP_AVOID])
@pytest.mark.parametrize('v', [.6, -.4, 0.])
def test_teleop_hard_intent_and_empty_candidate_batches(model, policy, v):
    engine = DWAEngine(model=model)
    r = request(v, .2 if isinstance(model, DifferentialDrive) else 0., policy=policy,
                steering=.25 if isinstance(model, Ackermann) else None)
    o = observation()
    result = engine.optimize(o, r)
    assert result.safe
    selected = engine.command_space.from_native_batch(model.encode_command(result.command)[None])[0]
    assert min(0, v)-1e-9 <= selected[0] <= max(0, v)+1e-9
    desired = engine.command_space.desired_coordinates(r)
    if policy == MotionPolicy.TELEOP_STOP or v == 0:
        assert selected[1] == pytest.approx(desired[1])
    else:
        assert abs(selected[1]-desired[1]) <= r.max_turn_deviation+1e-9
    batch = engine.evaluate_candidates(o, r, np.empty((0, 2)))
    assert batch.costs.shape == batch.feasible.shape == (0,)


@pytest.mark.parametrize('model', [DifferentialDrive(), Ackermann()])
def test_chunked_selection_is_identical_and_selected_trajectory_has_time(model):
    o = observation(points=[[1.3, 0]])
    r = request(policy=MotionPolicy.TELEOP_AVOID)
    whole = DWAEngine(model=model).optimize(o, r)
    chunked = DWAEngine(DWAConfig(batch_size=13), model=model).optimize(o, r) if isinstance(model, DifferentialDrive) else \
        DWAEngine(DWAConfig(batch_size=13, prediction_time=2.4, lookahead=1.6), model=model).optimize(o, r)
    assert whole.command == chunked.command
    assert whole.diagnostics.candidate_count == chunked.diagnostics.candidate_count
    assert whole.diagnostics.feasible_count == chunked.diagnostics.feasible_count
    assert whole.diagnostics.extra['candidate_id'] == chunked.diagnostics.extra['candidate_id']
    np.testing.assert_allclose(whole.trajectory.states, chunked.trajectory.states, atol=1e-12)
    np.testing.assert_allclose(whole.trajectory.times, chunked.trajectory.times, atol=1e-12)
    assert np.all(np.diff(whole.trajectory.times) > 0)


def test_stop_preserves_spin_or_explicitly_overrides_when_rectangle_cannot_rotate():
    engine = DWAEngine(footprint=Footprint.long())
    # Obstacle just outside the side, inside the rectangle's swept corner arc.
    o = observation(points=[[.4, .4]])
    result = engine.optimize(o, request(0., 1.2))
    assert result.command.mode == 'estop' and result.diagnostics.extra['safety_override']
    assert result.diagnostics.reason == 'no_feasible_candidate'
    assert result.diagnostics.feasible_count == 0


def test_unsafe_braking_is_never_labelled_safe():
    engine = DWAEngine(dynamics=DynamicsConfig(brake_force=0, longitudinal_drag=Drag(0, 0, 0)))
    o = observation(state=ChassisState(pose2d(0, 0), Twist2D(.7, 0)))
    result = engine.optimize(o, request(0))
    assert not result.safe and result.safety_assessment == SafetyAssessment.BRAKING_UNVERIFIED
