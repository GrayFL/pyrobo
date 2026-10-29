"""Execution freshness and whole-lease safety, independent of RSim transport."""
from dataclasses import replace

import numpy as np
import pytest

from pyrobo import (Bounds2D, ChassisDynamics, ChassisState, DifferentialDrive, Footprint,
                    PointCloud, RobotObservation, SafetyGuard, Twist2D, pose2d)
from pyrobo.core import MotionResult, MotionStatus, SafetyAssessment
from pyrobo.models.kinematics import command_space_for
from pyrobo.runtime.leases import CommandProposal, LeaseExecutor


def setup():
    model = DifferentialDrive()
    guard = SafetyGuard(ChassisDynamics(model), command_space_for(model), Footprint())
    observation = RobotObservation(ChassisState(pose2d(0, 0), timestamp=10.),
                                   PointCloud(np.empty((0, 2))), Bounds2D(-5, 5, -5, 5), .02)
    proposal = CommandProposal(model.command(.3, 0.), 10., 10.3, 1, 'task-a', 1)
    return LeaseExecutor(guard), observation, proposal


def at(observation, stamp):
    return replace(observation, chassis=replace(observation.chassis, timestamp=stamp), timestamp=stamp)


def test_async_command_is_freshly_checked_and_original_expiry_is_not_renewed():
    gate, obs, proposal = setup()
    first = gate.step(obs, task_epoch='task-a', map_revision=1, proposal=proposal)
    assert first.safe and first.command == proposal.command
    assert first.diagnostics.extra['final_gate_checked']
    held = gate.step(at(obs, 10.2), task_epoch='task-a', map_revision=1)
    assert held.command == proposal.command and gate.active.valid_until == 10.3
    expired = gate.step(at(obs, 10.31), task_epoch='task-a', map_revision=1)
    assert expired.safe and expired.command.mode == 'soft_stop'
    assert gate.active is None and gate.expirations == 1


@pytest.mark.parametrize('change, reason', [
    ({'task_epoch': 'old'}, 'proposal_version_mismatch'),
    ({'map_revision': 2}, 'proposal_version_mismatch'),
    ({'issued_at': 9.7, 'valid_until': 10.}, 'proposal_stale'),
    ({'issued_at': 11., 'valid_until': 11.3}, 'proposal_stale'),
    ({'valid_until': 10.8}, 'proposal_expired'),
])
def test_old_future_or_wrong_version_proposal_does_not_replace_current_lease(change, reason):
    gate, obs, proposal = setup()
    gate.step(obs, task_epoch='task-a', map_revision=1, proposal=proposal)
    result = gate.step(at(obs, 10.05), task_epoch='task-a', map_revision=1,
                       proposal=replace(proposal, sequence=2, **change))
    assert result.command == proposal.command
    assert gate.rejected == 1 and gate.last_rejection == reason
    assert gate.active.sequence == 1


def test_task_change_retires_old_commands_even_when_delayed_result_arrives():
    gate, obs, proposal = setup()
    gate.step(obs, task_epoch='task-a', map_revision=1, proposal=proposal)
    result = gate.step(at(obs, 10.05), task_epoch='task-b', map_revision=1, proposal=proposal)
    assert result.command.mode == 'soft_stop' and gate.active is None


def test_sensor_age_stops_execution_despite_fresh_proposal():
    gate, obs, proposal = setup()
    result = gate.step(replace(obs, timestamp=10.2), task_epoch='task-a', map_revision=1,
                       proposal=proposal)
    assert result.command.mode == 'estop'
    assert result.diagnostics.reason == 'state_stale' and not result.safe


def test_pause_does_not_accept_or_renew_motion():
    gate, obs, proposal = setup()
    result = gate.step(obs, task_epoch='task-a', map_revision=1, proposal=proposal, enabled=False)
    assert result.command.mode == 'soft_stop' and gate.active is None


def test_whole_lease_is_checked_even_when_one_short_execution_cycle_would_be_safe():
    gate, obs, proposal = setup()
    model = gate.guard.space.model
    obs = replace(obs, chassis=replace(obs.chassis, velocity=Twist2D(.4, 0.)),
                  obstacles=PointCloud(np.column_stack((np.full(50, .5), np.linspace(-1, 1, 50)))))
    command = model.command(.9, 0.)
    candidate = MotionResult(command, MotionStatus.TRACKING, SafetyAssessment.SAFE_UNDER_MODEL, None)
    assert gate.guard.enforce(obs, candidate).command == command
    result = gate.step(obs, task_epoch='task-a', map_revision=1, proposal=replace(proposal, command=command))
    assert result.safe and result.command.mode == 'estop' and gate.active is None


def test_delayed_measurement_is_advanced_through_sent_commands_and_actuator_expiry():
    gate, obs, proposal = setup()
    obs = replace(obs, dt=.04)
    gate.step(obs, task_epoch='task-a', map_revision=1, proposal=proposal)
    delayed = replace(obs, timestamp=10.08)
    predicted = gate._current_state(delayed).chassis
    expected = gate.guard.plant.step(obs.chassis, proposal.command, .04).state
    expected = gate.guard.plant.step(expected, gate.guard.space.decode_one([0., 0.], 'estop'), .04).state
    np.testing.assert_allclose(gate.guard.plant.pack(predicted), gate.guard.plant.pack(expected), atol=1e-12)
    assert predicted.timestamp == pytest.approx(delayed.timestamp)
    # Extrapolation must not make a stale source look fresh indefinitely.
    stale = gate.step(replace(obs, timestamp=10.2), task_epoch='task-a', map_revision=1)
    assert stale.diagnostics.reason == 'state_stale' and not stale.safe


def test_command_history_is_bounded_and_version_change_keeps_physical_history():
    gate, obs, proposal = setup()
    for i in range(80):
        stamp = 10+i*.02
        gate.step(at(obs, stamp), task_epoch='task-a' if i < 40 else 'task-b', map_revision=1)
    assert 1 <= len(gate._executed) <= 7


def test_ackermann_timeout_brakes_without_losing_measured_steering_preparation():
    from pyrobo import Ackermann
    model = Ackermann()
    gate = LeaseExecutor(SafetyGuard(ChassisDynamics(model), command_space_for(model), Footprint()))
    state = ChassisState(pose2d(0, 0), Twist2D(.2, float(model.yaw_rate(.2, .4))), steering=.4, timestamp=10.)
    obs = RobotObservation(state, PointCloud(np.empty((0, 2))), Bounds2D(-5, 5, -5, 5), .1)
    proposal = CommandProposal(model.command(.3, .4), 10., 10.3, 1, 'task-a', 1)
    gate.step(obs, task_epoch='task-a', map_revision=1, proposal=proposal)
    result = gate.step(at(obs, 10.31), task_epoch='task-a', map_revision=1)
    assert result.safe and result.command.speed == 0 and result.command.steering == .4
    assert result.diagnostics.reason == 'proposal_timeout_hold_steering'
    assert gate.active is None and gate.expirations == 1
