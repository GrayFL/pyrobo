import numpy as np
import pytest

from pyrobo import pose2d, Twist2D
from pyrobo.dynamics import ChassisDynamics, ChassisState, Drag, DynamicsConfig, implicit_axis
from pyrobo.kinematics import Ackermann, AckermannCommand, DifferentialDrive, WheelCommand
from pyrobo.types import planar_state


def evolve(plant, state, command, seconds=3):
    for _ in range(round(seconds / .05)):
        result = plant.step(state, command, .05)
        state = result.state
    return result


def test_differential_spins_but_ackermann_only_steers_at_rest():
    diff, ack = ChassisDynamics(), ChassisDynamics(Ackermann())
    initial = ChassisState(pose2d(0, 0))
    spin = evolve(diff, initial, diff.model.command(0, .7)).state
    assert abs(planar_state(spin.pose)[2]) > 1
    np.testing.assert_allclose(spin.pose.translation, 0, atol=1e-12)
    steered = evolve(ack, initial, AckermannCommand(0, .4)).state
    assert steered.steering == pytest.approx(.4, abs=.002)
    assert steered.velocity == Twist2D()
    np.testing.assert_allclose(steered.pose.matrix, initial.pose.matrix)
    with pytest.raises(ValueError, match="rotate in place"):
        ack.model.from_twist(Twist2D(0, 1))


@pytest.mark.parametrize("speed", [.6, -.4])
def test_ackermann_kinematic_constraint_and_reverse_sign(speed):
    plant = ChassisDynamics(Ackermann())
    state = evolve(plant, ChassisState(pose2d(0, 0)), AckermannCommand(speed, .3)).state
    assert state.velocity.linear == pytest.approx(speed, abs=.005)
    assert state.velocity.angular == pytest.approx(speed * np.tan(.3) / .5, abs=.003)
    assert np.sign(planar_state(state.pose)[2]) == np.sign(speed)


@pytest.mark.parametrize("speed", [-2., -.001, 0., .001, 2.])
def test_implicit_passive_drag_cannot_reverse_or_add_energy(speed):
    _, next_speed = implicit_axis(0., speed, 0., 1., 0., Drag(100, 20, 10), 10.)
    assert abs(next_speed) <= abs(speed)
    assert next_speed * speed >= 0


def test_low_motor_force_cannot_instantly_reach_speed_target():
    cfg = DynamicsConfig(drive_force=1, longitudinal_drag=Drag(0, 0, 0))
    plant = ChassisDynamics(config=cfg)
    result = evolve(plant, ChassisState(pose2d(0, 0)), plant.model.command(.8), 1)
    assert 0 < result.state.velocity.linear <= cfg.drive_force / cfg.mass + 1e-10
    assert abs(result.force) <= 1 and "effort_limited" in result.flags


@pytest.mark.parametrize("model", [DifferentialDrive(), Ackermann()])
def test_estop_does_not_reverse_and_coast_is_passive(model):
    plant = ChassisDynamics(model)
    for speed in (-.4, .7):
        state = ChassisState(pose2d(0, 0), Twist2D(speed))
        coast = plant.step(state, model.command(mode="coast"), .2)
        assert 0 < coast.state.velocity.linear * speed < speed * speed
        assert coast.force == coast.torque == 0
        for _ in range(20):
            result = plant.step(state, model.command(mode="estop"), .1)
            assert result.state.velocity.linear * speed >= 0
            state = result.state
        assert state.velocity.linear == 0


def test_steering_recenters_when_coasting_and_diff_has_no_heading_spring():
    ack = ChassisDynamics(Ackermann())
    state = ChassisState(pose2d(0, 0, 1), steering=.4)
    result = evolve(ack, state, AckermannCommand(mode="coast"))
    assert abs(result.state.steering) < .03
    assert planar_state(result.state.pose)[2] == pytest.approx(1)
    diff = ChassisDynamics()
    state = evolve(diff, ChassisState(pose2d(0, 0, 1)), WheelCommand(mode="coast")).state
    assert planar_state(state.pose)[2] == pytest.approx(1)


@pytest.mark.parametrize("model", [DifferentialDrive(), Ackermann()])
def test_numpy_batch_matches_scalar_plant_including_actuator_state(model):
    plant = ChassisDynamics(model)
    state = ChassisState(pose2d(1, -2, .3, frame="map", body="axle"))
    commands = [model.command(.7, .3), model.command(-.3, -.2), model.command()]
    batch = np.tile(plant.pack(state), (len(commands), 1))
    raw = np.stack([model.encode_command(c) for c in commands])
    predicted = plant.integrate(batch, raw, .2)
    for i, command in enumerate(commands):
        actual = plant.step(state, command, .2).state
        np.testing.assert_allclose(predicted[i], plant.pack(actual), atol=1e-10)
        assert actual.pose.wrd_frame == "map" and actual.pose.ego_frame == "axle"


def test_timeout_brakes_from_actual_state_and_models_reject_wrong_commands():
    plant = ChassisDynamics()
    state = ChassisState(pose2d(0, 0), Twist2D(.5))
    result = plant.step(state, WheelCommand(.8, .8), .1, command_age=1)
    assert result.state.velocity.linear < .5
    assert "command_timeout" in result.flags
    with pytest.raises(TypeError):
        plant.step(state, AckermannCommand(), .1)
    with pytest.raises(ValueError, match="agree"):
        ChassisDynamics(Ackermann()).pack(ChassisState(pose2d(0, 0), Twist2D(0, 1)))


def test_discrete_servo_jerk_and_effort_limits():
    plant = ChassisDynamics()
    state = ChassisState(pose2d(0, 0))
    for _ in range(100):
        result = plant.step(state, plant.model.command(.7, .4), .02)
        assert abs(result.state.acceleration - state.acceleration) <= plant.config.linear_jerk * .02 + 1e-8
        assert abs(result.force) <= plant.config.drive_force
        assert abs(result.torque) <= plant.config.yaw_torque
        state = result.state


@pytest.mark.parametrize("model", [DifferentialDrive(), Ackermann()])
def test_steady_drive_balances_drag_and_soft_stop_is_not_reverse_propulsion(model):
    plant = ChassisDynamics(model)
    result = evolve(plant, ChassisState(pose2d(0, 0)), model.command(.6), 6.)
    assert result.state.velocity.linear == pytest.approx(.6, abs=1e-7)
    assert result.force == pytest.approx(plant.config.longitudinal_drag(.6), abs=1e-5)
    state = result.state
    for _ in range(120):
        state = plant.step(state, model.command(mode='soft_stop'), .05).state
        assert state.velocity.linear >= 0
    assert state.velocity.linear == 0


def test_generalized_axis_constant_force_terminal_speed():
    drag = Drag(0., 2., 1.)
    speed = 0.
    for _ in range(200):
        _, speed = implicit_axis(0., speed, 8., 1., 0., drag, .05)
    assert speed == pytest.approx(2., abs=1e-8)  # 2*v + v**2 = 8


@pytest.mark.parametrize("model", [DifferentialDrive(), Ackermann()])
def test_reference_limits_are_distinct_from_actual_state_and_force_limits(model):
    plant = ChassisDynamics(model)
    result = plant.step(ChassisState(pose2d(0, 0)), model.command(20., 2.), .1)
    assert 'reference_limited' in result.flags
    assert result.state.velocity.linear < .1
