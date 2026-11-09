"""Speed references checked against independent geometry and actual plant responses."""
from dataclasses import replace

import numpy as np
import pytest

from pyrobo import (Ackermann, Bounds2D, ChassisDynamics, ChassisState, DifferentialDrive,
                    DynamicsConfig, EBDWAConfig, Footprint, PointCloud, RobotObservation,
                    Twist2D, pose2d)
from pyrobo.control.eb_dwa import EBReference
from pyrobo.control.eb_dwa.speed import SpeedPlanner
from pyrobo.models.dynamics.prediction import BatchPredictor
from pyrobo.models.kinematics import command_space_for


def setup(model=None, dynamics=None, **config):
    model = model or DifferentialDrive()
    plant = ChassisDynamics(model, dynamics)
    o = RobotObservation(ChassisState(pose2d(0, 0)), PointCloud(np.empty((0, 2))),
                         Bounds2D(-20, 20, -20, 20), .1)
    return SpeedPlanner(plant, command_space_for(model), Footprint(), EBDWAConfig(**config)), o


def reference(curvature=0., *, length=6., gear=1):
    s = np.linspace(0, length, 151)
    k = np.broadcast_to(curvature, s.shape)
    yaw = np.r_[0., np.cumsum((k[:-1]+k[1:])*.5*np.diff(s))]
    mid = (yaw[:-1]+yaw[1:])/2
    xy = np.vstack(([0., 0.], np.cumsum(np.diff(s)[:, None]*np.column_stack((np.cos(mid), np.sin(mid))), axis=0)))
    return EBReference(xy, s, yaw, k, np.ones(len(s)), np.ones(len(s)), s, gear)


@pytest.mark.parametrize('model', [DifferentialDrive(), Ackermann()])
def test_heterogeneous_initial_states_match_independent_predictions(model):
    planner, o = setup(model)
    plant = planner.plant
    states, commands = [], []
    for v, turn in [(0., 0.), (.6, .2), (-.3, -.1)]:
        state = replace(o.chassis, velocity=Twist2D(v, turn if planner.space.can_spin else model.yaw_rate(v, turn)),
                        steering=0. if planner.space.can_spin else turn, acceleration=.2)
        states.append(plant.codec.pack(state))
        commands.append(model.encode_command(model.command(v/2, turn)))
    states, commands = np.asarray(states), np.asarray(commands)
    batch = planner.predictor.predict(states, commands, o.dt, diagnostics=True)
    for i in range(len(states)):
        single = planner.predictor.predict(states[i], commands[i:i+1], o.dt, diagnostics=True)
        np.testing.assert_allclose(batch.states[i, :len(single.times)], single.states[0], atol=1e-12)
        np.testing.assert_allclose(batch.times[:len(single.times)], single.times, atol=1e-12)
        np.testing.assert_array_equal(batch.valid_time_mask[i, :len(single.times)], single.valid_time_mask[0])
        assert batch.stopped[i] == single.stopped[0]
        for key in single.constraint_flags:
            assert batch.constraint_flags[key][i] == single.constraint_flags[key][0]
    broadcast = planner.predictor.predict(states[0:1], commands, o.dt)
    old = planner.predictor.predict(states[0], commands, o.dt)
    np.testing.assert_array_equal(broadcast.states, old.states)
    with pytest.raises(ValueError):
        planner.predictor.predict(states[:2], commands, o.dt)


@pytest.mark.parametrize('model', [DifferentialDrive(), Ackermann()])
def test_straight_accelerates_from_measured_speed_and_brakes_before_endpoint(model):
    planner, o = setup(model)
    ref = reference()
    moving = replace(o, chassis=replace(o.chassis, velocity=Twist2D(.2, 0.)))
    profile = planner.update(ref, moving, terminal=True)
    assert profile.values[0] == pytest.approx(.2)
    assert profile.values[40:100].mean() > .8
    assert profile.values[-1] == 0.
    assert profile.values[-8] < profile.values[-25]  # anticipatory, before final sample
    assert not profile.values.flags.writeable
    assert not profile.diagnostics['eb_speed_table_cached']
    assert planner.update(ref, moving).diagnostics['eb_speed_table_cached']
    planner.update(reference(gear=-1), moving)
    assert len(planner._tables) == 2
    planner.update(ref, replace(moving, dt=.05))
    assert len(planner._tables) == 1
    planner.reset()
    assert not planner._tables


def test_friction_and_yaw_limits_reduce_curved_speed_but_not_free_straight_cruise():
    planner, o = setup(dynamics=DynamicsConfig(friction=.03))
    # Supply already moving measured state so limited drive force at this low
    # friction does not mask the independent lateral speed preference.
    o = replace(o, chassis=replace(o.chassis, velocity=Twist2D(.7, 0.)))
    curved = planner.update(reference(1.2), o)
    straight = planner.update(reference(), o)
    limit = np.sqrt(.03*9.81*planner.config.speed_lateral_fraction/1.2)
    assert curved.values.max() <= limit+1e-12
    assert straight.values.max() > curved.values.max()+.1
    fast_yaw, _ = setup(DifferentialDrive(max_yaw_rate=.3))
    assert fast_yaw.update(reference(1.2), o).values.max() <= .25+1e-12


def test_slow_steering_actuator_limits_changing_curvature_and_reverse_is_signed():
    k = .5+.4*np.sin(np.linspace(0, 6*np.pi, 151))
    slow, o = setup(Ackermann(max_steering_rate=.08))
    quick, _ = setup(Ackermann())
    ref = reference(k)
    a, b = slow.update(ref, o), quick.update(ref, o)
    assert a.values[30:120].mean() < .5*b.values[30:120].mean()
    delta = np.arctan(slow.space.model.wheelbase*k)
    assert np.max(abs(np.gradient(delta, ref.s))*a.values) <= .08+1e-10
    reverse = quick.update(reference(gear=-1), o, terminal=True)
    assert reverse.values.min() < -.4 and reverse.values.max() == 0.
    assert np.all(reverse.values >= -quick.space.model.max_reverse_speed)


def test_sampled_braking_accounts_for_normal_servo_and_weak_physical_brakes():
    ordinary, o = setup()
    weak, _ = setup(dynamics=DynamicsConfig(brake_force=2., wheel_force=2.))
    speed, normal_distance, _, _ = ordinary._response_table(o, 1)
    _, weak_distance, _, _ = weak._response_table(o, 1)
    assert weak_distance[-1] > normal_distance[-1]+.05
    state = ordinary.plant.codec.pack(replace(o.chassis, velocity=Twist2D(speed[-1], 0.)))
    emergency = BatchPredictor(ordinary.plant, Footprint()).predict(state, np.zeros((1, 2)), .1,
                                                                   stop_only=True, mode='estop')
    v = abs(ordinary.plant.codec.velocities(emergency.states)[0, :, 0])
    emergency_distance = np.sum((v[:-1]+v[1:])*.5*np.diff(emergency.times))
    assert normal_distance[-1] > emergency_distance+.15
    normal = ordinary.update(reference(length=1.5), o, terminal=True)
    slower = weak.update(reference(length=1.5), o, terminal=True)
    assert slower.values.mean() < normal.values.mean()


@pytest.mark.parametrize('config', [{'speed_profile_samples': 2}, {'speed_lateral_fraction': 1.01},
                                    {'speed_response_time_s': 0.}])
def test_speed_configuration_rejects_invalid_settings(config):
    with pytest.raises(ValueError):
        EBDWAConfig(**config)
