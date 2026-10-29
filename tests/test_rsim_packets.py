"""Wire contracts and delayed-reply retirement; process tests are separate."""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rsim')
from pyrobo import ChassisDynamics, ChassisState, DifferentialDrive, PointObstacles, pose2d
from pyrobo.integrations.rsim.components import RemotePlanner
from pyrobo.integrations.rsim.packets import CLOCK, ObservationReader


def frame(data, stamp=10.):
    return SimpleNamespace(data=data, stamp_ns=int(stamp*1e9), clock=CLOCK)


def test_stack_model_defaults_share_tracking_settings_and_validate_downstream_lease():
    from pyrobo import Ackermann
    from pyrobo.integrations.rsim import StackConfig
    config = StackConfig(model=Ackermann())
    assert config.navigation.lookahead == config.controller.lookahead == 1.6
    assert config.controller.prediction_time == 2.4
    with pytest.raises(ValueError, match='actuator lease'):
        StackConfig(execution_period=.3, control_period=.3)


def test_dynamic_map_reuses_one_frame_without_faking_its_acquisition_time():
    plant = ChassisDynamics(DifferentialDrive())
    reader = ObservationReader(plant)
    data = {'points': np.array([[1., 2., 3.]]), 'frame': 'world', 'revision': 1,
            'bounds': [-4., 4., -4., 4.], 'static': False}
    first, _ = reader.map(frame(data))
    assert reader.map(frame(data))[0] is first
    np.testing.assert_array_equal(first.points, data['points'])
    state = frame({'state': plant.codec.pack(ChassisState(pose2d(0, 0))), 'frame': 'world'}, 10.2)
    observed = reader.observation(state, frame(data), .02, now=10.25)
    assert observed.obstacles is first and observed.obstacles.timestamp == 10.
    assert observed.chassis.timestamp == 10.2 and observed.timestamp == 10.25
    renewed, _ = reader.map(frame(data, 10.3))
    assert renewed is not first and renewed.timestamp == 10.3
    wrong = frame(data)
    wrong.clock = 'other-host'
    with pytest.raises(ValueError, match='clock'):
        reader.map(wrong)


def test_reply_from_previous_task_cannot_be_consumed_after_reset():
    from pyrobo import Bounds2D, Footprint, PointCloud
    bridge = RemotePlanner(Footprint())
    world, bounds = PointObstacles(PointCloud(np.empty((0, 2)))), Bounds2D(-3, 3, -3, 3)
    bridge.plan(pose2d(0, 0), pose2d(2, 0), world, bounds)
    old = dict(bridge.pending)
    bridge.reset()
    bridge.plan(pose2d(0, 0), pose2d(0, 2), world, bounds)
    assert old['id'] != bridge.pending['id']
    bridge.accept({**old, 'path_xy': np.array([[0, 0], [2, 0]]), 'status': 'success',
                   'extensions': 1, 'elapsed_s': .1, 'diagnostics': {}})
    assert bridge.reply is None
    bridge.accept({**bridge.pending, 'path_xy': np.array([[0, 0], [0, 2]]), 'status': 'success',
                   'extensions': 1, 'elapsed_s': .1, 'diagnostics': {}})
    result = bridge.plan(pose2d(0, 0), pose2d(0, 2), world, bounds)
    assert result.success and bridge.pending is None
    np.testing.assert_array_equal(result.path.position[-1, :2], [0, 2])


def test_operating_command_limit_preserves_actual_measured_overshoot():
    from dataclasses import replace
    from pyrobo import Twist2D
    from pyrobo.rsim import StackConfig
    config = StackConfig(command_speed_limit=.45, command_yaw_limit=.75)
    assert config.fingerprint() != replace(config, command_speed_limit=.5).fingerprint()
    plant, _, optimizer, _ = config.control()
    state = ChassisState(pose2d(0, 0), Twist2D(.47, .2))
    packed = plant.codec.pack(state)
    assert packed[3] == .47  # Do not clip a physical measurement to a command bound.
    candidates = np.array([[.45, .75], [.46, 0.], [.4, .76]])
    _, valid = optimizer.command_space.to_native_batch(candidates)
    np.testing.assert_array_equal(valid, [True, False, False])
