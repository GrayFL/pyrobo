"""TEB crosses the existing ports/process/controlled-provider boundaries."""
import asyncio
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

pytest.importorskip('rsim')
from pyrobo import TEBConfig, TimedElasticBand, EBDWAConfig, ElasticBandDWA, Footprint
from pyrobo.apps.lockstep_navigation import lockstep_config, run_lockstep
from pyrobo.apps.navigation_provider import room_map
from pyrobo.rsim import MemoryChassisFactory, StackConfig


def test_teb_stack_configuration_roundtrip_limits_and_logical_budget():
    config = StackConfig(controller=TEBConfig(), command_speed_limit=.3, command_yaw_limit=.4)
    restored = pickle.loads(pickle.dumps(config))
    assert restored.fingerprint() == config.fingerprint() != StackConfig().fingerprint()
    plant, index, optimizer, guard = restored.control()
    assert isinstance(optimizer, TimedElasticBand) and optimizer.plant is plant
    np.testing.assert_array_equal(optimizer.command_space.bounds()[1], [.3, .4])
    assert guard.index is index
    with pytest.raises(ValueError, match='time_budget=None'):
        replace(lockstep_config(), controller=TEBConfig())
    assert isinstance(lockstep_config(local_optimizer='teb').control()[2], TimedElasticBand)


def test_eb_dwa_stack_configuration_roundtrip_limits_and_logical_budget():
    config = StackConfig(controller=EBDWAConfig(), command_speed_limit=.3, command_yaw_limit=.4)
    restored = pickle.loads(pickle.dumps(config))
    assert restored.fingerprint() == config.fingerprint() != StackConfig().fingerprint()
    plant, index, optimizer, guard = restored.control()
    assert isinstance(optimizer, ElasticBandDWA) and optimizer.plant is plant
    np.testing.assert_array_equal(optimizer.command_space.bounds()[1], [.3, .4])
    assert guard.index is index
    with pytest.raises(ValueError, match='time_budget_s=None'):
        replace(lockstep_config(), controller=EBDWAConfig())
    assert isinstance(lockstep_config(local_optimizer='eb_dwa').control()[2], ElasticBandDWA)


@pytest.mark.parametrize('backend,execution', [('memory', 'local'), ('memory', 'process'), ('mujoco', 'local')])
@pytest.mark.parametrize('optimizer', ['teb', 'eb_dwa'])
def test_elastic_controllers_identical_controlled_graph_reaches_on_real_providers(backend, execution, optimizer, tmp_path):
    if execution == 'process':
        pytest.importorskip('cyclonedds')
    if backend == 'memory':
        source = MemoryChassisFactory(scheduling='controlled', motion_enabled=True,
            bounds=(-3., 3., -3., 3.), footprint=Footprint(.6, .6)).build()
        timestep, substeps = .02, 5
    else:
        pytest.importorskip('mujoco')
        from mujocosim.config import SensorConfig
        from mujocosim.robots.differential_drive import DifferentialDrive, DifferentialDriveConfig
        from mujocosim.rsim import MujocoChassisFactory
        recipe = tmp_path/'room.json'
        recipe.write_text(json.dumps({'format': 'mujocosim-scene-v1', 'room': {'size': [6, 6, 2.6]},
                                      'spawn': [0, 0, 0], 'objects': []}))
        source = MujocoChassisFactory(scene=recipe, cache_root=tmp_path/'cache', motion_enabled=True,
            scheduling='controlled', spawn=(0., 0., 0.),
            robot=DifferentialDrive(DifferentialDriveConfig(wheel_track=.4)),
            sensors=SensorConfig(width=80, height=60)).build()
        timestep, substeps = .002, 50
    report = asyncio.run(run_lockstep(source, physics_timestep=timestep, requested_substeps=substeps,
        map_data=room_map(spacing=.3), config=lockstep_config(local_optimizer=optimizer),
        episodes=2 if backend == 'memory' and execution == 'local' else 1, execution=execution))
    (tmp_path/f'{optimizer}-navigation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    assert report['arrived'] and not report['collided']
    for episode in report['episodes']:
        assert abs(episode['samples'][-1]['speed']) < .02
        for row in episode['trace']:
            assert row['advanced']['command_id'] == row['accepted']['command_id']
            assert row['advanced']['stamp_ns'] == row['accepted']['source_deadline_ns']
            assert row['advanced']['actual_substeps'] == substeps
    if len(report['episodes']) == 2:
        a, b = report['episodes']
        np.testing.assert_array_equal([r['pose'] for r in a['samples']], [r['pose'] for r in b['samples']])
