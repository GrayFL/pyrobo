"""The same public controlled-navigation graph, with a real MuJoCo owner."""
import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('rsim')
pytest.importorskip('mujoco')
from mujocosim.config import SensorConfig
from mujocosim.robots.differential_drive import DifferentialDrive, DifferentialDriveConfig
from mujocosim.rsim import MujocoChassisFactory
from pyrobo.apps.lockstep_navigation import run_lockstep
from pyrobo.apps.lockstep_navigation import lockstep_config
from pyrobo.apps.navigation_provider import room_map, NavigationMission
from pyrobo.rsim import LockstepNavigation
from rsim import Runtime, PhysicsAdvanced
from rsim.devices import ChassisPortsDevice


def make_source(tmp_path):
    recipe = tmp_path/'room.json'
    recipe.write_text(json.dumps({'format': 'mujocosim-scene-v1', 'room': {'size': [6, 6, 2.6]},
                                 'spawn': [0, 0, 0], 'objects': []}))
    return MujocoChassisFactory(scene=recipe, cache_root=tmp_path/'cache', motion_enabled=True,
        scheduling='controlled', spawn=(0., 0., 0.),
        robot=DifferentialDrive(DifferentialDriveConfig(wheel_track=.4)),
        sensors=SensorConfig(width=80, height=60)).build()


def test_real_mujoco_uses_identical_lockstep_navigation_and_reset_replays(tmp_path):
    source = make_source(tmp_path)
    report = asyncio.run(run_lockstep(source, physics_timestep=.002, requested_substeps=50,
                                     map_data=room_map(spacing=.3), episodes=2))
    path = Path('assets/refactor-v4.1')/os.environ.get('PYROBO_EVIDENCE_RUN', 'phase3')/'mujoco-repeated.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    assert report['arrived'] and not report['collided'], report
    a, b = report['episodes']
    np.testing.assert_allclose([row['pose'] for row in a['samples']], [row['pose'] for row in b['samples']], atol=1e-12, rtol=0.)
    assert [row['command'] for row in a['trace']] == [row['command'] for row in b['trace']]
    for episode in report['episodes']:
        assert episode['completed_control_steps'] > 10
        assert abs(episode['samples'][-1]['speed']) < .02
        for index, row in enumerate(episode['trace']):
            assert row['advanced']['actual_substeps'] == 50
            assert row['advanced']['physical_substeps'] == (index+1)*50
            assert row['advanced']['stamp_ns'] == row['accepted']['source_deadline_ns'] == (index+1)*100_000_000
            assert row['accepted']['command_id'] == row['advanced']['command_id']
            assert [stage[0] for stage in row['trace']] == ['state', 'compute', 'accepted', 'advance']


def test_real_mujoco_navigation_lost_ack_requires_query_reset_then_reaches(tmp_path):
    async def run():
        source = make_source(tmp_path)
        config = lockstep_config()
        mission = NavigationMission(room_map(spacing=.3), (.8, 0., 0.), clock=config.clock)
        runner = LockstepNavigation(ChassisPortsDevice.from_component(source), source,
            mission.maps, mission.missions, config, physics_timestep=.002, requested_substeps=50)
        async with Runtime(runner):
            for _ in range(3):
                await runner.step()
            advance = runner.stepper.advance
            async def lose(context, accepted):
                await advance(context, accepted)
                raise TimeoutError('MuJoCo committed but ACK was lost')
            runner.stepper.advance = lose
            with pytest.raises(TimeoutError, match='ACK was lost'):
                await runner.step()
            status = await runner.synchronize()
            assert runner.stepper.aborted and runner.stepper.needs_synchronization
            assert status['phase'] == 'ABORTED' and status['physical_substeps'] == 200
            committed = PhysicsAdvanced.from_mapping(status['last_result'])
            assert committed.context.step_index == 4 and committed.stamp_ns == 400_000_000
            with pytest.raises(RuntimeError, match='aborted'):
                await runner.step()
            assert (await runner.synchronize())['physical_substeps'] == 200
            runner.stepper.advance = advance
            await runner.reset()
            for _ in range(150):
                result = await runner.step()
                if result['status'] == 'arrived':
                    break
            assert result['status'] == 'arrived'
            state = await source.state.get()
            assert abs(state.data['linear_x']) < .02 and not state.data.get('collided', False)
            assert np.linalg.norm((await source.pose.get()).data.position[:2]-[.8, 0.]) < .18
            return {'failed_owner': status, 'recovered_owner': await source.describe(), 'trace': runner.records}
    report = asyncio.run(run())
    path = Path('assets/refactor-v4.1')/os.environ.get('PYROBO_EVIDENCE_RUN', 'phase3')/'mujoco-lost-ack.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
