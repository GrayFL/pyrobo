"""Same consumer graph, fixed map and goal across independent providers."""
import asyncio
import json
import os
from pathlib import Path

import pytest

pytest.importorskip('rsim')
from rsim.devices import ChassisPortsDevice
from pyrobo import Footprint
from pyrobo.apps.navigation_provider import navigate, room_map
from pyrobo.rsim.drivers import MemoryChassisFactory


@pytest.fixture(params=['memory', 'mujoco'])
def provider(request, tmp_path):
    # Provider selection is outside navigate(): no backend-specific navigation.
    if request.param == 'memory':
        factory = MemoryChassisFactory(motion_enabled=True, bounds=(-3., 3., -3., 3.),
                                        footprint=Footprint(.6, .6))
    else:
        pytest.importorskip('mujoco')
        from mujocosim.config import SensorConfig
        from mujocosim.robots.differential_drive import DifferentialDrive, DifferentialDriveConfig
        from mujocosim.rsim.drivers import MujocoChassisFactory
        recipe = tmp_path/'room.json'
        recipe.write_text(json.dumps({'format': 'mujocosim-scene-v1', 'room': {'size': [6, 6, 2.6]},
                                       'spawn': [0, 0, 0], 'objects': []}))
        factory = MujocoChassisFactory(scene=recipe, cache_root=tmp_path/'cache', motion_enabled=True,
            spawn=(0., 0., 0.), robot=DifferentialDrive(DifferentialDriveConfig(wheel_track=.4)),
            sensors=SensorConfig(width=80, height=60))
    return request.param, factory.build()


@pytest.mark.parametrize('execution', ['local', 'thread', 'process'])
def test_identical_public_navigation_reaches_and_stops(provider, execution, tmp_path):
    if execution == 'process':
        pytest.importorskip('cyclonedds')
    backend, source = provider
    report = asyncio.run(navigate(ChassisPortsDevice.from_component(source), map_data=room_map(),
                                  execution=execution, duration=20.))
    evidence = os.environ.get('PYROBO_EVIDENCE_RUN')
    root = Path('assets/refactor-v4.1')/evidence if evidence else Path('assets/refactor-v4')
    log = root/f'navigation-{backend}-{execution}.json'
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    assert report['arrived'] and not report['collided'], report
    assert report['localization'] == 'ground_truth' and report['planar_projection']
    assert report['minimum_boundary_clearance_m'] > 0
    assert abs(report['samples'][-1]['speed']) < .02
    ids = {pid for rows in report['pids'].values() for pid in rows}
    assert len(ids) == (3 if execution == 'process' else 1)
