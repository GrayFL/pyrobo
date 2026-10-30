"""Real three-worker navigation with one authoritative physical owner."""
import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('rsim')
pytest.importorskip('cyclonedds')
from rsim import CommandRejected, ComponentError, PhysicsAdvanced, Runtime
from pyrobo import Footprint
from pyrobo.apps.lockstep_navigation import run_lockstep
from pyrobo.apps.navigation_provider import room_map
from pyrobo.rsim import MemoryChassisFactory
from test_rsim_navigation_stages import setup


def save(name, data):
    path = Path('assets/refactor-v4.1')/os.environ.get('PYROBO_EVIDENCE_RUN', 'phase4')/(name+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


@pytest.mark.parametrize('backend', ['memory', 'mujoco'])
def test_real_process_lockstep_reaches_stops_and_reset_reproduces_trajectory(backend, tmp_path):
    if backend == 'memory':
        source = MemoryChassisFactory(scheduling='controlled', motion_enabled=True,
            bounds=(-3., 3., -3., 3.), footprint=Footprint(.6, .6)).build()
        timestep, substeps = .02, 5
    else:
        from test_rsim_lockstep_mujoco import make_source
        source = make_source(tmp_path)
        timestep, substeps = .002, 50
    report = asyncio.run(run_lockstep(source, physics_timestep=timestep, requested_substeps=substeps,
                                     map_data=room_map(spacing=.3), episodes=2, execution='process'))
    save(backend+'-process-repeated', report)
    assert report['arrived'] and not report['collided']
    first, second = report['episodes']
    np.testing.assert_allclose([row['pose'] for row in first['samples']],
                               [row['pose'] for row in second['samples']], atol=1e-12, rtol=0.)
    assert [row['command'] for row in first['trace']] == [row['command'] for row in second['trace']]
    workers = set()
    for episode in report['episodes']:
        assert abs(episode['samples'][-1]['speed']) < .02 and episode['minimum_boundary_clearance_m'] > 0
        for index, record in enumerate(episode['trace']):
            stages = record['stages']
            assert [stage['task_id'] for stage in stages] == ['local', 'global', 'execution']
            assert all(stage['context'] == record['context'] for stage in stages)
            assert (stages[0]['result']['request'] is None) == (stages[1]['result']['route'] is None)
            assert stages[0]['result']['request_id'] == stages[1]['result']['request_id']
            assert stages[0]['finished_ns'] <= stages[1]['started_ns'] <= stages[1]['finished_ns'] <= stages[2]['started_ns']
            assert stages[2]['finished_ns'] <= dict((key, start) for key, start, _ in record['timings'])['accept']
            workers.update(stage['result']['pid'] for stage in stages)
            assert record['advanced']['physical_substeps'] == (index+1)*substeps
            assert record['advanced']['stamp_ns'] == record['accepted']['source_deadline_ns'] == (index+1)*100_000_000
            assert record['accepted']['command_id'] == record['advanced']['command_id']
    assert os.getpid() not in workers and len(workers) == 3


@pytest.mark.parametrize('failure', ['timeout', 'cancel', 'worker_exit', 'lost_physics_ack'])
def test_process_stage_fault_never_authorizes_an_extra_physical_step(failure):
    async def run():
        source, runner = setup(stage_timeout=.05 if failure == 'timeout' else .5, stage_retries=0)
        stage = runner.stages['local']
        started = stage.signal('fault_started', clock='simulation')
        original, parent_pid = stage.compute, os.getpid()
        async def injected(request):
            if failure != 'lost_physics_ack' and request.context.episode == 0:
                assert os.getpid() != parent_pid
                await started.publish(True, stamp_ns=0, clock='simulation')
                if failure == 'worker_exit':
                    await asyncio.sleep(.02)
                    os._exit(71)
                await asyncio.sleep(.15)
            return await original(request)
        stage.compute = injected
        async with Runtime(*runner.roots, started, placement=runner.placements()):
            advance = runner.stepper.advance
            if failure == 'lost_physics_ack':
                async def lose(context, accepted):
                    await advance(context, accepted)
                    raise TimeoutError('physical batch committed but ACK was lost')
                runner.stepper.advance = lose
            task = asyncio.create_task(runner.step())
            if failure == 'cancel':
                await started.get(timeout=1.)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
            with pytest.raises((CommandRejected, ComponentError, TimeoutError, asyncio.CancelledError)):
                await task
            status = await runner.synchronize()
            assert runner.stepper.aborted and status['phase'] == 'ABORTED'
            expected = 5 if failure == 'lost_physics_ack' else 0
            assert status['physical_substeps'] == expected
            with pytest.raises(RuntimeError, match='aborted'):
                await runner.step()
            assert (await runner.synchronize())['physical_substeps'] == expected
            if failure == 'lost_physics_ack':
                completed = PhysicsAdvanced.from_mapping(status['last_result'])
                retry = await source.advance(completed.request_context, completed.command_id, 5)
                assert retry == completed.to_dict()
                assert source._physical_substeps == 5
            if failure != 'worker_exit':
                if failure == 'cancel':
                    await asyncio.sleep(.18)  # Let the cancelled caller's bounded worker computation settle.
                runner.stepper.advance = advance
                await runner.reset()
                fresh = await runner.step()
                assert fresh['advanced']['actual_substeps'] == 5 and source._physical_substeps == 5
                assert fresh['accepted']['context']['episode'] > status['context']['episode']
            return {'failure': failure, 'failed_owner': status, 'trace': runner.records,
                    'final_owner': await runner.synchronize()}
    save('process-'+failure, asyncio.run(run()))
