"""Domain stage protocol faults are tested locally before moving to workers."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from time import monotonic_ns

import pytest

pytest.importorskip('rsim')
from rsim import CommandRejected, ProcessPlacement, Runtime, SampleId, StageAck, StageRequest
from rsim.devices import ChassisPortsDevice
from pyrobo import Footprint
from pyrobo.apps.lockstep_navigation import lockstep_config
from pyrobo.apps.navigation_provider import NavigationMission, room_map
from pyrobo.rsim import LockstepNavigation, MemoryChassisFactory
from pyrobo.rsim.components.navigation_stages import encode_revision, snapshot_revision


def setup(*, stage_timeout=1., stage_retries=1):
    source = MemoryChassisFactory(scheduling='controlled', motion_enabled=True,
        bounds=(-3., 3., -3., 3.), footprint=Footprint(.6, .6)).build()
    config = lockstep_config()
    mission = NavigationMission(room_map(spacing=.3), (.8, 0., 0.), clock=config.clock)
    runner = LockstepNavigation(ChassisPortsDevice.from_component(source), source, mission.maps,
        mission.missions, config, physics_timestep=.02, requested_substeps=5, execution='process',
        stage_timeout=stage_timeout, stage_retries=stage_retries)
    return source, runner


class LostStageReply:
    def __init__(self, sink):
        self.sink, self.dropped = sink, False

    async def set(self, data, *, ttl):
        result = await self.sink.set(data, ttl=ttl)
        if not self.dropped:
            self.dropped = True
            raise TimeoutError('computed ACK lost after publication')
        return result


@pytest.mark.parametrize('placed', [False, True])
def test_local_stage_retry_conflict_and_reset_preserve_one_computation_per_source_step(placed):
    async def run():
        source, runner = setup()
        async with Runtime(*runner.roots, placement=runner.placements() if placed else None):
            runner._stage_sinks['local'] = LostStageReply(runner.stages['local'].stage_request)
            record = await runner.step()
            assert record['stage_calls'][0]['attempts'] == 2
            assert record['stages'][0]['result']['computations'] == 1
            assert source.step_index == 1 and source._physical_substeps == 5
            ack = StageAck.from_mapping(record['stages'][0])
            assert await runner.stages['local'].run_stage(ack.request) == ack.to_dict()
            assert source._physical_substeps == 5
            with pytest.raises(CommandRejected, match='different input'):
                await runner.stages['local'].run_stage(replace(ack.request, input_revision='different'))
            second = await runner.step()
            assert second['stages'][0]['result']['computations'] == 2
            old_command = runner.output.convert_envelope(runner._computed_frame.data)
            await runner.reset()
            with pytest.raises(CommandRejected) as stale:
                await runner.stages['local'].run_stage(ack.request)
            assert stale.value.reason == 'stale_context'
            # Even an adversarial fresh host lease/controller cannot make the
            # old source-generation proof authorize a physical interval.
            with pytest.raises(CommandRejected) as old_motion:
                await runner.device.velocity.set(replace(old_command, controller_epoch='late-old-source',
                                                          sequence=1, deadline_ns=monotonic_ns()+400_000_000))
            assert old_motion.value.reason == 'retired'
            assert source._physical_substeps == 0
            fresh = await runner.step()
            assert fresh['advanced']['context']['episode'] != record['advanced']['context']['episode']
            assert source._physical_substeps == 5
            return runner.records
    evidence = asyncio.run(run())
    path = Path('assets/refactor-v4.1')/os.environ.get('PYROBO_EVIDENCE_RUN', 'phase4')/f'{"process" if placed else "local"}-stage-retry.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')


def test_stage_waits_for_exact_sample_id_not_its_local_sequence_or_arbitrary_latest():
    async def run():
        source, runner = setup()
        async with Runtime(*runner.roots):
            context = source.context
            snapshot = await runner._sample(context)
            request = StageRequest(context, 'local', encode_revision(
                {'snapshot': snapshot_revision(snapshot), 'route': None}))
            signal, frame = runner.inputs_adapter.states, snapshot.state
            await signal.clear_history()
            await signal.publish(frame.data, stamp_ns=frame.stamp_ns, clock=frame.clock,
                                 metadata=frame.metadata, sample_id=SampleId('unrelated', 'states', 1))
            task = asyncio.create_task(runner.stages['local'].run_stage(request))
            await asyncio.sleep(.02)
            assert not task.done()
            replay = await signal.publish(frame.data, stamp_ns=frame.stamp_ns, clock=frame.clock,
                                          metadata=frame.metadata, sample_id=frame.sample_id)
            assert replay.sequence != frame.sequence
            ack = StageAck.from_mapping(await task)
            assert ack.result['proposal'] and ack.result['request']
            assert source._physical_substeps == 0
    asyncio.run(run())


def test_stage_rejects_algorithm_in_another_worker_instead_of_calling_a_remote_python_method():
    async def run():
        source, runner = setup()
        placement = runner.placements()
        placement[runner.stages['local']] = ProcessPlacement('invalid-endpoint-only')
        with pytest.raises(Exception, match='co-located'):
            async with Runtime(*runner.roots, placement=placement):
                raise AssertionError('invalid placement opened')
        assert source.simulation is None
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['timeout', 'cancel'])
def test_local_stage_failure_never_accepts_or_advances_then_explicit_reset_recovers(failure):
    async def run():
        source, runner = setup(stage_timeout=.04 if failure == 'timeout' else 1., stage_retries=0)
        started, finish = asyncio.Event(), asyncio.Event()
        stage = runner.stages['local']
        original = stage.compute
        async def delayed(request):
            started.set()
            await finish.wait()
            return await original(request)
        stage.compute = delayed
        async with Runtime(*runner.roots):
            task = asyncio.create_task(runner.step())
            await started.wait()
            if failure == 'cancel':
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
            with pytest.raises((CommandRejected, TimeoutError, asyncio.CancelledError)):
                await task
            finish.set()
            await asyncio.sleep(.02)
            assert runner.stepper.aborted and runner.barrier.phase == 'ABORTED'
            assert source._physical_substeps == source.step_index == 0
            assert source.command_accepted == 0
            with pytest.raises(RuntimeError, match='aborted'):
                await runner.step()
            await runner.reset()
            record = await runner.step()
            assert record['advanced']['actual_substeps'] == 5 and source._physical_substeps == 5
    asyncio.run(run())
