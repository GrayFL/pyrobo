"""Real Memory dynamics, real navigation, and complete strict stage transactions."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from time import monotonic_ns
import time

import numpy as np
import pytest

pytest.importorskip('rsim')
from rsim import (CommandAccepted, CommandEnvelope, CommandRejected, PhysicsAdvanced,
                  Runtime, SourceCommandBound, VelocityCommand)
from rsim.devices import ChassisPortsDevice
from pyrobo import Footprint
from pyrobo.core.geometry import planar_state
from pyrobo.simulation import demo_world
from pyrobo.apps.lockstep_navigation import lockstep_config, run_lockstep
from pyrobo.apps.navigation_provider import NavigationMission, room_map
from pyrobo.rsim import LockstepNavigation, MemoryChassisFactory, StackConfig
from pyrobo.rsim.components import GlobalPlanning


def world(**options):
    return MemoryChassisFactory(scheduling='controlled', motion_enabled=True,
        bounds=(-3., 3., -3., 3.), footprint=Footprint(.6, .6), **options).build()


def command(source, *, sequence=1, end_ns=None, ttl=.4):
    bound = None if end_ns is None else SourceCommandBound(source.context, source.config.clock, end_ns)
    return CommandEnvelope(VelocityCommand(.3), 'test', 'epoch', sequence,
                           monotonic_ns()+round(ttl*1e9), bound)


def assembly(source):
    config = lockstep_config()
    mission = NavigationMission(room_map(spacing=.3), (.8, 0., 0.), clock=config.clock)
    return LockstepNavigation(ChassisPortsDevice.from_component(source), source, mission.maps, mission.missions,
                               config, physics_timestep=.02, requested_substeps=5)


def test_real_memory_lockstep_reaches_and_explicit_reset_repeats_exact_trajectory():
    report = asyncio.run(run_lockstep(world(), physics_timestep=.02, requested_substeps=5,
                                     map_data=room_map(spacing=.3), episodes=2))
    path = Path('assets/refactor-v4.1')/os.environ.get('PYROBO_EVIDENCE_RUN', 'phase3')/'memory-repeated.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    assert report['arrived'] and not report['collided']
    a, b = report['episodes']
    np.testing.assert_array_equal([row['pose'] for row in a['samples']], [row['pose'] for row in b['samples']])
    assert [row['command'] for row in a['trace']] == [row['command'] for row in b['trace']]
    assert a['trace'][0]['context']['episode'] != b['trace'][0]['context']['episode']
    for episode in report['episodes']:
        assert episode['completed_control_steps'] > 10
        for index, row in enumerate(episode['trace']):
            assert [stage[0] for stage in row['trace']] == ['state', 'compute', 'accepted', 'advance']
            accepted, advanced = CommandAccepted.from_mapping(row['accepted']), PhysicsAdvanced.from_mapping(row['advanced'])
            assert accepted.context.step_index == index
            assert advanced.context.step_index == index+1
            assert advanced.command_id == accepted.command_id
            assert advanced.stamp_ns == accepted.source_deadline_ns == (index+1)*100_000_000
            assert advanced.physical_substeps == (index+1)*5
            assert row['input_revision'][0:2] == ['navigation-example', 1]


def test_controlled_has_no_autonomous_ticks_and_duplicate_advance_is_cached():
    async def run():
        source = world(ack_capacity=1)
        async with Runtime(source):
            assert source._physics_task is None
            await asyncio.sleep(.03)
            assert source.context.step_index == 0 and source.stamp_ns == 0
            first = command(source)
            accepted = CommandAccepted.from_mapping(await source.velocity.set(first))
            assert accepted.source_deadline_ns is None
            results = await asyncio.gather(*(source.advance(accepted.context, accepted.command_id, 5) for _ in range(2)))
            assert results[0] == results[1]
            assert source.step_index == 1 and source.stamp_ns == 100_000_000
            await asyncio.sleep(.41)
            assert await source.advance(accepted.context, accepted.command_id, 5) == results[0]
            with pytest.raises(CommandRejected, match='different contents'):
                await source.advance(accepted.context, accepted.command_id, 6)
            second = CommandAccepted.from_mapping(await source.velocity.set(command(source, sequence=2)))
            await source.advance(second.context, second.command_id, 1)
            with pytest.raises(CommandRejected, match='no longer cached'):
                await source.advance(accepted.context, accepted.command_id, 5)
            await source.reset()
            assert (await source.describe())['ack_cache_size'] == 0
            with pytest.raises(CommandRejected) as retired:
                await source.advance(second.context, second.command_id, 1)
            assert retired.value.reason == 'retired'
    asyncio.run(run())


def test_partial_source_budget_aborts_and_failed_request_is_not_reintegrated():
    async def run():
        source = world()
        async with Runtime(source):
            accepted = CommandAccepted.from_mapping(await source.velocity.set(command(source, end_ns=40_000_000)))
            for _ in range(2):
                with pytest.raises(CommandRejected, match='expired'):
                    await source.advance(accepted.context, accepted.command_id, 5)
                status = await source.synchronize()
                assert status['phase'] == 'ABORTED' and status['physical_substeps'] == 2
                assert status['context']['step_index'] == 0 and status['stamp_ns'] == 40_000_000
                assert status['last_result']['kind'] == 'PhysicsAborted'
                assert source.applied == VelocityCommand()
            await source.reset()
            assert source.stamp_ns == 0 and (await source.describe())['phase'] == 'READY'
    asyncio.run(run())


def test_cancelled_caller_waits_for_commit_and_retry_does_not_repeat_physics():
    async def run():
        source = world()
        async with Runtime(source):
            accepted = CommandAccepted.from_mapping(await source.velocity.set(command(source)))
            task = asyncio.create_task(source.advance(accepted.context, accepted.command_id, 30))
            while source._physical_substeps == 0:
                await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            result = (await source.describe())['last_result']
            assert result['actual_substeps'] == 30
            assert await source.advance(accepted.context, accepted.command_id, 30) == result
            assert source._physical_substeps == 30
    asyncio.run(run())


def test_lost_physics_ack_aborts_stepper_and_query_recovers_actual_commit():
    async def run():
        source = world()
        runner = assembly(source)
        async with Runtime(runner):
            original = runner.stepper.advance
            async def lose(context, accepted):
                await original(context, accepted)
                raise TimeoutError('simulated lost physics reply')
            runner.stepper.advance = lose
            with pytest.raises(TimeoutError, match='lost physics'):
                await runner.step()
            assert runner.stepper.aborted and runner.stepper.needs_synchronization
            status = await runner.synchronize()
            assert status['phase'] == 'ABORTED' and status['physical_substeps'] == 5
            committed = PhysicsAdvanced.from_mapping(status['last_result'])
            assert committed.context == source.context
            assert await source.advance(committed.request_context, committed.command_id, 5) == status['last_result']
            with pytest.raises(RuntimeError, match='aborted'):
                await runner.step()
            assert source._physical_substeps == 5
            runner.stepper.advance = original
            await runner.reset()
            await runner.step()
            assert source._physical_substeps == 5 and source.context.step_index == 1
    asyncio.run(run())


def test_compute_failure_and_input_revision_change_cannot_advance():
    async def run(change_revision):
        source = world()
        runner = assembly(source)
        async with Runtime(runner):
            original = runner.stepper.compute
            async def fail(context, snapshot):
                if not change_revision:
                    raise RuntimeError('optimizer failed')
                result = await original(context, snapshot)
                data = dict(snapshot.mission.data, epoch='changed-task')
                await runner.navigation.local.missions.publish(data, stamp_ns=0, clock='simulation')
                return result
            runner.stepper.compute = fail
            with pytest.raises((RuntimeError, ValueError)):
                await runner.step()
            assert source.stamp_ns == 0 and source._physical_substeps == 0
            assert runner.stepper.aborted
    asyncio.run(run(False))
    asyncio.run(run(True))


def test_lockstep_config_rejects_wall_budget_or_wrong_fixed_interval():
    with pytest.raises(ValueError, match='logical|time_budget'):
        StackConfig(scheduling='lockstep')
    source = world()
    runner = assembly(source)
    with pytest.raises(ValueError, match='control period'):
        LockstepNavigation(runner.device, source, runner.navigation.local.maps, runner.navigation.local.missions,
                           lockstep_config(), physics_timestep=.02, requested_substeps=4)


@pytest.mark.parametrize('operation', ['pause', 'abort'])
def test_management_stop_interrupts_before_the_next_fixed_substep(operation):
    async def run():
        source = world()
        async with Runtime(source):
            accepted = CommandAccepted.from_mapping(await source.velocity.set(command(source)))
            task = asyncio.create_task(source.advance(accepted.context, accepted.command_id, 80))
            while source._physical_substeps == 0:
                await asyncio.sleep(0)
            await getattr(source, operation)()
            with pytest.raises(CommandRejected, match='interrupted'):
                await task
            status = await source.describe()
            assert status['phase'] == 'ABORTED' and status['paused']
            assert 0 < status['physical_substeps'] < 80
            assert status['context']['step_index'] == 0
            assert source.applied == VelocityCommand()
    asyncio.run(run())


@pytest.mark.parametrize('scheduling', ['realtime', 'accelerated', 'controlled'])
def test_command_conversion_crossing_host_deadline_cannot_integrate_motion(monkeypatch, scheduling):
    async def run():
        source = MemoryChassisFactory(scheduling=scheduling, motion_enabled=True).build()
        async with Runtime(source):
            await source._stop_physics()
            old_step = source._physical_substeps
            original = type(source.config.model).from_twist
            def slow(model, *args, **kwargs):
                time.sleep(.035)
                return original(model, *args, **kwargs)
            monkeypatch.setattr(type(source.config.model), 'from_twist', slow)
            accepted = CommandAccepted.from_mapping(await source.velocity.set(command(source, ttl=.025)))
            if scheduling == 'controlled':
                with pytest.raises(CommandRejected, match='expired'):
                    await source.advance(accepted.context, accepted.command_id, 1)
                assert source._physical_substeps == old_step
            else:
                await source._step()
                assert source._physical_substeps == old_step+1
            assert source.applied == VelocityCommand()
            assert source.simulation.state.velocity.linear == 0
    asyncio.run(run())


@pytest.mark.parametrize('operation', ['resume', 'set_rate'])
def test_boundary_publication_failure_never_leaves_a_ready_source(operation, monkeypatch):
    async def run():
        source = MemoryChassisFactory(scheduling='accelerated').build()
        async with Runtime(source):
            if operation == 'resume':
                await source.pause()
            async def fail(*args, **kwargs):
                raise RuntimeError('injected state publication failure')
            monkeypatch.setattr(source.state, 'publish', fail)
            with pytest.raises(RuntimeError, match='publication failure'):
                if operation == 'resume':
                    await source.resume()
                else:
                    await source.set_rate(2.)
            status = await source.describe()
            assert status['phase'] == 'ABORTED' and status['paused']
            assert source._physics_task is None
            assert all(not port.frames for port in source.outputs.values())
    asyncio.run(run())


def test_command_waiting_for_world_cannot_cross_a_reset_generation():
    async def run():
        source = world()
        async with Runtime(source):
            initial = command(source)
            await source.velocity.set(initial)
            await source._management.acquire()
            reset_started = asyncio.Event()
            original = source._transaction
            async def observed(operation):
                reset_started.set()
                return await original(operation)
            source._transaction = observed
            reset = asyncio.create_task(source.reset())
            await reset_started.wait()
            await asyncio.sleep(0)
            pending = asyncio.create_task(source.velocity.set(replace(initial, sequence=2)))
            try:
                async with asyncio.timeout(.5):
                    while source.velocity.guard.sequences.get(('test', 'epoch')) != 2:
                        await asyncio.sleep(0)
            finally:
                source._management.release()
            await reset
            with pytest.raises(CommandRejected, match='retired'):
                await pending
            assert source.applied == VelocityCommand()
            assert source.velocity.guard.owner is None
            assert (await source.describe())['phase'] == 'READY'
    asyncio.run(run())


def test_accept_uses_the_computed_frame_even_if_latest_is_replaced():
    async def run():
        source = world()
        runner = assembly(source)
        async with Runtime(runner):
            original = runner.stepper.compute
            async def replace_latest(context, snapshot):
                computed = await original(context, snapshot)
                frame = runner._computed_frame
                foreign = replace(computed, sequence=computed.sequence+1000)
                await runner.navigation.commands.publish(foreign, stamp_ns=frame.stamp_ns,
                                                          clock=frame.clock, metadata=frame.metadata)
                return computed
            runner.stepper.compute = replace_latest
            row = await runner.step()
            assert row['accepted']['command_id']['sequence'] == 1
            assert row['advanced']['command_id'] == row['accepted']['command_id']
            assert runner.navigation.commands.frames[-1].data.sequence == 1001
    asyncio.run(run())


def test_pause_releases_event_backpressure_after_a_committed_advance():
    async def run():
        source = world()
        async with Runtime(source):
            subscription = source.subscribe(policy='every', capacity=1)
            first = CommandAccepted.from_mapping(await source.velocity.set(command(source)))
            await source.advance(first.context, first.command_id, 1)
            second = CommandAccepted.from_mapping(await source.velocity.set(command(source, sequence=2)))
            pending = asyncio.create_task(source.advance(second.context, second.command_id, 1))
            while source.context.step_index < 2:
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not pending.done()
            async with asyncio.timeout(.5):
                await source.pause()
                result = await pending
            assert result['kind'] == 'PhysicsAdvanced' and result['physical_substeps'] == 2
            assert (await source.describe())['last_result'] == result
            await subscription.aclose()
    asyncio.run(run())


def test_actual_global_rrt_resets_rng_and_cache_on_source_generation_change():
    from rsim import Component
    from pyrobo.planning import RRTConnectConfig
    from copy import deepcopy
    async def run():
        physical, start, goal = demo_world()
        class Requests(Component):
            def __init__(self):
                super().__init__()
                self.requests = self.signal('requests', clock='simulation')
                self.maps = self.signal('maps', clock='simulation')
            async def open(self):
                bounds = physical.bounds
                await self.maps.publish({'points': physical.point_cloud().points[:, :2], 'frame': 'world',
                    'bounds': [bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax], 'revision': 1, 'static': True},
                    stamp_ns=0, clock='simulation', metadata={'session': 'map', 'episode': 0})
        requests = Requests()
        config = StackConfig(clock='simulation', scheduling='lockstep', adaptive_clearance=False,
                             planner=RRTConnectConfig(time_budget=None, max_extensions=1800, seed=7))
        planning = GlobalPlanning(requests.requests, requests.maps, config)
        async with Runtime(planning.routes):
            async def plan(name, episode, step):
                await requests.requests.publish({'id': name, 'start': planar_state(start), 'goal': planar_state(goal),
                    'map_revision': 1, 'task_epoch': 'task', 'episode': episode, 'session': 'world',
                    'map_session': 'map', 'map_episode': 0}, stamp_ns=step*100_000_000, clock='simulation',
                    metadata={'session': 'world', 'episode': episode, 'step_index': step})
                return await planning.step_once()
            first = await plan('a', 0, 0)
            assert first['status'] == 'success' and first['extensions'] > 0
            rng = deepcopy(planning.planner.rng.bit_generator.state)
            await plan('b', 0, 1)
            assert planning.planner.rng.bit_generator.state != rng
            old_index = planning.index
            repeated = await plan('c', 1, 0)
            np.testing.assert_array_equal(first['path_xy'], repeated['path_xy'])
            assert first['extensions'] == repeated['extensions']
            assert planning.index is not old_index
            assert all(frame.metadata['episode'] == 1 for frame in planning.routes.frames)
    asyncio.run(run())


def test_publication_failure_keeps_actual_native_progress_queryable(monkeypatch):
    async def run():
        source = world()
        async with Runtime(source):
            accepted = CommandAccepted.from_mapping(await source.velocity.set(command(source)))
            async def fail(*args, **kwargs):
                raise RuntimeError('publication failed after native integration')
            monkeypatch.setattr(source.state, 'publish', fail)
            with pytest.raises(CommandRejected, match='publication failed'):
                await source.advance(accepted.context, accepted.command_id, 3)
            status = await source.synchronize()
            assert status['phase'] == 'ABORTED' and status['state_known']
            assert status['physical_substeps'] == 3 and status['stamp_ns'] == 60_000_000
            assert status['last_result']['actual_substeps'] == 3
            assert all(not port.frames for port in source.outputs.values())
    asyncio.run(run())
