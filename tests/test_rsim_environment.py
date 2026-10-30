"""Real source-time navigation and its pause/overload safety boundaries."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from time import monotonic_ns
from types import SimpleNamespace

import pytest

pytest.importorskip('rsim')
from rsim import CommandEnvelope, CommandRejected, Runtime, VelocityCommand
from rsim.devices import ChassisPortsDevice
from pyrobo import DWAConfig, Footprint
from pyrobo.apps.navigation_provider import navigate, room_map
from pyrobo.rsim import SourceProgress, StackConfig
from pyrobo.rsim.drivers import MemoryChassisFactory


def test_source_progress_uses_consumed_intervals_and_retires_old_generations():
    def frame(step, stamp, episode=0, session='world'):
        return SimpleNamespace(sequence=step+1, stamp_ns=stamp, data={},
            metadata=dict(step_index=step, episode=episode, session=session))
    progress = SourceProgress(.1)
    first = frame(0, 0)
    assert progress.consume(first) == .1 and progress.initial
    assert progress.consume(first) is None
    assert progress.duplicates == 1
    assert progress.consume(frame(5, 100_000_000)) == .1
    assert progress.skipped == 4 and not progress.initial
    assert progress.consume(frame(9, 180_000_000)) == .08
    assert progress.skipped == 7
    assert progress.consume(frame(0, 0, episode=1)) == .1
    assert progress.initial and progress.resets == 1
    with pytest.raises(ValueError, match='retired'):
        progress.consume(frame(10, 200_000_000))
    with pytest.raises(ValueError, match='advance together'):
        progress.consume(frame(1, 0, episode=1))


def test_pause_resume_rate_boundaries_preserve_state_and_retire_commands():
    async def run():
        source = MemoryChassisFactory(scheduling='accelerated', speedup=2., motion_enabled=True).build()
        async with Runtime(source):
            old = CommandEnvelope(VelocityCommand(.3), 'test', 'initial-command', 1,
                                  monotonic_ns()+450_000_000)
            await source.velocity.set(old)
            await asyncio.sleep(.04)
            await source.pause()
            before = await source.state.get()
            position = (await source.pose.get()).data.position.copy()
            assert source.applied == VelocityCommand() and source.deadline_ns == 0
            await asyncio.sleep(.07)
            assert (await source.state.get()).sequence == before.sequence
            with pytest.raises(CommandRejected, match='paused'):
                await source.velocity.set(VelocityCommand(.1), ttl=.1)
            await source.resume()
            first = source.state.frames[0]
            assert first.metadata['episode'] == before.metadata['episode']+1
            assert first.stamp_ns == before.stamp_ns
            assert first.metadata['step_index'] == before.metadata['step_index']
            assert source.pose.frames[0].data.position.tolist() == position.tolist()
            with pytest.raises(CommandRejected, match='retired'):
                await source.velocity.set(replace(old, sequence=2))
            await source.set_rate(10.)
            assert source.state.frames[0].metadata['episode'] == first.metadata['episode']+1
            assert all(f.metadata['source_rate'] == 10. for f in source.state.frames)
            await source.pause()
            paused = (await source.state.get()).sequence
            await source.set_rate(1.)
            await asyncio.sleep(.03)
            assert (await source.state.get()).sequence == paused
            await source.resume()
            assert source.state.frames[0].metadata['source_rate'] == 1.
    asyncio.run(run())


def test_every_source_backpressure_can_be_paused_reset_and_closed():
    async def run():
        source = MemoryChassisFactory(scheduling='accelerated', speedup=10.).build()
        async with Runtime(source):
            subscription = source.subscribe(policy='every', capacity=1)
            await asyncio.sleep(.03)
            blocked_step = source.step_index
            await asyncio.sleep(.03)
            assert source.step_index == blocked_step
            async with asyncio.timeout(.5):
                await source.pause()
                await source.reset()
                await source.resume()
            tick = await subscription.tick()
            assert tick.episode == source.episode
            assert tick.session == source._instance_id
        await subscription.aclose()
    asyncio.run(run())


@pytest.mark.parametrize('rate', [1., 2., 10.])
@pytest.mark.parametrize('control_period', [.2, .25])
def test_environment_navigation_tracks_source_time_at_each_rate(rate, control_period):
    async def run():
        source = MemoryChassisFactory(scheduling='accelerated', speedup=rate, motion_enabled=True,
                                      bounds=(-3., 3., -3., 3.), footprint=Footprint(.6, .6)).build()
        config = StackConfig(clock='simulation', scheduling='environment', control_period=control_period,
                             execution_period=.04, controller=DWAConfig(prediction_time=.6, integration_dt=.1,
                                 linear_samples=3, angular_samples=3),
                             footprint=Footprint(.6, .6), command_speed_limit=.45, command_yaw_limit=.75)
        return await navigate(ChassisPortsDevice.from_component(source), config=config,
                              map_data=room_map(spacing=.3), execution='process', duration=20.)
    report = asyncio.run(run())
    path = Path('assets/refactor-v4.1')/os.environ.get('PYROBO_EVIDENCE_RUN', 'environment-final')/f'memory-rate-{rate:g}-control-{control_period:g}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    assert report['arrived'] and not report['collided'], report
    assert abs(report['samples'][-1]['speed']) < .02
    assert report['rtf'] > 0 and report['rtf'] < rate*1.3
    rows = report['computation_samples']['local']
    assert rows and all(row['control_dt_s'] == row['source_dt_s'] for row in rows)
    for row in rows:
        previous = row['previous_source_stamp_ns']
        if previous is not None:
            assert row['source_dt_s'] == pytest.approx((row['stamp_ns']-previous)/1e9)
    for samples in report['computation_samples'].values():
        assert all(row['stamp_ns'] == row['context']['step_index']*20_000_000 for row in samples)
    assert all(s['source_rate'] == rate for s in report['samples'])
    assert report['navigation_compute_steps'] >= len(rows)


@pytest.mark.parametrize('scheduling', ['realtime', 'environment'])
def test_same_lightweight_navigation_runs_in_both_scheduling_modes(scheduling):
    async def run():
        source = MemoryChassisFactory(motion_enabled=True, bounds=(-3., 3., -3., 3.),
                                      footprint=Footprint(.6, .6)).build()
        config = StackConfig(clock='simulation', scheduling=scheduling, control_period=.1,
                             execution_period=.02, controller=DWAConfig(prediction_time=.6, integration_dt=.1,
                                 linear_samples=3, angular_samples=3),
                             footprint=Footprint(.6, .6), command_speed_limit=.45, command_yaw_limit=.75)
        return await navigate(ChassisPortsDevice.from_component(source), config=config,
                              map_data=room_map(spacing=.3), execution='process', duration=20.)
    report = asyncio.run(run())
    path = Path('assets/refactor-v4.1')/f'paired-lightweight-{scheduling}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    assert report['arrived'] and not report['collided'], report


def test_navigation_resumes_after_pause_without_any_old_command_acceptance():
    async def run():
        source = MemoryChassisFactory(scheduling='accelerated', speedup=2., motion_enabled=True,
                                      bounds=(-3., 3., -3., 3.), footprint=Footprint(.6, .6)).build()
        config = StackConfig(clock='simulation', scheduling='environment', control_period=.2,
                             execution_period=.04, controller=DWAConfig(prediction_time=.6, integration_dt=.1,
                                 linear_samples=3, angular_samples=3),
                             footprint=Footprint(.6, .6), command_speed_limit=.45, command_yaw_limit=.75)
        accepted, actions = [], []
        apply = source.velocity.apply
        async def tracked(envelope):
            await apply(envelope)
            accepted.append({'source_episode': source.episode,
                             'bound_episode': envelope.source_bound.context.episode,
                             'source_step': source.step_index,
                             'deadline_ns': envelope.source_bound.deadline_ns,
                             'controller_epoch': envelope.controller_epoch})
        source.velocity.apply = tracked
        async def manage():
            async with asyncio.timeout(8):
                while source.applied.linear_x <= .05:
                    await asyncio.sleep(.005)
            owner = source.velocity.guard.owner
            await source.pause()
            before = await source.state.get()
            await asyncio.sleep(.12)
            assert (await source.state.get()).sequence == before.sequence
            assert source.applied == VelocityCommand() and source.deadline_ns == 0
            await source.resume()
            first = source.state.frames[0]
            assert first.stamp_ns == before.stamp_ns
            assert first.metadata['step_index'] == before.metadata['step_index']
            assert owner in source.velocity.guard.retired
            actions.append({'paused_stamp_ns': before.stamp_ns, 'resumed_stamp_ns': first.stamp_ns,
                            'old_episode': before.metadata['episode'], 'new_episode': first.metadata['episode']})
        manager = asyncio.create_task(manage())
        try:
            report = await navigate(ChassisPortsDevice.from_component(source), config=config, goal=(1.2, 0., 0.),
                                    map_data=room_map(spacing=.3), execution='process', duration=20.)
            await manager
        finally:
            if not manager.done():
                manager.cancel()
                await asyncio.gather(manager, return_exceptions=True)
        report.update(accepted_commands=accepted, lifecycle_actions=actions)
        return report
    report = asyncio.run(run())
    path = Path('assets/refactor-v4.1/environment-pause-resume.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    assert report['arrived'] and not report['collided'], report
    assert {row['source_episode'] for row in report['accepted_commands']} == {0, 1}
    assert all(row['source_episode'] == row['bound_episode'] for row in report['accepted_commands'])
    assert any(row['source_resets'] == 1 for row in report['computation_samples']['local'])
