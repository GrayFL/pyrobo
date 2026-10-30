"""Source validity and local every delivery through the actual component graph."""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from time import monotonic, monotonic_ns

import pytest

pytest.importorskip('rsim')
from rsim import CommandEnvelope, CommandRejected, Runtime, SourceCommandBound, StepContext
from rsim import VelocityCommand
from rsim.devices import ChassisPortsDevice
from pyrobo import DWAConfig, Footprint
from pyrobo.apps.navigation_provider import NavigationMission, room_map
from pyrobo.rsim import NavigationCommandOutput, NavigationInputs, NavigationStack, StackConfig
from pyrobo.rsim.drivers import MemoryChassisFactory


def test_fixed_source_bound_never_reanchors_at_acceptance_or_replay():
    async def run():
        source = MemoryChassisFactory(scheduling='accelerated', speedup=10., motion_enabled=True).build()
        async with Runtime(source):
            await source._stop_physics()
            context = StepContext(source._instance_id, source.episode, source.step_index)
            start = round(source.step_index*source.config.dt*1e9)
            bound = SourceCommandBound(context, source.config.clock, start+60_000_000)
            command = CommandEnvelope(VelocityCommand(.3), 'bounded', 'a', 1,
                                      monotonic_ns()+400_000_000, bound)
            # Real physics already advanced after the observation that was
            # checked by the controller. Acceptance cannot reset this budget.
            await source._step()
            await source.velocity.set(command)
            assert source.source_deadline_ns == bound.deadline_ns
            await source._step()
            with pytest.raises(CommandRejected, match='replayed'):
                await source.velocity.set(command)
            assert source.source_deadline_ns == bound.deadline_ns
            await source._step()
            assert source.applied == command.value
            await source._step()
            assert source.applied == VelocityCommand() and source.deadline_ns == 0
            with pytest.raises(CommandRejected, match='source interval expired'):
                await source.velocity.set(replace(command, sequence=2))
            await source.pause()
            await source.resume()
            await source._stop_physics()
            with pytest.raises(CommandRejected, match='generation retired'):
                await source.velocity.set(replace(command, controller_epoch='fresh', sequence=1))
    asyncio.run(run())


def test_host_ttl_expires_without_a_source_tick():
    async def run():
        source = MemoryChassisFactory(scheduling='accelerated', motion_enabled=True).build()
        async with Runtime(source):
            await source._stop_physics()
            before = (await source.state.get()).sequence
            await source.velocity.set(VelocityCommand(.3), ttl=.03)
            assert source.applied.linear_x == .3
            await asyncio.sleep(.07)
            assert source.applied == VelocityCommand()
            assert (await source.state.get()).sequence == before
    asyncio.run(run())


@pytest.mark.parametrize('rate', [1., 2., 10.])
def test_underload_source_rate_has_fixed_physics_dt(rate):
    async def run():
        source = MemoryChassisFactory(scheduling='accelerated', speedup=rate).build()
        async with Runtime(source):
            before, started = (await source.state.get()), monotonic()
            await asyncio.sleep(.4)
            elapsed, after = monotonic()-started, await source.state.get()
            actual = (after.stamp_ns-before.stamp_ns)/1e9/elapsed
            rows = [{'stamp_ns': f.stamp_ns, 'step_index': f.metadata['step_index']}
                    for f in source.state.frames]
            assert all(b['stamp_ns']-a['stamp_ns'] == 20_000_000 for a, b in zip(rows, rows[1:]))
            assert 0 < actual < 1.15*rate
            return {'configured_rate': rate, 'measured_rtf': actual, 'wall_s': elapsed, 'samples': rows}
    result = asyncio.run(run())
    path = Path('assets/refactor-v4.1')/os.environ.get('PYROBO_EVIDENCE_RUN', '')/f'underload-rate-{rate:g}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')


def _graph(*, capacity=1, history=128, explicit_source=True):
    source = MemoryChassisFactory(scheduling='accelerated', speedup=10., motion_enabled=True,
                                  footprint=Footprint(.6, .6)).build()
    config = StackConfig(clock='simulation', scheduling='environment', environment_policy='every',
                         tick_capacity=capacity, environment_history=history, execution_period=.04,
                         footprint=Footprint(.6, .6),
                         controller=DWAConfig(prediction_time=.6, linear_samples=3, angular_samples=3))
    mission = NavigationMission(room_map(), (.8, 0., 0.), clock=config.clock)
    device = ChassisPortsDevice.from_component(source)
    inputs = NavigationInputs(device, mission.maps, mission.missions, config,
                              tick_source=source if explicit_source else None)
    navigation = NavigationStack(inputs.states, inputs.maps, inputs.missions, config,
        tick_source=source if explicit_source else None,
        local_options={'offload': True, 'delay_every': 1, 'delay_s': .035},
        global_options={'offload': True})
    output = NavigationCommandOutput(navigation.commands, device, config)
    return source, inputs, navigation, output


def test_every_matches_frames_and_pause_reset_releases_backpressure():
    async def run():
        source, inputs, navigation, output = _graph()
        async with Runtime(*output.roots, navigation.diagnostics):
            await asyncio.sleep(.45)
            for component in (source, inputs, navigation.local, navigation.execution, output):
                assert component._failure is None
            local = navigation.local
            assert local.progress.received >= 2
            assert local.pacer.subscription.skipped == 0
            assert local.pacer.last_tick.step_index == local.pacer.last_frame.metadata['step_index']
            assert local.progress.dt == pytest.approx(.1)
            assert local.pacer.last_frame.stamp_ns == local.pacer.last_tick.stamp_ns
            async with asyncio.timeout(.5):
                await source.pause()
                paused = (await source.state.get()).sequence
                await asyncio.sleep(.05)
                assert (await source.state.get()).sequence == paused
                await source.reset()
                await source.resume()
            generation = source.episode
            await asyncio.sleep(.2)
            assert local._failure is None
            assert local.progress.context.episode == generation
            assert all(frame.metadata['episode'] == generation for frame in local.proposals.frames)
    asyncio.run(run())


def test_every_rejects_missing_source_placement_and_insufficient_history():
    async def missing():
        _, _, navigation, output = _graph(explicit_source=False)
        with pytest.raises(ValueError, match='local tick_source'):
            async with Runtime(*output.roots):
                pass
        with pytest.raises(ValueError, match='co-located'):
            navigation.placement()
    async def too_short():
        _, _, _, output = _graph(history=8)
        with pytest.raises(ValueError, match='history >='):
            async with Runtime(*output.roots):
                pass
    asyncio.run(missing())
    asyncio.run(too_short())
