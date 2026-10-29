"""Adversarial public-port arrival order and provider restart boundaries."""
import asyncio
from dataclasses import replace
from time import monotonic_ns

import pytest

pytest.importorskip('rsim')
from rsim import Component, CommandEnvelope, Runtime
from rsim.devices import ChassisPortsDevice
from pyrobo import pose2d
from pyrobo.rsim import NavigationCommandOutput, NavigationInputs, StackConfig
from pyrobo.rsim.components.memory import MemoryChassis
from pyrobo.rsim.drivers import MemoryChassisFactory


class ManualMemory(MemoryChassis):
    def task(self, *args, **kwargs):
        pass  # Runtime owns lifecycle; this test controls publication order.


class ManualInputs(NavigationInputs):
    def task(self, *args, **kwargs):
        pass


class ManualOutput(NavigationCommandOutput):
    def task(self, *args, **kwargs):
        pass


def test_state_ahead_of_pose_uses_newest_complete_pair_without_renewing_age():
    async def run():
        factory = MemoryChassisFactory()
        source = ManualMemory(factory, factory.port_specs())
        owner, config = Component(), StackConfig(clock=factory.clock)
        inputs = ManualInputs(ChassisPortsDevice.from_component(source),
                              owner.signal('maps', clock=config.clock),
                              owner.signal('missions', clock=config.clock), config)
        async with Runtime(inputs.states):
            initial = (await source.state.get()).data
            metadata = {'episode': 0, 'session': source._instance_id, 'pose_kind': 'ground_truth',
                        'step_index': 1, 'host_monotonic_ns': monotonic_ns()-40_000_000}
            await source.state.publish(dict(initial, step_index=1), stamp_ns=20_000_000,
                                       clock=config.clock, metadata=metadata)
            await source.state.publish(dict(initial, step_index=2), stamp_ns=40_000_000,
                                       clock=config.clock, metadata=dict(metadata, step_index=2))
            await source.pose.publish(pose2d(.1, 0.), stamp_ns=20_000_000,
                                      clock=config.clock, metadata=metadata)
            await inputs.convert()
            converted = await inputs.states.get()
            assert converted.stamp_ns == 20_000_000
            assert converted.metadata['host_monotonic_ns'] == metadata['host_monotonic_ns']
            assert converted.data['state'][0] == pytest.approx(.1)
            await inputs.convert()
            assert len(inputs.states.frames) == 1
            await source.pose.publish(pose2d(.2, 0.), stamp_ns=40_000_000,
                                      clock=config.clock, metadata=dict(metadata, step_index=2))
            await inputs.convert()
            assert (await inputs.states.get()).stamp_ns == 40_000_000
    asyncio.run(run())


def test_command_from_previous_provider_session_is_rejected_at_same_episode():
    async def run():
        factory = MemoryChassisFactory()
        source = ManualMemory(factory, factory.port_specs())
        owner, config = Component(), StackConfig(clock=factory.clock)
        commands = owner.signal('commands', clock=config.clock)
        output = ManualOutput(commands, ChassisPortsDevice.from_component(source), config)
        async with Runtime(output.commands):
            previous = await source.state.get()
            current_metadata = dict(previous.metadata, session='restarted-provider')
            await source.state.publish(previous.data, stamp_ns=previous.stamp_ns,
                                       clock=config.clock, metadata=current_metadata)
            envelope = CommandEnvelope({'native': [.1, .1], 'mode': 'track', 'episode': 0},
                                       'controller', 'epoch', 1, monotonic_ns()+200_000_000)
            await commands.publish(envelope, stamp_ns=0, clock=config.clock, metadata=previous.metadata)
            await output.convert()
            assert not output.commands.frames
            envelope = replace(envelope, sequence=2)
            await commands.publish(envelope, stamp_ns=0, clock=config.clock, metadata=current_metadata)
            await output.convert()
            converted = (await output.commands.get()).data
            assert replace(converted, value=None) == replace(envelope, value=None)
    asyncio.run(run())
