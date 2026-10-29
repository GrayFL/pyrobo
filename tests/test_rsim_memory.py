"""Real provider lifecycle, host deadline safety and common-port navigation."""
import asyncio
from dataclasses import replace
import json
import subprocess
import sys
from time import monotonic, monotonic_ns
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('rsim')
from rsim import Component, CommandEnvelope, CommandRejected, Runtime, VelocityCommand
from rsim.devices import ChassisPortsDevice
from rsim.runtime.sharing import component_ports
from pyrobo import Ackermann
from pyrobo.rsim import NavigationStack, StackConfig
from pyrobo.rsim.components import NavigationInputs, NavigationCommandOutput
from pyrobo.rsim.drivers import MemoryChassisFactory


class Scenario(Component):
    def __init__(self, clock='simulation', goal=(.8, 0., 0.)):
        super().__init__()
        self.clock, self.goal = clock, goal
        self.maps = self.signal('maps', clock=clock)
        self.missions = self.signal('missions', clock=clock)

    async def open(self):
        await self.maps.publish({'points': np.empty((0, 2)), 'frame': 'world', 'revision': 1,
                                 'static': True, 'bounds': [-3., 3., -3., 3.]},
                                stamp_ns=0, clock=self.clock)
        await self.missions.publish({'epoch': 'mission-1', 'goal': self.goal, 'mode': 'auto'},
                                    stamp_ns=0, clock=self.clock)


def test_memory_factory_is_static_unique_and_fingerprint_sensitive():
    factory = MemoryChassisFactory()
    a, b = factory.build(), factory.build()
    assert a is not b and a.simulation is b.simulation is None
    assert component_ports(a) == factory.port_specs()
    assert factory.device().port_specs == factory.shared().port_specs == component_ports(a)
    specs = factory.port_specs()
    specs.pop('imu')
    assert 'imu' in factory.port_specs()
    for change in ({'motion_enabled': True}, {'dt': .01}, {'max_ttl': .25}, {'clock': 'other'},
                   {'initial_pose': (1., 0., 0.)}, {'boxes': ((1., 2., 1., 2.),)}):
        assert replace(factory, **change).fingerprint() != factory.fingerprint()
    fingerprint = subprocess.check_output([sys.executable, '-c',
        'from pyrobo.rsim.drivers import MemoryChassisFactory; print(MemoryChassisFactory().fingerprint())'], text=True).strip()
    assert fingerprint == factory.fingerprint()


def test_memory_requires_motion_permission_and_expires_before_physical_step():
    async def run():
        disabled = MemoryChassisFactory().build()
        async with Runtime(disabled):
            with pytest.raises(CommandRejected, match='motion is disabled'):
                await disabled.velocity.set(VelocityCommand(.4), ttl=.1)
        source = MemoryChassisFactory(motion_enabled=True).build()
        async with Runtime(source):
            first = await source.pose.get()
            original = first.data.position.copy()
            await source.velocity.set(VelocityCommand(.4), ttl=.08)
            await asyncio.sleep(.25)
            assert source.applied == VelocityCommand() and source.deadline_ns == 0
            np.testing.assert_array_equal(first.data.position, original)
            assert source.step_index > 3
        assert source.simulation is None
        await source.close()
    asyncio.run(run())


def test_memory_reset_retires_commands_clears_history_and_restarts_native_time():
    async def run():
        source = MemoryChassisFactory(motion_enabled=True).build()
        async with Runtime(source):
            old = CommandEnvelope(VelocityCommand(.3), 'test', 'old-episode', 1, monotonic_ns()+250_000_000)
            await source.velocity.set(old)
            await asyncio.sleep(.05)
            previous = (await source.pose.get()).sequence
            await source.reset()
            reset = await source.pose.get()
            assert reset.stamp_ns == 0 and reset.metadata['episode'] == 1
            assert reset.sequence > previous
            assert all(f.metadata['episode'] == 1 for f in source.pose.frames)
            with pytest.raises(CommandRejected, match='retired'):
                await source.velocity.set(replace(old, sequence=2))
            await source.velocity.set(VelocityCommand(), ttl=.05)
    asyncio.run(run())


def test_factory_bound_build_runs_in_real_shared_worker():
    pytest.importorskip('cyclonedds')
    from uuid import uuid4
    async def run():
        factory = MemoryChassisFactory(name='memory-test-'+uuid4().hex, motion_enabled=True)
        provider = factory.shared()
        async with Runtime(provider):
            device = ChassisPortsDevice(key=factory.name, clock=factory.clock)
            async with Runtime(device, device.velocity):
                pose = await device.pose.get(timeout=10)
                state = await device.state.get(timeout=10)
                assert pose.metadata['pose_kind'] == state.data['pose_kind'] == 'ground_truth'
                await device.velocity.set(VelocityCommand(.2), ttl=.1)
                await asyncio.sleep(.25)
                assert (await device.state.get()).data['step_index'] > state.data['step_index']
    asyncio.run(run())


def test_environment_pause_cannot_extend_input_or_command_host_lifetime():
    from pyrobo.rsim.components.memory import MemoryChassis
    class FrozenSource(MemoryChassis):
        frozen = False
        async def _step(self):
            if not self.frozen:
                await super()._step()
    async def run():
        factory = MemoryChassisFactory(motion_enabled=True)
        source = FrozenSource(factory, factory.port_specs())
        device, scenario, config = ChassisPortsDevice.from_component(source), Scenario(), StackConfig(clock='simulation')
        inputs = NavigationInputs(device, scenario.maps, scenario.missions, config)
        navigation = NavigationStack(inputs.states, inputs.maps, inputs.missions, config,
                                     local_options={'offload': True}, global_options={'offload': True})
        output = NavigationCommandOutput(navigation.commands, device, config)
        async with Runtime(output.drive, navigation.execution.diagnostics):
            deadline = monotonic()+3
            while source.applied == VelocityCommand() and monotonic() < deadline:
                await asyncio.sleep(.03)
            assert source.applied.linear_x != 0
            source.frozen = True
            stamp = (await source.pose.get()).stamp_ns
            await asyncio.sleep(.4)
            assert (await source.pose.get()).stamp_ns == stamp
            assert source.applied == VelocityCommand()
            report = (await navigation.execution.diagnostics.get()).data
            assert report['reason'] == 'state_stale' and report['mode'] == 'estop'
    asyncio.run(run())


def test_command_conversion_preserves_deadline_identity_and_rejects_unsupported_capabilities():
    source = MemoryChassisFactory().build()
    commands = Component().signal('commands', clock='simulation')
    output = NavigationCommandOutput(commands, ChassisPortsDevice.from_component(source), StackConfig(clock='simulation'))
    envelope = CommandEnvelope({'native': [.1, .3], 'mode': 'track'}, 'controller', 'epoch', 7, 123)
    converted = output.convert_envelope(envelope)
    assert replace(converted, value=None) == replace(envelope, value=None)
    assert converted.value.linear_x == pytest.approx(.2)
    assert converted.value.angular_z == pytest.approx(.5)
    assert output.convert_envelope(replace(envelope, value={'native': [.1, .3], 'mode': 'estop'})).value == VelocityCommand()
    with pytest.raises(ValueError, match='cannot represent'):
        output.convert_envelope(replace(envelope, value={'native': [.1, .3], 'mode': 'coast'}))
    with pytest.raises(TypeError, match='DifferentialDrive'):
        NavigationCommandOutput(commands, source, StackConfig(model=Ackermann()))


@pytest.mark.parametrize('offload', [False, True])
def test_same_navigation_assembly_reaches_goal_from_public_memory_device(offload):
    async def run():
        source = MemoryChassisFactory(motion_enabled=True, bounds=(-3., 3., -3., 3.)).build()
        device = ChassisPortsDevice.from_component(source)
        scenario = Scenario()
        config = StackConfig(clock='simulation')
        inputs = NavigationInputs(device, scenario.maps, scenario.missions, config)
        navigation = NavigationStack(inputs.states, inputs.maps, inputs.missions, config,
                                     local_options={'offload': offload}, global_options={'offload': offload})
        output = NavigationCommandOutput(navigation.commands, device, config)
        arrived = False
        async with Runtime(output.drive, navigation.diagnostics):
            deadline = monotonic()+12.
            while monotonic() < deadline:
                await asyncio.sleep(.05)
                rows = navigation.diagnostics.frames
                if rows and rows[-1].data['status'] == 'arrived':
                    state = await device.state.get()
                    if abs(state.data['linear_x']) < .02 and abs(state.data['angular_z']) < .02:
                        arrived = True
                        break
            pose = await device.pose.get()
            state = await device.state.get()
            diagnostics = dict(navigation.diagnostics.frames[-1].data) if navigation.diagnostics.frames else {}
        assert arrived, (pose.data.position, state.data, diagnostics)
        assert not state.data['collided']
        assert np.linalg.norm(pose.data.position[:2]-[.8, 0.]) < config.navigation.goal_tolerance
        assert pose.metadata['pose_kind'] == 'ground_truth'
        assert source.simulation is None
    asyncio.run(run())
