"""Execution prediction must use the downstream lease that was actually sent."""
import asyncio

import numpy as np
import pytest

pytest.importorskip('rsim')
from rsim import Component, Runtime
from pyrobo import ChassisState, pose2d
from pyrobo.rsim import StackConfig
from pyrobo.rsim.components.navigation import Execution
import pyrobo.rsim.components.navigation as navigation_module


def test_command_deadline_matches_execution_history_and_stop_precedes_expiry(monkeypatch):
    now = [10.]
    monkeypatch.setattr(navigation_module, 'monotonic', lambda: now[0])
    config = StackConfig(clock='test:simulation')
    plant = config.control()[0]

    class Inputs(Component):
        def __init__(self):
            super().__init__()
            for name in ('states', 'maps', 'missions', 'proposals'):
                setattr(self, name, self.signal(name, clock=config.clock))

        async def state(self):
            await self.states.publish({'state': plant.codec.pack(ChassisState(pose2d(0, 0))), 'frame': 'world'},
                stamp_ns=round((now[0]-10.)*1e9), clock=config.clock,
                metadata={'session': self._instance_id, 'episode': 0, 'host_monotonic_ns': round(now[0]*1e9)})

        async def open(self):
            await self.state()
            await self.maps.publish({'points': np.empty((0, 2)), 'frame': 'world', 'revision': 1,
                                     'bounds': [-5., 5., -5., 5.], 'static': True},
                                    stamp_ns=0, clock=config.clock)
            await self.missions.publish({'epoch': 'test', 'mode': 'auto'}, stamp_ns=0, clock=config.clock)
            await self.proposals.publish({'native': [.2, .2], 'mode': 'track', 'issued_at': 0., 'valid_until': .3,
                'sequence': 1, 'task_epoch': 'test', 'map_revision': 1, 'session': self._instance_id, 'episode': 0,
                'source_host_ns': 10_000_000_000, 'host_deadline_ns': 10_125_000_000},
                stamp_ns=0, clock=config.clock)

    class ManualExecution(Execution):
        def task(self, *args, **kwargs):
            pass  # Deterministic caller-driven ticks; Runtime still owns lifecycle.

    async def run():
        source = Inputs()
        execution = ManualExecution(source.proposals, source.states, source.maps, source.missions, config)
        async with Runtime(execution.commands):
            await execution.execute()
            command = (await execution.commands.get()).data
            assert command.value['mode'] == 'track'
            assert command.deadline_ns == 10_100_000_000
            assert execution.executor._executed[-1][2] == pytest.approx(command.deadline_ns/1e9-10.)
            now[0] = 10.04
            await source.state()
            await execution.execute()
            stop = (await execution.commands.get()).data
            assert stop.value['mode'] in ('soft_stop', 'estop')
            assert stop.deadline_ns > 10_125_000_000  # A fresh stop can outlive the old movement proposal.
            assert execution.executor._executed[-1][2] == pytest.approx(stop.deadline_ns/1e9-10.)
    asyncio.run(run())


def test_optimizer_result_from_before_source_reset_is_not_published():
    from time import monotonic
    from pyrobo.apps.navigation_provider import NavigationMission, room_map
    from pyrobo.rsim import NavigationInputs, NavigationStack
    from pyrobo.rsim.drivers import MemoryChassisFactory
    from rsim.devices import ChassisPortsDevice

    async def run():
        source = MemoryChassisFactory().build()
        mission = NavigationMission(room_map(), (.8, 0., 0.), clock='simulation')
        config = StackConfig(clock='simulation')
        inputs = NavigationInputs(ChassisPortsDevice.from_component(source), mission.maps, mission.missions, config)
        stack = NavigationStack(inputs.states, inputs.maps, inputs.missions, config,
            local_options={'offload': True, 'delay_every': 1, 'delay_s': .2}, global_options={'offload': True})
        seen = []
        publish = stack.local.proposals.publish

        async def record(data, **kwargs):
            seen.append(data['episode'])
            return await publish(data, **kwargs)
        stack.local.proposals.publish = record
        async with Runtime(stack.local.proposals):
            deadline = monotonic()+2.
            while not stack.local.previous and monotonic() < deadline:
                await asyncio.sleep(.01)
            assert stack.local.previous and not seen  # optimizer is in the injected delay
            await source.reset()
            while not seen and monotonic() < deadline:
                await asyncio.sleep(.02)
            assert seen and set(seen) == {1}
    asyncio.run(run())
