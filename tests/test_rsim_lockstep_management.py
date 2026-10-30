"""The controlled management API exercised through an actual shared worker."""
import asyncio
from time import monotonic_ns
from uuid import uuid4

import pytest

pytest.importorskip('rsim')
pytest.importorskip('cyclonedds')
from rsim import CommandAccepted, CommandEnvelope, CommandRejected, PhysicsAdvanced, Runtime, VelocityCommand
from pyrobo.rsim import MemoryChassisFactory


def test_shared_management_stale_context_does_not_retire_port_controller():
    async def run():
        factory = MemoryChassisFactory(name='controlled-memory-'+uuid4().hex,
                                       scheduling='controlled', motion_enabled=True)
        provider, device = factory.shared(), factory.device()
        async with Runtime(provider):
            async with Runtime(device, device.velocity, device.reset_request,
                               device.advance_request, device.inspect_request, device.control_request):
                await device.state.get(timeout=10)
                initial = await device.describe()
                assert initial['stamp_ns'] == initial['physical_substeps'] == 0
                first = CommandAccepted.from_mapping(await device.velocity.set(CommandEnvelope(
                    VelocityCommand(.2), 'management-test', 'before-reset', 1, monotonic_ns()+400_000_000)))
                advanced = await device.advance(first.context, first.command_id.to_dict(), 5)
                assert PhysicsAdvanced.from_mapping(advanced).stamp_ns == 100_000_000
                await asyncio.sleep(.41)
                assert await device.advance(first.context, first.command_id.to_dict(), 5) == advanced
                assert (await device.pause())['paused']
                with pytest.raises(CommandRejected, match='requires scheduling'):
                    await device.set_rate(2.)
                with pytest.raises(CommandRejected, match='finite and positive'):
                    await device.set_rate(None)
                resumed = await device.resume()
                assert not resumed['paused'] and resumed['stamp_ns'] == 100_000_000
                assert (await device.abort('management test boundary'))['phase'] == 'ABORTED'
                with pytest.raises(CommandRejected):
                    await device.reset_request.set({'spawn': 3})
                assert (await device.reset((.1, 0., 0.)))['step_index'] == 0
                reset = await device.synchronize()
                assert reset['paused'] and reset['ack_cache_size'] == 0
                assert reset['stamp_ns'] == reset['physical_substeps'] == 0
                await device.resume()
                second = CommandAccepted.from_mapping(await device.velocity.set(CommandEnvelope(
                    VelocityCommand(.2), 'management-test', 'after-reset', 1, monotonic_ns()+400_000_000)))
                with pytest.raises(CommandRejected) as stale:
                    await device.advance(first.context, first.command_id.to_dict(), 5)
                assert stale.value.reason == 'stale_context'
                fresh = PhysicsAdvanced.from_mapping(await device.advance(second.context, second.command_id.to_dict(), 3))
                assert fresh.context.step_index == 1 and fresh.physical_substeps == 3
                assert fresh.stamp_ns == 60_000_000
                assert (await device.describe())['last_result']['kind'] == 'PhysicsAdvanced'
    asyncio.run(run())
