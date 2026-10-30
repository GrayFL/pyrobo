"""Connection-only views of pyrobo providers."""
from rsim import SharedComponent, StepContext
from ..drivers.memory import MemoryChassisFactory


class MemoryChassisDevice(SharedComponent):
    """Connect to all memory chassis ports without constructing a world."""
    def __init__(self, *, key, clock='simulation', history=32, max_ttl=.5,
                 transport=None, options=None):
        super().__init__(key=key,
            ports=MemoryChassisFactory.port_specs(clock, history=history, max_ttl=max_ttl),
            interface_version=MemoryChassisFactory.INTERFACE_VERSION,
            transport=transport, options=options)

    async def reset(self, spawn=None, *, ttl=None):
        return await self.reset_request.set({'spawn': None if spawn is None else tuple(spawn)}, ttl=ttl)

    async def advance(self, context, command_id, requested_substeps, *, ttl=None):
        if isinstance(context, StepContext):
            context = context.to_dict()
        return await self.advance_request.set(dict(context=context, command_id=command_id,
                                                   requested_substeps=requested_substeps), ttl=ttl)

    async def describe(self, *, ttl=None):
        return await self.inspect_request.set({}, ttl=ttl)

    synchronize = describe

    async def pause(self, *, ttl=None):
        return await self.control_request.set({'operation': 'pause'}, ttl=ttl)

    async def resume(self, *, ttl=None):
        return await self.control_request.set({'operation': 'resume'}, ttl=ttl)

    async def abort(self, reason='caller aborted', *, ttl=None):
        return await self.control_request.set({'operation': 'abort', 'reason': reason}, ttl=ttl)

    async def set_rate(self, rate, *, ttl=None):
        return await self.control_request.set({'operation': 'set_rate', 'rate': rate}, ttl=ttl)


__all__ = ['MemoryChassisDevice']
