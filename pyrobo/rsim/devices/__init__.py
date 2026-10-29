"""Connection-only views of pyrobo providers."""
from rsim import SharedComponent
from ..drivers.memory import MemoryChassisFactory


class MemoryChassisDevice(SharedComponent):
    """Connect to all memory chassis ports without constructing a world."""
    def __init__(self, *, key, clock='simulation', history=32, max_ttl=.5,
                 transport=None, options=None):
        super().__init__(key=key,
            ports=MemoryChassisFactory.port_specs(clock, history=history, max_ttl=max_ttl),
            interface_version=MemoryChassisFactory.INTERFACE_VERSION,
            transport=transport, options=options)


__all__ = ['MemoryChassisDevice']
