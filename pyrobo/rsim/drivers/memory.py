"""Immutable recipes for the pure-Python differential chassis provider."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import ClassVar

from pyrobo import DifferentialDrive, Footprint
from pyrobo.models.dynamics.config import DynamicsConfig
from rsim import PortSpec, SharedProvider, VelocityCommand


@dataclass(frozen=True)
class MemoryChassisFactory:
    """Build describes resources; only Runtime.open creates the simulation.

    Coordinates use metres/radians, with a ground-truth pose in ``frame``.
    This uses pyrobo's finite-force plant, not the retired RSim ideal kinematics.
    """
    INTERFACE_VERSION: ClassVar[str] = 'chassis-ports-v1'
    FACTORY_VERSION: ClassVar[str] = 'memory-chassis-v3'
    name: str = 'memory-chassis'
    clock: str = 'simulation'
    frame: str = 'world'
    body: str = 'base_footprint'
    initial_pose: tuple[float, float, float] = (0., 0., 0.)
    bounds: tuple[float, float, float, float] = (-5., 5., -5., 5.)
    boxes: tuple[tuple[float, float, float, float], ...] = ()
    model: DifferentialDrive = field(default_factory=DifferentialDrive)
    footprint: Footprint = field(default_factory=Footprint)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    dt: float = .02
    speedup: float = 1.
    scheduling: str = 'realtime'
    history: int = 32
    max_ttl: float = .5
    motion_enabled: bool = False
    max_substeps: int = 1000
    ack_capacity: int = 32

    def __post_init__(self):
        object.__setattr__(self, 'initial_pose', tuple(self.initial_pose))
        object.__setattr__(self, 'bounds', tuple(self.bounds))
        object.__setattr__(self, 'boxes', tuple(tuple(box) for box in self.boxes))
        if not all(isinstance(value, str) and value for value in (self.name, self.clock, self.frame, self.body)):
            raise ValueError('name, clock and frames must be explicit nonempty strings')
        if not isinstance(self.model, DifferentialDrive):
            raise TypeError('MemoryChassisFactory currently exposes differential velocity capability')
        if (len(self.initial_pose) != 3 or len(self.bounds) != 4
                or any(len(box) != 4 for box in self.boxes)):
            raise ValueError('initial_pose requires x/y/yaw and bounds/boxes xmin/xmax/ymin/ymax')
        if not all(math.isfinite(v) for v in (*self.initial_pose, *self.bounds, *(v for b in self.boxes for v in b))):
            raise ValueError('geometry must be finite')
        for bounds in (self.bounds, *self.boxes):
            if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
                raise ValueError('geometry bounds must be ordered')
        if not all(math.isfinite(v) and v > 0 for v in (self.dt, self.speedup, self.max_ttl)):
            raise ValueError('dt, speedup and max_ttl must be finite and positive')
        if self.scheduling not in ('realtime', 'accelerated', 'controlled'):
            raise ValueError('memory scheduling must be realtime, accelerated or controlled')
        if any(type(value) is not int or value < 1 for value in (self.max_substeps, self.ack_capacity)):
            raise ValueError('max_substeps and ack_capacity must be positive integers')
        if not isinstance(self.motion_enabled, bool):
            raise TypeError('motion_enabled must be a boolean')
        self.port_specs(self.clock, history=self.history, max_ttl=self.max_ttl)

    @classmethod
    def port_specs(cls, clock='simulation', *, history=32, max_ttl=.5):
        """The sole complete declaration for this concrete device."""
        ports = {name: PortSpec(clock=clock, history_capacity=history)
                 for name in ('pose', 'state', 'odom', 'imu')}
        ports['velocity'] = PortSpec(direction='sink', clock='host:monotonic',
                                     max_ttl=max_ttl, fallback=VelocityCommand())
        for name in ('reset_request', 'advance_request', 'inspect_request', 'control_request'):
            ports[name] = PortSpec(direction='sink', clock='host:monotonic', max_ttl=max_ttl, fallback=None)
        return ports

    def fingerprint(self):
        data = {'factory': f'{type(self).__module__}.{type(self).__qualname__}',
                'version': self.FACTORY_VERSION, 'interface': self.INTERFACE_VERSION,
                'config': asdict(self)}
        return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False,
                                        separators=(',', ':')).encode()).hexdigest()

    signature = fingerprint

    def build(self):
        from ..components.memory import MemoryChassis
        return MemoryChassis(self, self.port_specs(self.clock, history=self.history, max_ttl=self.max_ttl))

    __call__ = build

    def shared(self, *, key=None, transport=None, options=None):
        return SharedProvider(factory=self.build, key=key or self.name,
            ports=self.port_specs(self.clock, history=self.history, max_ttl=self.max_ttl),
            interface_version=self.INTERFACE_VERSION, provider_version=self.fingerprint(),
            transport=transport, options=options)

    def device(self, *, key=None, transport=None, options=None):
        from ..devices import MemoryChassisDevice
        return MemoryChassisDevice(key=key or self.name, clock=self.clock, history=self.history,
                                  max_ttl=self.max_ttl, transport=transport, options=options)
