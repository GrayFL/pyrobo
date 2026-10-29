"""Explicit conversion between public differential ports and pyrobo codecs."""
from time import monotonic_ns
import math

from rsim import Component, CommandEnvelope, Connect, VelocityCommand

from pyrobo import ChassisState, DifferentialDrive, Twist2D, pose2d
from ..packets import episode, latest, source_host_ns, source_session


class NavigationInputs(Component):
    """Convert an injected localization + measured state; maps stay caller-owned.

    Pose and state must describe the same source sample. No engine/truth access
    exists here; callers explicitly select the localization Device they trust.
    """
    def __init__(self, device, maps, missions, config, *, planar_projection=False):
        if not isinstance(config.model, DifferentialDrive):
            raise TypeError('public chassis navigation currently requires DifferentialDrive')
        super().__init__(inputs=(device.pose, device.state, maps, missions))
        self.device, self.maps, self.missions, self.config = device, maps, missions, config
        self.planar_projection = planar_projection
        self.states = self.signal('states', history=8, clock=config.clock)
        self.codec = config.control()[0].codec
        self.previous = 0

    async def open(self):
        self.previous = 0
        self.task('convert_state', self.convert, hz=1/self.config.execution_period)

    async def convert(self):
        frame = latest(self.device.state)
        if frame is None or frame.sequence == self.previous:
            return
        poses = self.device.pose.frames
        pose_frame = next((p for p in reversed(poses)
                           if p.stamp_ns == frame.stamp_ns and episode(p) == episode(frame)
                           and source_session(p) == source_session(frame)), None)
        if pose_frame is None:
            return  # A separate transport port may arrive one event-loop turn later.
        if frame.clock != self.config.clock or pose_frame.clock != self.config.clock:
            raise ValueError('device clock does not match navigation clock')
        data, pose = frame.data, pose_frame.data
        required = ('linear_x', 'angular_z', 'acceleration', 'yaw_acceleration',
                    'frame', 'episode', 'step_index', 'pose_kind')
        missing = set(required) - data.keys()
        if missing:
            raise ValueError(f'device state missing measured fields: {sorted(missing)}')
        if (pose.wrd_frame != data['frame'] or data['episode'] != episode(pose_frame)
                or pose_frame.metadata.get('step_index') != data['step_index']):
            raise ValueError('localization and measured state frame/episode/step mismatch')
        if self.planar_projection:
            forward = pose.rotation.as_matrix()[:, 0]
            pose = pose2d(float(pose.position[0]), float(pose.position[1]),
                          math.atan2(float(forward[1]), float(forward[0])),
                          frame=pose.wrd_frame, body=pose.ego_frame)
        state = ChassisState(pose, Twist2D(data['linear_x'], data['angular_z']),
                             acceleration=data['acceleration'], yaw_acceleration=data['yaw_acceleration'],
                             timestamp=frame.stamp_ns/1e9)
        metadata = dict(frame.metadata, host_monotonic_ns=source_host_ns(frame),
                        session=source_session(frame), planar_projection=self.planar_projection)
        await self.states.publish({'state': self.codec.pack(state), 'frame': data['frame'],
                                   'episode': data['episode'], 'pose_kind': data['pose_kind']},
                                  stamp_ns=frame.stamp_ns, clock=frame.clock,
                                  received_ns=frame.received_ns, metadata=metadata)
        self.previous = frame.sequence


class NavigationCommandOutput(Component):
    """Preserve command identity and deadlines while converting wheel speeds.

    The public velocity capability has no brake/coast modes. ``coast`` is
    rejected. ``soft_stop``/``estop`` deliberately request zero velocity; this
    is a documented safe-stop request, not a claim of equivalent brake force.
    """
    def __init__(self, commands, device, config):
        if not isinstance(config.model, DifferentialDrive):
            raise TypeError('VelocityCommand conversion requires DifferentialDrive')
        super().__init__(inputs=(commands, device.state))
        self.source, self.device, self.config = commands, device, config
        self.commands = self.signal('commands', history=2, clock=config.clock)
        self.drive = Connect(self.commands, device.velocity, hz=1/config.execution_period)
        self.previous = 0

    @property
    def roots(self):
        return (self.drive,)

    async def open(self):
        self.previous = 0
        self.task('convert_command', self.convert, hz=1/self.config.execution_period)

    def convert_envelope(self, envelope):
        if not isinstance(envelope, CommandEnvelope):
            raise TypeError('navigation command conversion requires a CommandEnvelope')
        data = envelope.value
        mode = data['mode']
        if mode == 'track':
            command = self.config.model.decode_command(data['native'], mode)
            value = VelocityCommand((command.left+command.right)/2,
                                    (command.right-command.left)/self.config.model.track)
        elif mode in ('soft_stop', 'estop'):
            value = VelocityCommand()
        else:
            raise ValueError(f'public velocity capability cannot represent mode {mode!r}')
        return CommandEnvelope(value, envelope.controller_id, envelope.controller_epoch,
                               envelope.sequence, envelope.deadline_ns)

    async def convert(self):
        frame = latest(self.source)
        if frame is None or frame.sequence == self.previous:
            return
        self.previous = frame.sequence
        state = latest(self.device.state)
        if state is None or frame.data.value.get('episode', 0) != episode(state):
            return
        command = self.convert_envelope(frame.data)
        if command.deadline_ns <= monotonic_ns():
            return
        await self.commands.publish(command, stamp_ns=frame.stamp_ns, clock=frame.clock,
                                    received_ns=frame.received_ns, metadata=frame.metadata)
