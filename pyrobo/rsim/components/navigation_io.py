"""Explicit conversion between public differential ports and pyrobo codecs."""
from time import monotonic_ns
from dataclasses import replace
import math

from rsim import Component, CommandEnvelope, Connect, VelocityCommand, SignalPacer, FrameTickPacer, HistoryMiss

from pyrobo import ChassisState, DifferentialDrive, Twist2D, pose2d
from ..packets import episode, latest, source_host_ns, source_session
from ..scheduling import SourceProgress, source_context, source_pacer


class NavigationInputs(Component):
    """Convert an injected localization + measured state; maps stay caller-owned.

    Pose and state must describe the same source sample. No engine/truth access
    exists here; callers explicitly select the localization Device they trust.
    """
    def __init__(self, device, maps, missions, config, *, planar_projection=False, tick_source=None):
        if not isinstance(config.model, DifferentialDrive):
            raise TypeError('public chassis navigation currently requires DifferentialDrive')
        super().__init__(*((tick_source,) if tick_source is not None else ()),
                         inputs=(device.pose, device.state, maps, missions))
        self.tick_source = tick_source
        self.device, self.maps, self.missions, self.config = device, maps, missions, config
        self.planar_projection = planar_projection
        self.states = self.signal('states', history=config.environment_history
                                  if config.environment_policy == 'every' else 8, clock=config.clock)
        self.codec = config.control()[0].codec
        self.previous = 0

    async def open(self):
        self.previous = 0
        self.progress = SourceProgress(self.config.execution_period)
        self.pacer = None
        if self.config.scheduling == 'realtime':
            self.task('convert_state', self.convert, hz=1/self.config.execution_period)
        elif self.config.scheduling == 'environment':
            self.pacer = source_pacer(self, self.device.state, self.config, tick_source=self.tick_source)
            if self.config.environment_policy == 'every' and self.device.pose.history_size < self.config.tick_capacity+2:
                raise ValueError('every localization history must cover the source queue')
            self.task('convert_state', self.convert, pacer=self.pacer)

    async def convert(self):
        pair = self.capture(self.pacer.last_frame) if isinstance(self.pacer, FrameTickPacer) else None
        return await self.step_once(snapshot=pair)

    def capture(self, frame=None):
        poses = {(p.stamp_ns, episode(p), source_session(p)): p for p in self.device.pose.frames}
        if frame is not None:
            pose = poses.get((frame.stamp_ns, episode(frame), source_session(frame)))
            if pose is None:
                raise HistoryMiss('localization history no longer contains the delivered source tick')
            return pose, frame
        # Separate transport ports can remain one sample apart. Consume the
        # newest complete source pair, preserving its original age, rather than
        # starving forever while waiting for the latest state to catch a pose.
        for frame in reversed(self.device.state.frames):
            if frame.sequence <= self.previous:
                return None
            pose_frame = poses.get((frame.stamp_ns, episode(frame), source_session(frame)))
            if pose_frame is not None:
                break
        else:
            return None
        return pose_frame, frame

    async def step_once(self, context=None, snapshot=None):
        pair = self.capture() if snapshot is None else snapshot
        if pair is None:
            return None
        pose_frame, frame = pair
        if context is not None and source_context(frame) != context:
            raise ValueError('chassis snapshot does not match the requested source context')
        if frame.sequence <= self.previous:
            return None
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
        if self.progress.consume(frame) is None:
            return None
        metadata.update(observed_dt_s=self.progress.dt)
        converted = await self.states.publish({'state': self.codec.pack(state), 'frame': data['frame'],
                                   'episode': data['episode'], 'pose_kind': data['pose_kind']},
                                  stamp_ns=frame.stamp_ns, clock=frame.clock,
                                  received_ns=frame.received_ns, metadata=metadata)
        self.previous = frame.sequence
        return converted


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
        self.drive = (None if config.scheduling == 'lockstep' else
                      Connect(self.commands, device.velocity,
                              hz=1/config.execution_period if config.scheduling == 'realtime' else None))
        self.previous = 0

    @property
    def roots(self):
        return (self.drive,) if self.drive is not None else (self.commands, self.device.velocity)

    async def open(self):
        self.previous = 0
        self.pacer = None
        if self.config.scheduling == 'realtime':
            self.task('convert_command', self.convert, hz=1/self.config.execution_period)
        elif self.config.scheduling == 'environment':
            self.pacer = SignalPacer(self.source)
            self.task('convert_command', self.convert, pacer=self.pacer)

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
        return replace(envelope, value=value)

    async def convert(self):
        frame = self.pacer.last_frame if self.pacer is not None else None
        return await self.step_once(snapshot=frame)

    async def step_once(self, context=None, snapshot=None):
        frame = latest(self.source) if snapshot is None else snapshot
        if frame is None or frame.sequence == self.previous:
            return
        self.previous = frame.sequence
        state = latest(self.device.state)
        if (state is None or frame.data.value.get('episode', 0) != episode(state)
                or source_session(frame) != source_session(state)):
            return
        if context is not None and (source_context(frame) != context or source_context(state) != context):
            raise ValueError('command and actuator sample do not match the requested source context')
        command = self.convert_envelope(frame.data)
        if command.deadline_ns <= monotonic_ns():
            return
        await self.commands.publish(command, stamp_ns=frame.stamp_ns, clock=frame.clock,
                                    received_ns=frame.received_ns, metadata=frame.metadata)
        return command
