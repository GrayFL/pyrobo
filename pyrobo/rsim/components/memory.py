"""One owner advances the existing pure Simulator2D and publishes snapshots."""
import asyncio
import math
from time import monotonic_ns
from uuid import uuid4

import numpy as np
from rsim import Component, CommandSink, CommandRejected, VelocityCommand, TickSource, StepContext

from pyrobo import Bounds2D, Twist2D, pose2d
from pyrobo.simulation.world import Box, Simulator2D, World2D


class MemoryChassis(Component):
    def __init__(self, config, specs):
        super().__init__()
        self.config = config
        self.simulation = None
        self.episode = self.step_index = 0
        self.applied = VelocityCommand()
        self.deadline_ns = 0
        self.source_deadline_ns = 0
        self.source_bound = None
        self.command_accepted = self.command_expirations = 0
        self._physics_task = self._pacer = self._ticks = None
        self._physics_not_before_ns = 0
        self._paused = False
        self.rate = config.speedup
        self._management = None
        self._transactions = set()
        for name, spec in specs.items():
            if spec.direction == 'signal':
                port = self.signal(name, history=spec.history_capacity, clock=spec.clock)
            else:
                port = CommandSink(self, name, self._apply, safe=self._safe,
                                   fallback=spec.fallback, max_ttl=spec.max_ttl,
                                   hz=1000 if config.scheduling == 'accelerated' else 100)
            setattr(self, name, port)

    def configuration(self):
        return {'factory': self.config.fingerprint()}

    @property
    def tick_period_ns(self):
        return round(self.config.dt*1e9)

    async def open(self):
        self._management = asyncio.Lock()
        self._ticks = None
        self._paused = False
        self.rate = self.config.speedup
        self._create_world()
        await self._publish()
        self._start_physics()

    def _start_physics(self, *, after_boundary=False):
        self._physics_not_before_ns = (monotonic_ns()+round(self.config.dt/self.rate*1e9)
                                       if after_boundary else 0)
        if self.config.scheduling == 'realtime':
            self._physics_task = self.task('physics', self._step, hz=self.rate/self.config.dt)
        else:
            from rsim import RatePacer
            self._pacer = RatePacer(1/self.config.dt, rate=self.rate)
            self._physics_task = self.task('physics', self._step, pacer=self._pacer)

    async def _stop_physics(self):
        task, self._physics_task = self._physics_task, None
        if task is not None:
            # Retire ownership before yielding: the Runtime watchdog must not
            # mistake an intentional pause/reset cancellation for a failure.
            self._tasks.discard(task)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._pacer = None

    def subscribe(self, *, hz=None, policy='latest', capacity=1):
        """Bind a worker-local subscription after this source has opened.

        Every-step delivery backpressures the actual publisher; it is not an
        algorithm or physical-advance acknowledgement. Cross-process clients
        use SignalPacer's latest policy instead of serializing this object.
        """
        if self.simulation is None:
            raise RuntimeError('source subscriptions bind during consumer open, after source open')
        if self._ticks is None:
            self._ticks = TickSource(clock=self.config.clock)
        return self._ticks.subscribe(hz=hz, policy=policy, capacity=capacity)

    def _create_world(self):
        cfg = self.config
        world = World2D(Bounds2D(*cfg.bounds), [Box(*box) for box in cfg.boxes], frame=cfg.frame)
        self.simulation = Simulator2D(world,
            pose2d(*cfg.initial_pose, frame=cfg.frame, body=cfg.body),
            footprint=cfg.footprint, model=cfg.model, dynamics=cfg.dynamics)
        self.step_index = 0

    async def _apply(self, envelope):
        value = envelope.value
        if not isinstance(value, VelocityCommand):
            raise CommandRejected('memory chassis requires VelocityCommand')
        if not self.config.motion_enabled and value != VelocityCommand():
            raise CommandRejected('motion is disabled; configure motion_enabled=True')
        if self._paused and value != VelocityCommand():
            raise CommandRejected('memory physics is paused', reason='retired')
        host_now = monotonic_ns()
        if envelope.deadline_ns <= host_now:
            raise CommandRejected('command expired before physical application', reason='expired')
        if envelope.source_bound is not None:
            envelope.source_bound.check(StepContext(self._instance_id, self.episode, self.step_index),
                clock=self.config.clock, stamp_ns=round((self.step_index+1)*self.config.dt*1e9))
        self.applied, self.deadline_ns = value, envelope.deadline_ns
        self.source_bound = envelope.source_bound
        self.command_accepted += 1
        if self.config.scheduling == 'accelerated':
            # Reserve only the source-time budget still covered at acceptance.
            # Checking a complete next step also handles a command arriving just
            # before a host tick, without integrating a whole extra physics dt.
            self.source_deadline_ns = (round(self.step_index*self.config.dt*1e9)
                                      + int((envelope.deadline_ns-host_now)*self.rate))
            if self.source_bound is not None:
                self.source_deadline_ns = min(self.source_deadline_ns, self.source_bound.deadline_ns)

    async def _safe(self, _value):
        self.applied, self.deadline_ns = VelocityCommand(), 0
        self.source_deadline_ns = 0
        self.source_bound = None

    async def _step(self):
        if self._paused or monotonic_ns() < self._physics_not_before_ns:
            return
        if self.simulation.collided:
            await self._safe(None)
            return
        valid = monotonic_ns() < self.deadline_ns
        if self.config.scheduling == 'accelerated':
            valid &= round((self.step_index+1)*self.config.dt*1e9) <= self.source_deadline_ns
        if valid and self.source_bound is not None:
            try:
                self.source_bound.check(StepContext(self._instance_id, self.episode, self.step_index),
                    clock=self.config.clock, stamp_ns=round((self.step_index+1)*self.config.dt*1e9))
            except CommandRejected:
                valid = False
        if not valid:
            self.command_expirations += int(self.deadline_ns != 0)
            await self._safe(None)
        mode = 'track' if valid else 'estop'
        command = self.config.model.from_twist(Twist2D(self.applied.linear_x, self.applied.angular_z), mode=mode)
        self.simulation.step(command, self.config.dt)
        self.step_index += 1
        await self._publish()

    async def _publish(self):
        state = self.simulation.state
        stamp_ns = round(self.step_index*self.config.dt*1e9)
        metadata = {'episode': self.episode, 'step_index': self.step_index,
                    'session': self._instance_id, 'pose_kind': 'ground_truth',
                    'host_monotonic_ns': monotonic_ns(), 'source_rate': self.rate}
        kw = dict(stamp_ns=stamp_ns, clock=self.config.clock, metadata=metadata)
        pose = state.pose.copy()
        packet = {'linear_x': state.velocity.linear, 'angular_z': state.velocity.angular,
                  'acceleration': state.acceleration, 'yaw_acceleration': state.yaw_acceleration,
                  'frame': self.config.frame, 'episode': self.episode, 'step_index': self.step_index,
                  'pose_kind': 'ground_truth', 'collided': self.simulation.collided,
                  'source_rate': self.rate, 'scheduling': self.config.scheduling}
        if self.config.scheduling == 'accelerated':
            packet.update(command_accepted=self.command_accepted, command_expirations=self.command_expirations,
                          applied_command=[self.applied.linear_x, self.applied.angular_z],
                          command_host_deadline_ns=self.deadline_ns,
                          command_source_deadline_ns=self.source_deadline_ns)
        odom = {'header': {'frame_id': pose.wrd_frame}, 'child_frame_id': pose.ego_frame,
                'pose': {'pose': {'position': dict(zip('xyz', map(float, pose.position))),
                                  'orientation': dict(zip('xyzw', map(float, pose.quat)))},
                         'covariance': np.zeros(36)},
                'twist': {'twist': {'linear': {'x': state.velocity.linear, 'y': 0., 'z': 0.},
                                    'angular': {'x': 0., 'y': 0., 'z': state.velocity.angular}}}}
        imu = {'header': {'frame_id': pose.ego_frame},
               'angular_velocity': {'x': 0., 'y': 0., 'z': state.velocity.angular},
               'linear_acceleration': {'x': state.acceleration, 'y': 0., 'z': 0.},
               'orientation': dict(zip('xyzw', map(float, pose.quat))),
               'angular_velocity_covariance': np.zeros(9),
               'linear_acceleration_covariance': np.zeros(9), 'orientation_covariance': np.zeros(9)}
        await self.pose.publish(pose, **kw)
        await self.state.publish(packet, **kw)
        await self.odom.publish(odom, **kw)
        await self.imu.publish(imu, **kw)
        if self._ticks is not None:
            await self._ticks.publish(stamp_ns=stamp_ns, episode=self.episode,
                                      step_index=self.step_index, session=self._instance_id)

    async def _transaction(self, operation):
        """Finish a small owner transaction before propagating cancellation."""
        async def serialized():
            if self._management is None:
                raise RuntimeError('memory source is not running')
            async with self._management:
                if self.simulation is None or self._closing:
                    raise RuntimeError('memory source is closing or closed')
                return await operation()
        task = asyncio.create_task(serialized())
        self._transactions.add(task)
        cancellation = None
        try:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as error:
                    cancellation = error
            result = task.result()
            if cancellation is not None:
                raise cancellation
            return result
        finally:
            self._transactions.discard(task)

    async def _retire_commands(self):
        await self._safe(None)
        self.velocity.guard.retire()
        self.velocity._epoch, self.velocity._sequence = uuid4().hex, 0

    async def _new_episode(self):
        self.episode += 1
        for signal in self.outputs.values():
            await signal.clear_history()
        await self._publish()

    async def pause(self):
        async def operation():
            self._paused = True
            await self._retire_commands()
            await self._stop_physics()
        await self._transaction(operation)

    async def resume(self):
        async def operation():
            if not self._paused:
                return
            await self._retire_commands()
            self._paused = False
            await self._new_episode()
            self._start_physics(after_boundary=True)
        await self._transaction(operation)

    async def set_rate(self, rate):
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('source rate must be finite and positive')
        if self.config.scheduling != 'accelerated':
            raise ValueError('dynamic source rate requires scheduling="accelerated"')
        async def operation():
            if rate == self.rate:
                return
            await self._stop_physics()
            await self._retire_commands()
            self.rate = float(rate)
            if not self._paused:
                await self._new_episode()
                self._start_physics(after_boundary=True)
        await self._transaction(operation)

    async def reset(self):
        """Local management operation, serialized on this component's loop.

        No command from the previous episode is carried into the new world.
        This is intentionally not advertised as a remote request/reply API.
        """
        async def operation():
            await self._stop_physics()
            await self._retire_commands()
            self._create_world()
            await self._new_episode()
            if not self._paused:
                self._start_physics(after_boundary=True)
        await self._transaction(operation)

    async def close(self):
        if self._transactions:
            await asyncio.gather(*tuple(self._transactions), return_exceptions=True)
        await self._stop_physics()
        await self._safe(None)
        if self._ticks is not None:
            await self._ticks.aclose()
            self._ticks = None
        self.simulation = None
