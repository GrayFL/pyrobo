"""One owner advances the existing pure Simulator2D and publishes snapshots."""
from time import monotonic_ns
from uuid import uuid4

import numpy as np
from rsim import Component, CommandSink, CommandRejected, VelocityCommand

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
        for name, spec in specs.items():
            if spec.direction == 'signal':
                port = self.signal(name, history=spec.history_capacity, clock=spec.clock)
            else:
                port = CommandSink(self, name, self._apply, safe=self._safe,
                                   fallback=spec.fallback, max_ttl=spec.max_ttl)
            setattr(self, name, port)

    def configuration(self):
        return {'factory': self.config.fingerprint()}

    async def open(self):
        self._create_world()
        await self._publish()
        self.task('physics', self._step, hz=self.config.speedup/self.config.dt)

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
        if envelope.deadline_ns <= monotonic_ns():
            raise CommandRejected('command expired before physical application', reason='expired')
        self.applied, self.deadline_ns = value, envelope.deadline_ns

    async def _safe(self, _value):
        self.applied, self.deadline_ns = VelocityCommand(), 0

    async def _step(self):
        if self.simulation.collided:
            await self._safe(None)
            return
        valid = monotonic_ns() < self.deadline_ns
        if not valid:
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
                    'host_monotonic_ns': monotonic_ns()}
        kw = dict(stamp_ns=stamp_ns, clock=self.config.clock, metadata=metadata)
        pose = state.pose.copy()
        packet = {'linear_x': state.velocity.linear, 'angular_z': state.velocity.angular,
                  'acceleration': state.acceleration, 'yaw_acceleration': state.yaw_acceleration,
                  'frame': self.config.frame, 'episode': self.episode, 'step_index': self.step_index,
                  'pose_kind': 'ground_truth', 'collided': self.simulation.collided}
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

    async def reset(self):
        """Local management operation, serialized on this component's loop.

        No command from the previous episode is carried into the new world.
        This is intentionally not advertised as a remote request/reply API.
        """
        await self._safe(None)
        self.velocity.guard.retire()
        self.velocity._epoch, self.velocity._sequence = uuid4().hex, 0
        self.episode += 1
        for signal in self.outputs.values():
            await signal.clear_history()
        self._create_world()
        await self._publish()

    async def close(self):
        await self._safe(None)
        self.simulation = None
