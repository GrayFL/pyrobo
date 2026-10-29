"""RSim dataflow: global search, local optimization, independent execution.

All queues are Signals with bounded history. Workers read the latest sample,
never replay a FIFO backlog of control jobs after a stall.
"""
from dataclasses import replace
import asyncio
import os
from time import monotonic, monotonic_ns, perf_counter, sleep
from uuid import uuid4

import numpy as np
from rsim import Component, CommandEnvelope

from pyrobo import Navigator, Supervisor, pose2d
from pyrobo.core import SafetyAssessment
from pyrobo.core.geometry import path_poses, planar_state
from pyrobo.core.results import PlanResult
from pyrobo.geometry.cache import SnapshotIndex
from pyrobo.runtime.leases import CommandProposal, LeaseExecutor
from ..packets import (CLOCK, ObservationReader, latest, proposal_packet, decode_proposal,
                       episode, source_host_ns, source_session)


class RemotePlanner:
    """Nonblocking GlobalPlanner implementation consumed by Navigator."""
    def __init__(self, footprint):
        self.safety_footprint = footprint
        self.serial = 0
        self.identity = uuid4().hex
        self.map_revision, self.task_epoch, self.episode = None, None, 0
        self.session = ''
        self.reset()

    def reset(self):
        self.pending, self.reply = None, None

    def accept(self, data):
        if self.pending and data['id'] == self.pending['id']:
            self.reply = data

    def plan(self, start, goal, obstacles, bounds):
        if self.reply is not None:
            data, self.reply, self.pending = self.reply, None, None
            return PlanResult(path_poses(data['path_xy'], obstacles.cloud.frame), data['status'],
                              data['extensions'], data['elapsed_s'], data['diagnostics'])
        if self.pending is None:
            self.serial += 1
            self.pending = {'id': f'{self.identity}:{self.serial}', 'start': planar_state(start),
                            'goal': planar_state(goal), 'map_revision': self.map_revision,
                            'task_epoch': self.task_epoch, 'episode': self.episode, 'session': self.session}
        return PlanResult(path_poses([], obstacles.cloud.frame), 'budget_exhausted', 0, 0., {'pending': True})


class GlobalPlanning(Component):
    def __init__(self, requests, maps, config, *, planner_factory=None, delay_s=0., offload=False):
        super().__init__(inputs=(requests, maps))
        self.requests, self.maps, self.config = requests, maps, config
        self.planner_factory, self.delay_s = planner_factory, delay_s
        self.offload = offload
        self.routes = self.signal('routes', history=2, clock=config.clock)

    async def open(self):
        self.planner = self.planner_factory() if self.planner_factory else self.config.global_planner()
        self.reader = ObservationReader(self.config.control()[0], clock=self.config.clock)
        self.index = SnapshotIndex()
        self.previous = 0
        self.task('global_plan', self.plan, hz=100)

    async def plan(self):
        frame = await self.requests.get(after=self.previous)
        self.previous = frame.sequence
        job = frame.data
        map_frame = await self.maps.get()
        cloud, bounds = self.reader.map(map_frame)
        if job['map_revision'] != self.reader.revision:
            return  # The controller publishes a new version after observing it.
        started = perf_counter()
        def compute():
            if self.delay_s:
                sleep(self.delay_s)  # Deliberate worker stall for fault-injection tests.
            return self.planner.plan(pose2d(*job['start'], frame=cloud.frame),
                                      pose2d(*job['goal'], frame=cloud.frame), self.index.get(cloud), bounds)
        result = await asyncio.to_thread(compute) if self.offload else compute()
        packet = {**job, 'path_xy': result.path.position[:, :2], 'status': str(result.status),
                  'extensions': result.extensions, 'elapsed_s': perf_counter()-started,
                  'diagnostics': result.diagnostics, 'pid': os.getpid()}
        current = latest(self.requests)
        if current is not None and current.data['id'] != job['id']:
            return
        await self.routes.publish(packet, stamp_ns=frame.stamp_ns, clock=self.config.clock,
                                  metadata=frame.metadata)


class LocalNavigation(Component):
    def __init__(self, states, maps, missions, config, *, optimizer_factory=None,
                 delay_every=0, delay_s=0., offload=False):
        super().__init__(inputs=(states, maps, missions))
        self.states, self.maps, self.missions, self.config = states, maps, missions, config
        self.optimizer_factory = optimizer_factory
        self.delay_every, self.delay_s = delay_every, delay_s
        self.offload = offload
        self.requests = self.signal('requests', history=2, clock=config.clock)
        self.proposals = self.signal('proposals', history=2, clock=config.clock)
        self.diagnostics = self.signal('diagnostics', history=8, clock=config.clock)
        self.routes = None

    def connect_routes(self, routes):
        self.routes = routes
        self.inputs = (*self.inputs, routes)

    async def open(self):
        cfg = self.config
        plant, index, optimizer, guard = cfg.control()
        if self.optimizer_factory:
            optimizer = self.optimizer_factory()
        self.reader = ObservationReader(plant, clock=self.config.clock)
        self.bridge = RemotePlanner(cfg.global_planner().safety_footprint)
        self.navigator = Navigator(model=cfg.model, footprint=cfg.footprint, dynamics=plant,
                                   local_optimizer=optimizer, global_planner=self.bridge, index=index,
                                   settings=cfg.navigation, safety=guard, retry_interval=cfg.control_period)
        self.supervisor = Supervisor(self.navigator)
        self.previous = self.sequence = self.route_sequence = 0
        self.version, self.sent_request = None, None
        self.task('local_control', self.control, hz=1/cfg.control_period)

    async def control(self):
        state_frame = latest(self.states)
        map_frame, mission_frame = latest(self.maps), latest(self.missions)
        if state_frame is None or map_frame is None or mission_frame is None or state_frame.sequence == self.previous:
            return
        self.previous = state_frame.sequence
        started = perf_counter()
        mission = mission_frame.data
        observation = self.reader.observation(state_frame, map_frame, self.config.control_period,
            now=monotonic() if self.config.clock == CLOCK else None)
        version = (mission['epoch'], self.reader.revision, episode(state_frame), source_session(state_frame))
        if version != self.version:
            self.navigator.reset()
            self.version, self.sent_request = version, None
            if mission.get('goal') is not None:
                self.supervisor.set_goal(pose2d(*mission['goal'], frame=observation.frame), observation.timestamp)
        self.bridge.task_epoch, self.bridge.map_revision, self.bridge.episode, self.bridge.session = version
        mode = mission.get('mode', 'auto')
        if mode not in ('auto', 'paused'):
            raise ValueError('this navigation transport accepts auto/paused; manual and estop require an authority adapter')
        self.supervisor.set_mode(mode)
        if self.routes is not None:
            route = latest(self.routes)
            if route is not None and route.sequence != self.route_sequence:
                self.route_sequence = route.sequence
                self.bridge.accept(route.data)
        self.sequence += 1
        def compute():
            result = self.supervisor.step(observation)
            if self.delay_every and self.sequence % self.delay_every == 0:
                sleep(self.delay_s)
            return result
        motion = await asyncio.to_thread(compute) if self.offload else compute()
        current_state, current_mission, current_map = latest(self.states), latest(self.missions), latest(self.maps)
        if (current_state is None or current_mission is None or current_map is None
                or (current_mission.data['epoch'], current_map.data['revision'], episode(current_state),
                    source_session(current_state)) != version):
            return
        pending = self.bridge.pending
        if pending and pending['id'] != self.sent_request:
            await self.requests.publish(pending, stamp_ns=state_frame.stamp_ns, clock=self.config.clock,
                                        metadata=state_frame.metadata)
            self.sent_request = pending['id']
        issued_at = observation.chassis.timestamp
        proposal = CommandProposal(motion.command, issued_at, issued_at+self.config.command_horizon,
                                   self.sequence, mission['epoch'], self.reader.revision)
        packet = proposal_packet(proposal, self.config.model)
        packet.update(episode=episode(state_frame), session=source_session(state_frame),
                      source_host_ns=source_host_ns(state_frame),
                      host_deadline_ns=source_host_ns(state_frame)+int(self.config.command_horizon*1e9))
        await self.proposals.publish(packet, stamp_ns=state_frame.stamp_ns, clock=self.config.clock,
                                     metadata=state_frame.metadata)
        packet = {'pid': os.getpid(), 'sequence': self.sequence, 'task_epoch': mission['epoch'],
                  'observed_at': issued_at, 'completed_at': monotonic(),
                  'status': str(self.navigator.status), 'planner_status': str(self.navigator._planner_status),
                  'compute_ms': (perf_counter()-started)*1000, 'dwa_ms': motion.diagnostics.elapsed_s*1000,
                  'candidates': motion.diagnostics.candidate_count, 'feasible': motion.diagnostics.feasible_count,
                  'reason': motion.diagnostics.reason, 'safe': motion.safe,
                  'controller_mode': motion.diagnostics.extra.get('controller_mode', 'braking'),
                  'map_revision': self.reader.revision}
        await self.diagnostics.publish(packet, stamp_ns=state_frame.stamp_ns, clock=self.config.clock,
                                       metadata=state_frame.metadata)


class Execution(Component):
    """Independent task/map gate with an unrenewed native-command lease."""
    def __init__(self, proposals, states, maps, missions, config):
        super().__init__(inputs=(proposals, states, maps, missions))
        self.proposals, self.states, self.maps, self.missions = proposals, states, maps, missions
        self.config = config
        self.commands = self.signal('commands', history=2, clock=config.clock)
        self.diagnostics = self.signal('diagnostics', history=8, clock=config.clock)

    async def open(self):
        plant, _, _, guard = self.config.control()
        self.reader = ObservationReader(plant, clock=self.config.clock)
        self.executor = LeaseExecutor(guard, max_result_age=self.config.max_result_age,
                                      max_horizon=self.config.command_horizon)
        self.previous = self.sequence = 0
        self.environment_episode = None
        self.active_host_deadline_ns = 0
        self.epoch = uuid4().hex
        self.task('execute', self.execute, hz=1/self.config.execution_period)

    async def execute(self):
        state_frame, map_frame, mission_frame = latest(self.states), latest(self.maps), latest(self.missions)
        if any(f is None for f in (state_frame, map_frame, mission_frame)):
            return  # Downstream CommandSink expires independently without output.
        started = perf_counter()
        host_now = monotonic()
        now = host_now if self.config.clock == CLOCK else state_frame.stamp_ns/1e9
        current_episode = episode(state_frame)
        environment_version = (source_session(state_frame), current_episode)
        if self.environment_episode is not None and environment_version != self.environment_episode:
            self.executor = LeaseExecutor(self.executor.guard, max_result_age=self.config.max_result_age,
                                          max_horizon=self.config.command_horizon)
            self.active_host_deadline_ns = 0
            self.epoch, self.sequence = uuid4().hex, 0
        self.environment_episode = environment_version
        actuator_horizon = max(2*self.config.execution_period, self.config.control_period)
        observation = self.reader.observation(state_frame, map_frame, actuator_horizon, now=now)
        mission = mission_frame.data
        incoming, proposal = latest(self.proposals), None
        proposal_host_deadline_ns = 0
        if incoming is not None and incoming.sequence != self.previous:
            self.previous = incoming.sequence
            data = incoming.data
            if (data.get('session', ''), data.get('episode', 0)) != environment_version:
                self.executor.rejected += 1
                self.executor.last_rejection = 'proposal_episode_mismatch'
            elif (not 0 <= host_now*1e9-data.get('source_host_ns', int(data['issued_at']*1e9))
                    <= self.config.max_result_age*1e9
                    or data.get('host_deadline_ns', int(data['valid_until']*1e9)) <= host_now*1e9):
                self.executor.rejected += 1
                self.executor.last_rejection = 'proposal_stale'
            else:
                proposal = decode_proposal(data, self.executor.guard.space)
                proposal_host_deadline_ns = data.get('host_deadline_ns', int(data['valid_until']*1e9))
        if host_now*1e9 >= self.active_host_deadline_ns:
            self.executor.active = None
        host_stale = not 0 <= host_now*1e9-source_host_ns(state_frame) <= self.config.safety.max_state_age*1e9
        map_stale = (not map_frame.data.get('static', False)
                     and not 0 <= host_now*1e9-source_host_ns(map_frame) <= self.config.safety.max_cloud_age*1e9)
        accepted_before = self.executor.accepted
        if self.config.clock != CLOCK and (host_stale or map_stale):
            self.executor.active = None
            result = self.executor.guard.stop(observation, reason='state_stale' if host_stale else 'map_stale',
                                              assessment=SafetyAssessment.INPUT_STALE)
        else:
            result = self.executor.step(observation, task_epoch=mission['epoch'], map_revision=self.reader.revision,
                                        proposal=proposal, enabled=mission.get('mode', 'auto') == 'auto')
        if self.executor.accepted > accepted_before:
            self.active_host_deadline_ns = proposal_host_deadline_ns
        self.sequence += 1
        packet = {'native': self.config.model.encode_command(result.command), 'mode': result.command.mode,
                  'issued_at': now, 'task_epoch': mission['epoch'], 'episode': current_episode}
        # This short downstream lease is separate from the optimizer lease. It
        # expires at the actuator if the execution process itself stalls/dies.
        deadline_ns = int((host_now+actuator_horizon)*1e9)
        if result.command.mode == 'track':
            # A forwarded decision cannot extend its source/proposal evidence.
            deadline_ns = min(deadline_ns, self.active_host_deadline_ns)
        envelope = CommandEnvelope(packet, 'pyrobo:execution', self.epoch, self.sequence, deadline_ns)
        await self.commands.publish(envelope, stamp_ns=int(now*1e9), clock=self.config.clock,
                                    metadata=state_frame.metadata)
        await self.diagnostics.publish({'pid': os.getpid(), 'compute_ms': (perf_counter()-started)*1000,
            'timestamp': now,
            'safe': result.safe, 'reason': result.diagnostics.reason, 'mode': result.command.mode,
            'clearance_m': result.diagnostics.clearance_m, 'accepted': self.executor.accepted,
            'rejected': self.executor.rejected, 'expirations': self.executor.expirations,
            'last_rejection': self.executor.last_rejection}, stamp_ns=int(now*1e9), clock=self.config.clock,
                                       metadata=state_frame.metadata)
