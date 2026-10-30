"""One explicit navigation transaction using the public controlled-owner API."""
from time import monotonic_ns
import math

from rsim import (CommandAccepted, CommandIdentity, Component, PhysicsAdvanced, ProcessPlacement,
                  SingleProcessStepper, StageBarrier, StageRequest, StepContext)

from ..components import NavigationInputs, NavigationCommandOutput
from ..scheduling import snapshot_from, source_context
from ..components.navigation_stages import (LocalNavigationStage, GlobalPlanningStage, ExecutionStage,
                                            encode_revision, snapshot_revision, wait_publication)
from .navigation import NavigationStack


class LockstepNavigation(Component):
    """Caller-owned Runtime composition; no engine or transport-specific branch.

    ``management`` supplies describe/advance/abort/reset/resume. Each step is a
    fixed number of physics substeps; the execution proof covers that full
    interval. A failed or lost advance ACK requires synchronize/reset, never an
    assumption that physical progress was zero.
    """
    def __init__(self, device, management, maps, missions, config, *, physics_timestep,
                 requested_substeps, timeout=5., planar_projection=True, execution='local',
                 stage_timeout=1., stage_retries=1, stage_request_ttl=.5):
        if config.scheduling != 'lockstep':
            raise ValueError('LockstepNavigation requires scheduling="lockstep"')
        if (not math.isfinite(physics_timestep) or physics_timestep <= 0
                or type(requested_substeps) is not int or requested_substeps < 1):
            raise ValueError('physics_timestep and requested_substeps must be positive')
        self.interval_ns = round(physics_timestep*1e9)*requested_substeps
        if self.interval_ns <= 0 or self.interval_ns != round(config.control_period*1e9):
            raise ValueError('requested fixed substeps must equal the configured control period')
        if not isinstance(management, Component):
            raise TypeError('management must be an explicitly owned provider or device Component')
        if execution not in ('local', 'process'):
            raise ValueError('lockstep execution must be local or process')
        if execution == 'process' and (not math.isfinite(stage_request_ttl)
                or not 0 < stage_request_ttl <= .5
                or type(stage_retries) is not int or stage_retries < 0):
            raise ValueError('stage request TTL must fit the .5s endpoint limit and retries must be nonnegative')
        self.execution = execution
        self.stage_timeout, self.stage_retries = stage_timeout, stage_retries
        self.stage_request_ttl = stage_request_ttl
        self.device, self.management, self.config = device, management, config
        self.requested_substeps, self.timeout = requested_substeps, timeout
        self.inputs_adapter = NavigationInputs(device, maps, missions, config,
                                               planar_projection=planar_projection)
        self.navigation = NavigationStack(self.inputs_adapter.states, maps, missions, config)
        self.output = NavigationCommandOutput(self.navigation.commands, device, config)
        self.stages = {}
        if execution == 'process':
            self.stages = {
                'local': LocalNavigationStage(self.navigation.local, timeout=stage_timeout),
                'global': GlobalPlanningStage(self.navigation.global_planning, self.inputs_adapter.states,
                                              missions, timeout=stage_timeout),
                'execution': ExecutionStage(self.navigation.execution, actuator_horizon=self.interval_ns/1e9,
                                            timeout=stage_timeout),
            }
        self._stage_sinks = {name: stage.stage_request for name, stage in self.stages.items()}
        super().__init__(management, self.inputs_adapter, self.navigation.local,
                         self.navigation.global_planning, self.navigation.execution, self.output,
                         *self.stages.values(),
                         inputs=(device.pose, device.state, maps, missions))
        self.records = []
        self.stepper = None
        self._snapshot = self._owner_sample = None
        self._computed_frame = None
        self.barrier = None
        self._route_publication = None
        self._navigation_status = None

    @property
    def roots(self):
        """Explicit management port demands for the parent coordinator."""
        return (self, *(stage.stage_request for stage in self.stages.values()))

    def placements(self, prefix='lockstep'):
        """Place each algorithm with its stage endpoint in one dedicated worker."""
        if not self.stages:
            return None
        return {component: ProcessPlacement(f'{prefix}_{name}')
                for name, stage in self.stages.items() for component in (stage, stage.algorithm)}

    async def open(self):
        status = await self.management.describe()
        if status['scheduling'] != 'controlled' or status['paused'] or status['phase'] != 'READY':
            raise ValueError('lockstep requires a ready, unpaused controlled source')
        context = StepContext.from_mapping(status['context'])
        self.records = []
        self._route_publication = self._navigation_status = None
        if self.stages:
            self.barrier = StageBarrier(tuple(self.stages), context=context,
                timeout=self.stage_timeout, retries=self.stage_retries, request_ttl=self.stage_request_ttl)
        self.stepper = SingleProcessStepper(sample=self._sample, compute=self._compute,
            accept=self._accept, advance=self._advance, stop=self.management.abort,
            timeout=self.timeout, context=context, require_physics_ack=True,
            requested_substeps=self.requested_substeps)

    async def _sample(self, context):
        self._owner_sample = await self.management.describe()
        if (StepContext.from_mapping(self._owner_sample['context']) != context
                or self._owner_sample['phase'] != 'READY'):
            raise ValueError('owner is not at the requested ready control boundary')
        frame = await self.inputs_adapter.step_once(context)
        snapshot = snapshot_from(self.inputs_adapter.states, self.navigation.local.maps,
                                 self.navigation.local.missions, state=frame)
        if snapshot is None:
            raise RuntimeError('navigation source/map/mission snapshot is not ready')
        snapshot.require_context(context)
        if snapshot.state.stamp_ns != self._owner_sample['stamp_ns']:
            raise ValueError('published observation differs from committed owner time')
        self._snapshot = snapshot
        return snapshot

    def _check_inputs(self):
        current = snapshot_from(self.inputs_adapter.states, self.navigation.local.maps,
                                self.navigation.local.missions)
        if current is None or current.version != self._snapshot.version:
            raise ValueError('navigation input revision changed during the control step')
        state = self.device.state.frames[-1]
        if (source_context(state) != self._snapshot.context
                or state.stamp_ns != self._snapshot.state.stamp_ns):
            raise ValueError('physical source changed during navigation computation')

    async def _compute(self, context, snapshot):
        if self.stages:
            return await self._compute_stages(context, snapshot)
        command = await self.navigation.step_once(context, snapshot,
                                                  actuator_horizon=self.interval_ns/1e9)
        self._check_inputs()
        if command is None:
            raise RuntimeError('navigation did not produce a command for this source step')
        self._computed_frame = self.navigation.commands.frames[-1]
        if self._computed_frame.data is not command:
            raise ValueError('computed command was replaced before its stage snapshot was captured')
        self._navigation_status = str(self.navigation.local.navigator.status)
        return command

    async def _compute_stages(self, context, snapshot):
        base = snapshot_revision(snapshot)
        self.barrier.begin(context)
        local = await self.barrier.call(self._stage_sinks['local'], StageRequest(context, 'local',
            encode_revision({'snapshot': base, 'route': self._route_publication})))
        planned = await self.barrier.call(self._stage_sinks['global'], StageRequest(context, 'global',
            encode_revision({'snapshot': base, 'request': local.result['request'],
                             'request_id': local.result['request_id']})))
        executed = await self.barrier.call(self._stage_sinks['execution'], StageRequest(context, 'execution',
            encode_revision({'snapshot': base, 'proposal': local.result['proposal']})))
        self._computed_frame = await wait_publication(self.navigation.commands, executed.result['command'])
        self._check_inputs()
        if source_context(self._computed_frame) != context:
            raise ValueError('execution ACK command belongs to another source context')
        if planned.result['route'] is not None:
            self._route_publication = planned.result['route']
        self._navigation_status = local.result['status']
        self.barrier.complete()
        return self._computed_frame.data

    def _stage_evidence(self):
        return {} if self.barrier is None else {
            'stages': [ack.to_dict() for ack in self.barrier.acks],
            'stage_calls': list(self.barrier.timings), 'barrier_phase': self.barrier.phase}

    async def _accept(self, context, command):
        self._check_inputs()
        if self._computed_frame is None or self._computed_frame.data is not command:
            raise ValueError('accept requires the exact command produced by this compute stage')
        converted = await self.output.step_once(context, snapshot=self._computed_frame)
        if converted is None:
            raise RuntimeError('navigation command expired before provider submission')
        accepted = CommandAccepted.from_mapping(await self.device.velocity.set(converted))
        identity = CommandIdentity(command.controller_id, command.controller_epoch, command.sequence)
        if (accepted.command_id != identity or accepted.deadline_ns != command.deadline_ns
                or accepted.source_deadline_ns != command.source_bound.deadline_ns):
            raise ValueError('owner accepted a different command identity or validity interval')
        return accepted

    async def _advance(self, context, accepted):
        self._check_inputs()
        result = PhysicsAdvanced.from_mapping(await self.management.advance(
            context, accepted.command_id.to_dict(), self.requested_substeps))
        if (result.stamp_ns != self._snapshot.state.stamp_ns+self.interval_ns
                or result.physical_substeps != self._owner_sample['physical_substeps']+self.requested_substeps):
            raise ValueError('owner ACK differs from the requested fixed physics interval')
        return result

    async def step(self):
        if self.stepper is None:
            raise RuntimeError('lockstep navigation must be opened by Runtime')
        started = monotonic_ns()
        try:
            snapshot, command = await self.stepper.step()
        except BaseException as error:
            self.records.append({'ok': False, 'host_started_ns': started, 'host_finished_ns': monotonic_ns(),
                'context': self.stepper.context.to_dict(), 'error': repr(error),
                'needs_synchronization': self.stepper.needs_synchronization,
                'trace': [(name, context.to_dict()) for name, context in self.stepper.trace],
                'timings': list(self.stepper.timings), **self._stage_evidence()})
            raise
        record = {'ok': True, 'host_started_ns': started, 'host_finished_ns': monotonic_ns(),
                  'context': snapshot.context.to_dict(), 'input_revision': list(snapshot.input_revision),
                  'source_stamp_ns': snapshot.state.stamp_ns,
                  'command': {'native': command.value['native'].tolist(), 'mode': command.value['mode']},
                  'accepted': self.stepper.last_command_ack.to_dict(),
                  'advanced': self.stepper.last_physics_ack.to_dict(),
                  'status': self._navigation_status,
                  'trace': [(name, context.to_dict()) for name, context in self.stepper.trace],
                  'timings': list(self.stepper.timings), **self._stage_evidence()}
        self.records.append(record)
        return record

    async def reset(self):
        async def reset_source():
            await self.management.reset()
            await self.management.resume()
            context = StepContext.from_mapping((await self.management.describe())['context'])
            for stage in self.stages.values():
                await stage.reset_stage(context, ttl=self.stage_request_ttl)
            if self.barrier is not None:
                self.barrier.reset(context)
            self._route_publication = None
            return context
        await self.stepper.reset(reset_source)
        self._snapshot = self._owner_sample = None
        self._computed_frame = None
        return self.stepper.context

    async def synchronize(self):
        return await self.management.synchronize()

    async def close(self):
        if self.stepper is not None:
            await self.management.abort('navigation closed')
