"""Queryable fixed-step transactions owned by the MemoryChassis event loop."""
import asyncio
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
import math
from time import monotonic_ns

from rsim import CommandIdentity, CommandRejected, PacerClosed, PacerReset, StepContext


class MemoryControl:
    def _init_control(self):
        self._physical_substeps = 0
        self._phase, self._abort_reason = 'READY', None
        self._state_known = True
        self._suspend_requested = False
        self._envelope = self._staged_context = self._last_result = None
        self._step_results = OrderedDict()
        self._event_tasks = set()

    @property
    def context(self):
        return StepContext(self._instance_id, self.episode, self.step_index)

    @property
    def stamp_ns(self):
        return round(self._physical_substeps*self.config.dt*1e9) if self._state_known else None

    def _tick_fields(self):
        return dict(stamp_ns=self.stamp_ns, episode=self.episode, step_index=self.step_index,
                    session=self._instance_id)

    def _describe(self):
        return dict(context=asdict(self.context), phase=self._phase, stamp_ns=self.stamp_ns,
                    physical_substeps=self._physical_substeps, state_known=self._state_known,
                    scheduling=self.config.scheduling, source_rate=self.rate, paused=self._paused,
                    abort_reason=self._abort_reason, last_result=deepcopy(self._last_result),
                    command_expirations=self.command_expirations,
                    ack_cache_size=len(self._step_results), ack_capacity=self.config.ack_capacity)

    async def describe(self):
        async with self._management:
            return self._describe()

    synchronize = describe

    @staticmethod
    def _check_deadline(deadline_ns):
        if deadline_ns is not None and monotonic_ns() >= deadline_ns:
            raise CommandRejected('management request expired', reason='expired')

    async def _emit_tick(self, tick):
        if (self._ticks is None or self._paused or self._phase == 'ABORTED'
                or (tick['session'], tick['episode']) != (self._instance_id, self.episode)):
            return
        task = asyncio.create_task(self._ticks.publish(**tick))
        self._event_tasks.add(task)
        try:
            await task
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
        except (PacerClosed, PacerReset):
            pass
        finally:
            self._event_tasks.discard(task)

    async def _cancel_events(self):
        tasks = tuple(self._event_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _cache_step(self, context, signature, result, reason=None):
        self._step_results[context] = (signature, deepcopy(result), reason)
        while len(self._step_results) > self.config.ack_capacity:
            self._step_results.popitem(last=False)
        self._last_result = deepcopy(result)

    def _check_command(self, command_id, next_stamp_ns):
        if self._suspend_requested:
            raise CommandRejected('physical advancement interrupted by a management boundary')
        envelope = self._envelope
        if envelope is None or self.deadline_ns <= monotonic_ns():
            raise CommandRejected('staged command expired before physical substep', reason='expired')
        identity = CommandIdentity(envelope.controller_id, envelope.controller_epoch, envelope.sequence)
        if identity != command_id:
            raise CommandRejected('advance identity does not match the staged command')
        if envelope.source_bound is not None:
            envelope.source_bound.check(self.context, clock=self.config.clock, stamp_ns=next_stamp_ns)

    async def advance(self, context, command_id, requested_substeps, *, _deadline_ns=None):
        try:
            context, command_id = StepContext.from_mapping(context), CommandIdentity.from_mapping(command_id)
        except ValueError as error:
            raise CommandRejected(str(error)) from error
        if self.config.scheduling != 'controlled':
            raise CommandRejected('advance requires scheduling=controlled')
        if type(requested_substeps) is not int or not 1 <= requested_substeps <= self.config.max_substeps:
            raise CommandRejected(f'requested_substeps must be an integer in [1, {self.config.max_substeps}]')
        result, tick = await self._transaction(lambda: self._advance_step(
            context, command_id, requested_substeps, _deadline_ns))
        if tick is not None:
            await self._emit_tick(tick)
        return result

    async def _advance_step(self, context, command_id, requested_substeps, deadline_ns):
        signature = (command_id, requested_substeps)
        cached = self._step_results.get(context)
        if cached is not None:
            previous, result, reason = cached
            if signature != previous:
                raise CommandRejected('step identity was already used with different contents')
            if reason is not None:
                raise CommandRejected(result['error'], reason=reason)
            return deepcopy(result), None
        if context != self.context:
            raise CommandRejected('step context is stale, unknown or no longer cached', reason='retired')
        if self._paused or self._phase == 'ABORTED':
            raise CommandRejected('source is paused or aborted; reset before advancing')
        before, completed, state = self._physical_substeps, 0, None
        error = None
        try:
            if self._phase != 'COMMAND_STAGED' or self._staged_context != context:
                raise CommandRejected('this control step has no staged command')
            self._check_deadline(deadline_ns)
            self._phase = 'ADVANCING'
            for _ in range(requested_substeps):
                self._check_deadline(deadline_ns)
                self._check_command(command_id, round((self._physical_substeps+1)*self.config.dt*1e9))
                def recheck():
                    self._check_deadline(deadline_ns)
                    self._check_command(command_id, round((self._physical_substeps+1)*self.config.dt*1e9))
                self._integrate('track', before_step=recheck)
                if not math.isclose(self.simulation.state.timestamp,
                                    (self._physical_substeps+1)*self.config.dt, abs_tol=1e-9):
                    raise RuntimeError('native dynamics did not advance by the fixed physical dt')
                self._physical_substeps += 1
                completed += 1
                # The transaction itself is shielded. This lets caller cancel
                # and host safety tasks run without interrupting its commit.
                await asyncio.sleep(0)
            self.step_index += 1
            self._phase, self._staged_context = 'READY', None
            await self._publish(emit_tick=False)
            state = deepcopy(self.state.frames[-1].data)
        except BaseException as failure:
            error = failure
            self.step_index = context.step_index
            self._phase, self._abort_reason = 'ABORTED', str(error)
            await self._retire_commands()
            try:
                native_time = self.simulation.state.timestamp
                if not math.isfinite(native_time):
                    raise ValueError('native source time is unknown')
                total = round(native_time/self.config.dt)
                if not math.isclose(native_time, total*self.config.dt, abs_tol=1e-9):
                    raise ValueError('native time is not on a fixed physics boundary')
                self._physical_substeps, completed = total, total-before
            except BaseException:
                self._state_known = False
                self._physical_substeps, completed = None, None
            try:
                if not self._state_known:
                    raise RuntimeError('native state is unknown')
                await self._publish(emit_tick=False)
                state = deepcopy(self.state.frames[-1].data)
            except BaseException:
                for port in self.outputs.values():
                    await port.clear_history()
        result = dict(kind='PhysicsAdvanced' if error is None else 'PhysicsAborted',
                      request_context=asdict(context), context=asdict(self.context),
                      command_id=command_id.to_dict(), requested_substeps=requested_substeps,
                      actual_substeps=completed, physical_substeps=self._physical_substeps,
                      stamp_ns=self.stamp_ns, phase=self._phase, state=state)
        if error is not None:
            reason = getattr(error, 'reason', 'invalid')
            result.update(error=self._abort_reason, state_known=self._state_known)
            self._cache_step(context, signature, result, reason)
            raise CommandRejected(str(error), reason=reason) from error
        self._cache_step(context, signature, result)
        return deepcopy(result), self._tick_fields()

    async def abort(self, reason='caller aborted', *, _deadline_ns=None):
        if not isinstance(reason, str) or not reason:
            raise CommandRejected('abort reason must be a nonempty string')
        self._check_deadline(_deadline_ns)
        self._request_stop()
        await self._cancel_events()
        async def operation():
            self._check_deadline(_deadline_ns)
            self._paused = True
            await self._stop_physics()
            await self._retire_commands()
            self._phase, self._abort_reason = 'ABORTED', reason
            return self._describe()
        return await self._transaction(operation)

    def _request_stop(self):
        # A management request can arrive between physical substeps while the
        # transaction holds the world lock. Its intent must take effect before
        # another substep, rather than waiting for the whole batch to finish.
        self._paused = self._suspend_requested = True
        self._stop_command()
        self.velocity.guard.retire()
        self.velocity._armed = False

    async def _advance_request(self, envelope):
        if envelope.value is None:
            return None
        if not isinstance(envelope.value, dict) or set(envelope.value) != {'context', 'command_id', 'requested_substeps'}:
            raise CommandRejected('advance request requires context, command_id and requested_substeps')
        try:
            return await self.advance(**envelope.value, _deadline_ns=envelope.deadline_ns)
        except CommandRejected as error:
            # The payload refers to a simulation generation, independently of
            # the management port's controller epoch. A stale source request
            # must not retire that transport controller for later valid steps.
            if error.reason == 'retired':
                raise CommandRejected(str(error), reason='stale_context') from error
            raise

    async def _inspect_request(self, envelope):
        if envelope.value is None:
            return None
        if envelope.value != {}:
            raise CommandRejected('inspect request must be an empty mapping')
        return await self.describe()

    async def _reset_request(self, envelope):
        if envelope.value is None:
            return None
        if not isinstance(envelope.value, dict) or set(envelope.value)-{'spawn'}:
            raise CommandRejected('reset request must be a mapping with optional spawn')
        spawn = envelope.value.get('spawn')
        try:
            self._validate_spawn(spawn)
        except (TypeError, ValueError) as error:
            raise CommandRejected(str(error)) from error
        return await self.reset(spawn=spawn, _deadline_ns=envelope.deadline_ns)

    async def _control_request(self, envelope):
        value = envelope.value
        if value is None:
            return None
        if not isinstance(value, dict) or set(value)-{'operation', 'rate', 'reason'}:
            raise CommandRejected('control request must be a mapping with operation')
        operation = value.get('operation')
        if operation == 'set_rate':
            rate = value.get('rate')
            try:
                self._validate_rate(rate)
            except (TypeError, ValueError) as error:
                # Invalid management input is a rejected request, not a failed
                # service task; later valid requests must still be processed.
                raise CommandRejected(str(error)) from error
            return await self.set_rate(rate, _deadline_ns=envelope.deadline_ns)
        if operation == 'abort':
            return await self.abort(value.get('reason', 'caller aborted'), _deadline_ns=envelope.deadline_ns)
        if operation not in ('pause', 'resume'):
            raise CommandRejected('unknown simulation control operation')
        return await getattr(self, operation)(_deadline_ns=envelope.deadline_ns)
