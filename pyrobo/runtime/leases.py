"""Deadline-aware execution of asynchronous native-command proposals.

No transport or clock is owned here. The caller supplies fresh observations and
task/map versions; a proposal's original deadline is never renewed on reuse.
"""
from dataclasses import dataclass, replace
from collections import deque
import math

from ..core import Diagnostics, MotionResult, MotionStatus, SafetyAction, SafetyAssessment
from ..safety.fallback import verified_stop


@dataclass(frozen=True)
class CommandProposal:
    command: object
    issued_at: float
    valid_until: float
    sequence: int
    task_epoch: str
    map_revision: int

    def __post_init__(self):
        if (not math.isfinite(self.issued_at) or not math.isfinite(self.valid_until)
                or self.valid_until <= self.issued_at):
            raise ValueError('proposal requires a finite, positive validity interval')
        if not isinstance(self.sequence, int) or self.sequence < 1 or not self.task_epoch:
            raise ValueError('proposal requires a positive sequence and task epoch')


class LeaseExecutor:
    """Check a full remaining hold interval + braking at the execution state.

    Delayed commands may still be useful, but their old safety evidence is not:
    every executed command is independently checked against the current state.
    """
    def __init__(self, guard, *, max_result_age=.25, max_horizon=.3):
        if not 0 < max_result_age <= max_horizon <= guard.config.max_control_dt:
            raise ValueError('result age and horizon must fit the safety control interval')
        if max_horizon > guard.plant.config.timeout:
            raise ValueError('proposal horizon exceeds chassis watchdog')
        self.guard, self.max_result_age, self.max_horizon = guard, max_result_age, max_horizon
        self.active = None
        self._version, self._sequence = None, 0
        self.accepted = self.rejected = self.expirations = 0
        self.last_rejection = ''
        self._executed = deque()

    def step(self, observation, *, task_epoch, map_revision, proposal=None, enabled=True):
        """`observation.dt` must cover the downstream actuator's entire lease.

        Extrapolate a delayed measurement through commands actually sent by
        this executor. Expired downstream commands become finite-force estop.
        This assumes ordered, promptly applied actuator commands; hardware
        adapters must provide their own delivery/actuation error bounds.
        """
        now = observation.timestamp
        reason = self.guard.input_reason(observation)
        current = None if reason else self._current_state(observation)
        if reason or current is None:
            self.active = None
            result = self.guard.stop(observation, reason=reason or 'execution_state_untracked',
                                     assessment=SafetyAssessment.INPUT_STALE)
        else:
            result = self._decide(current, task_epoch=task_epoch, map_revision=map_revision,
                                  proposal=proposal, enabled=enabled)
        if self._executed and now <= self._executed[-1][0]:
            # Duplicate observations are not new actuator evidence.
            if now < self._executed[-1][0]:
                raise ValueError('execution timestamps must be monotonic')
            self._executed.pop()
        self._executed.append((now, result.command, now+observation.dt))
        cutoff = now-self.guard.config.max_state_age
        while len(self._executed) > 1 and self._executed[1][0] <= cutoff:
            self._executed.popleft()
        return result

    def _current_state(self, observation):
        state, now = observation.chassis, observation.timestamp
        if now-state.timestamp <= 1e-9:
            return observation
        commands = list(self._executed)
        if not commands or commands[0][0] > state.timestamp+1e-9:
            return None
        stop = self.guard.space.decode_one([0.]*self.guard.space.native_size, 'estop')
        def advance(source, command, duration):
            plant = self.guard.plant
            native = self.guard.space.model.encode_command(command)[None]
            values = plant.integrate(plant.codec.pack(source)[None], native, duration, mode=command.mode)
            return plant.codec.unpack(values[0], source, duration)
        for i, (start, command, deadline) in enumerate(commands):
            end = min(now, commands[i+1][0] if i+1 < len(commands) else now)
            if end <= state.timestamp:
                continue
            held_until = min(end, deadline)
            if held_until > state.timestamp:
                state = advance(state, command, held_until-state.timestamp)
            if end > state.timestamp:
                state = advance(state, stop, end-state.timestamp)
        return replace(observation, chassis=state)

    def _decide(self, observation, *, task_epoch, map_revision, proposal=None, enabled=True):
        now = observation.timestamp
        version = (task_epoch, map_revision)
        if version != self._version:
            self.active, self._sequence, self._version = None, 0, version
        if not enabled:
            self.active = None
        elif proposal is not None:
            reason = ''
            if (proposal.task_epoch, proposal.map_revision) != version:
                reason = 'proposal_version_mismatch'
            elif proposal.sequence <= self._sequence:
                reason = 'proposal_replayed'
            elif not -1e-9 <= now-proposal.issued_at <= self.max_result_age:
                reason = 'proposal_stale'
            elif proposal.valid_until <= now or proposal.valid_until-proposal.issued_at > self.max_horizon+1e-9:
                reason = 'proposal_expired'
            if reason:
                self.rejected += 1
                self.last_rejection = reason
            else:
                self.active, self._sequence = proposal, proposal.sequence
                self.accepted += 1
                self.last_rejection = ''
        if self.active is not None and self.active.valid_until <= now:
            self.active = None
            self.expirations += 1
        reason = self.guard.input_reason(observation)
        if reason:
            self.active = None
            return self.guard.stop(observation, reason=reason, assessment=SafetyAssessment.INPUT_STALE)
        if self.active is None:
            if enabled and not self.guard.space.can_spin:
                # A timeout must remove propulsion, but need not repeatedly
                # center a car's wheels while its optimizer is preparing a turn.
                # Hold the measured steering with a zero-speed reference, only
                # if the independent hold + braking proof succeeds.
                search = self.guard.space.state_coordinates(self.guard.plant.codec.pack(observation.chassis)).copy()
                search[0] = 0.
                native, valid = self.guard.space.to_native_batch(search[None])
                if valid[0]:
                    candidate = MotionResult(self.guard.space.decode_one(native[0]), MotionStatus.BRAKING,
                        SafetyAssessment.SAFE_UNDER_MODEL, None,
                        Diagnostics(reason='proposal_timeout_hold_steering'), SafetyAction.SOFT_STOP)
                    held = self.guard.enforce(observation, candidate)
                    if held.safe:
                        gap = held.diagnostics.extra.get('final_gate_clearance_m', held.diagnostics.clearance_m)
                        return replace(held, diagnostics=replace(held.diagnostics, clearance_m=gap))
            # Try the smoother stop first, but prove its first execution period
            # and braking continuation with the same swept rectangle checker.
            stop = verified_stop(observation, self.guard.predictor, self.guard.space,
                                 self.guard.index.get(observation.obstacles), mode='soft_stop',
                                 reason='proposal_timeout' if enabled else 'execution_paused')
            return stop if stop.safe else self.guard.stop(observation, reason='timeout_requires_estop')
        command = self.active.command
        candidate = MotionResult(command, MotionStatus.TRACKING, SafetyAssessment.SAFE_UNDER_MODEL, None,
                                 Diagnostics(extra={'proposal_sequence': self.active.sequence,
                                                    'proposal_age_s': now-self.active.issued_at}))
        # Include the upcoming execution interval even just before expiry: the
        # independent actuator watchdog must expire no later than this interval.
        hold = max(observation.dt, self.active.valid_until-now)
        result = self.guard.enforce(replace(observation, dt=hold), candidate)
        if 'final_gate_clearance_m' in result.diagnostics.extra:
            result = replace(result, diagnostics=replace(result.diagnostics,
                             clearance_m=result.diagnostics.extra['final_gate_clearance_m']))
        if result.command != command:
            self.active = None
        return result
