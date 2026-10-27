"""Control authority and timed intent arbitration; never integrates chassis state."""
from dataclasses import replace

import numpy as np

from ..core import MotionPolicy, MotionRequest, SafetyAssessment
from ..teleop import TeleopAdapter
from .modes import ControlMode


class Supervisor:
    def __init__(self, navigator, *, optimizer=None, teleop=None, safety=None,
                 mode=ControlMode.AUTO, authorized_sources=('operator',), auto_task_ttl=None):
        self.navigator = navigator
        self.optimizer = optimizer or navigator.optimizer
        self.safety = safety or navigator.safety
        self.teleop = teleop or TeleopAdapter(navigator.command_space)
        self.authorized_sources = frozenset(authorized_sources)
        if not self.authorized_sources or any(not isinstance(s, str) or not s for s in self.authorized_sources):
            raise ValueError('authorized sources must be nonempty names')
        if auto_task_ttl is not None and (not np.isfinite(auto_task_ttl) or auto_task_ttl <= 0):
            raise ValueError('auto_task_ttl must be positive or None')
        self.auto_task_ttl = auto_task_ttl
        self._task_stamp = None
        self._mode = ControlMode(mode)
        self._last_timestamp = None
        self.last_request = self.last_motion = None

    @property
    def mode(self):
        return self._mode

    def _authorize(self, source):
        if source not in self.authorized_sources:
            raise PermissionError(f'control source {source!r} is not authorized')

    def set_mode(self, mode, *, source='operator'):
        self._authorize(source)
        mode = ControlMode(mode)
        if self.mode == ControlMode.ESTOP and mode != ControlMode.ESTOP:
            raise RuntimeError('estop is latched; call release_estop explicitly')
        self._mode = mode

    def release_estop(self, *, source='operator'):
        self._authorize(source)
        if self.mode != ControlMode.ESTOP:
            raise RuntimeError('estop is not engaged')
        self._mode = ControlMode.PAUSED
        self.teleop.clear()  # an old held joystick must not silently restart motion

    def renew_task(self, timestamp, *, source='operator'):
        self._authorize(source)
        if not np.isfinite(timestamp) or (self._task_stamp is not None and timestamp < self._task_stamp):
            raise ValueError('task timestamp must be finite and monotonic')
        self._task_stamp = float(timestamp)

    def set_goal(self, goal, timestamp, *, source='operator'):
        self.renew_task(timestamp, source=source)
        self.navigator.set_goal(goal)

    def submit_teleop(self, desired, timestamp, *, source='operator', steering=None, policy=None):
        self._authorize(source)
        return self.teleop.submit(desired, timestamp, steering=steering, source=source, policy=policy)

    def _finish(self, result, request=None):
        self.last_request = request
        result = replace(result, diagnostics=replace(result.diagnostics,
                         extra={**result.diagnostics.extra, 'control_mode': self.mode.value,
                                'input_source': request.source if request else None}))
        self.last_motion = result
        return result

    def step(self, observation):
        o = observation
        reason = self.safety.input_reason(o)
        if self._last_timestamp is not None and o.timestamp <= self._last_timestamp:
            reason = 'observation_not_monotonic'
        if reason:
            return self._finish(self.safety.stop(o, reason=reason, assessment=SafetyAssessment.INPUT_STALE))
        self._last_timestamp = o.timestamp
        try:
            self.navigator.observe(o)
        except (ValueError, TypeError, FloatingPointError, OverflowError):
            return self._finish(self.safety.stop(o, reason='navigation_observation_error',
                                                assessment=SafetyAssessment.MODEL_INVALID))
        if self.mode != ControlMode.AUTO:
            self.navigator.suspend_tracking(o)
        if self.mode == ControlMode.ESTOP:
            return self._finish(self.safety.stop(o, reason='estop_latched'))
        if self.mode == ControlMode.PAUSED:
            return self._finish(self.safety.stop(o, reason='paused'))
        if self.mode == ControlMode.MANUAL:
            request = self.teleop.latest_request(o.timestamp)
            if request is None:
                return self._finish(self.safety.stop(o, reason='teleop_timeout', assessment=SafetyAssessment.INPUT_STALE))
            # Validate again even if an external caller has directly accessed the adapter.
            if request.source not in self.authorized_sources:
                return self._finish(self.safety.stop(o, reason='unauthorized_source', assessment=SafetyAssessment.INPUT_STALE))
        else:
            if self.auto_task_ttl is not None and (self._task_stamp is None or
                    not -1e-9 <= o.timestamp - self._task_stamp <= self.auto_task_ttl + 1e-9):
                return self._finish(self.safety.stop(o, reason='auto_task_timeout', assessment=SafetyAssessment.INPUT_STALE))
            try:
                request = self.navigator.make_motion_request(o)
            except (ValueError, TypeError, FloatingPointError, OverflowError):
                return self._finish(self.safety.stop(o, reason='planning_error', assessment=SafetyAssessment.MODEL_INVALID))
        try:
            result = self.optimizer.optimize(o, request)
        except (ValueError, TypeError, FloatingPointError, OverflowError):
            return self._finish(self.safety.stop(o, reason='optimizer_error', assessment=SafetyAssessment.MODEL_INVALID), request)
        return self._finish(self.safety.enforce(o, result, request), request)
