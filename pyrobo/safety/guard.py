"""Final command gate, independent of optimizer status or reported trajectory."""
from dataclasses import dataclass, replace
import numpy as np

from ..core import Diagnostics, MotionResult, MotionStatus, SafetyAction, SafetyAssessment
from ..geometry.cache import SnapshotIndex
from ..geometry.sweep import swept_clearances
from ..models.dynamics.prediction import BatchPredictor
from .fallback import verified_stop


@dataclass(frozen=True)
class SafetyConfig:
    max_state_age: float = .1
    max_cloud_age: float = .3
    max_control_dt: float = .5
    expected_dt: float | None = None
    braking_timeout: float = 4.

    def __post_init__(self):
        for name, value in vars(self).items():
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f'invalid safety parameter: {name}')
        if self.max_control_dt == 0 or self.braking_timeout == 0 or self.expected_dt == 0:
            raise ValueError('control period and braking timeout must be positive')


class SafetyGuard:
    def __init__(self, dynamics, command_space, footprint, *, config=None, index=None):
        self.plant, self.space, self.footprint = dynamics, command_space, footprint
        self.config = config or SafetyConfig()
        self.index = index or SnapshotIndex()
        self.predictor = BatchPredictor(dynamics, footprint, braking_timeout=self.config.braking_timeout)

    def input_reason(self, observation, request=None):
        c, o = self.config, observation
        state_age = o.timestamp - o.chassis.timestamp
        if not -1e-9 <= state_age <= c.max_state_age + 1e-9:
            return 'state_stale'
        stamp = o.obstacles.timestamp
        if stamp is not None and not -1e-9 <= o.timestamp-stamp <= c.max_cloud_age + 1e-9:
            return 'map_stale'
        if o.dt > c.max_control_dt or (c.expected_dt is not None and not np.isclose(o.dt, c.expected_dt, atol=1e-9, rtol=0)):
            return 'control_period_mismatch'
        if o.dt > getattr(self.plant.config, 'timeout', float('inf')) + 1e-9:
            return 'control_period_exceeds_command_timeout'
        if request is not None and not request.fresh(o.timestamp):
            return 'request_stale'
        return ''

    def stop(self, observation, *, reason='stop_requested', assessment=None):
        try:
            return verified_stop(observation, self.predictor, self.space, self.index.get(observation.obstacles),
                                 reason=reason, assessment=assessment)
        except (ValueError, TypeError, FloatingPointError) as error:
            command = self.space.decode_one(np.zeros(self.space.native_size), 'estop')
            return MotionResult(command, MotionStatus.EMERGENCY_STOP, SafetyAssessment.MODEL_INVALID, None,
                                Diagnostics(reason=reason or type(error).__name__,
                                            extra={'safety_override': True, 'stop_verified': False}), SafetyAction.UNVERIFIED)

    def enforce(self, observation, result, request=None, *, emergency=False):
        reason = self.input_reason(observation, request)
        if reason:
            return self.stop(observation, reason=reason, assessment=SafetyAssessment.INPUT_STALE)
        if emergency:
            return self.stop(observation, reason='estop_latched')
        try:
            # Verify the issued representation; never trust an optimizer's safe=True.
            native = self.space.model.encode_command(result.command)[None]
            search = self.space.from_native_batch(native)
            encoded, valid = self.space.to_native_batch(search)
            if not valid.all() or not np.allclose(encoded, native, atol=1e-10, rtol=0):
                return self.stop(observation, reason='command_not_representable', assessment=SafetyAssessment.MODEL_INVALID)
            state = self.plant.codec.pack(observation.chassis)
            batch = self.predictor.predict(state, native, observation.dt, stop_only=True, mode=result.command.mode)
            if not np.isfinite(batch.states).all():
                return self.stop(observation, reason='model_prediction_invalid', assessment=SafetyAssessment.MODEL_INVALID)
            world = self.index.get(observation.obstacles)
            clearance = swept_clearances(self.plant.codec.poses(batch.states), batch.sweep, world,
                                         self.footprint, observation.bounds, batch.valid_time_mask)[0]
            if not np.isfinite(clearance):
                return self.stop(observation, reason='collision_query_invalid', assessment=SafetyAssessment.MODEL_INVALID)
            if not batch.stopped[0] or clearance <= 0:
                stop = self.stop(observation, reason='final_gate_rejected')
                return replace(stop, diagnostics=replace(stop.diagnostics,
                               extra={**stop.diagnostics.extra, 'upstream_reason': result.diagnostics.reason,
                                      'upstream_candidates': result.diagnostics.candidate_count,
                                      'upstream_feasible': result.diagnostics.feasible_count}))
            return replace(result, diagnostics=replace(result.diagnostics,
                           extra={**result.diagnostics.extra, 'final_gate_checked': True,
                                  'final_gate_clearance_m': float(clearance)}))
        except (TypeError, ValueError, FloatingPointError) as error:
            return self.stop(observation, reason='command_adapter_error', assessment=SafetyAssessment.MODEL_INVALID)
