"""Model-driven stop verification, usable independently of any optimizer."""
import numpy as np

from ..core import Diagnostics, MotionResult, MotionStatus, SafetyAction, SafetyAssessment, Trajectory
from ..geometry.sweep import swept_clearances
from pyrobo.core.geometry import pose_array


def selected_trajectory(batch, index, observation, codec):
    mask = batch.valid_time_mask[index]
    values = batch.states[index, mask]
    poses = pose_array(codec.poses(values), observation.frame, observation.chassis.pose.ego_frame)
    return Trajectory(poses, batch.times[mask], observation.timestamp, values,
                      min(batch.prediction_end, len(values)-1))


def verified_stop(observation, predictor, space, world, *, mode='estop', reason='stop_requested',
                  assessment=None, extra=None):
    native = np.zeros((1, space.native_size))
    command = space.decode_one(native[0], mode)
    state = predictor.codec.pack(observation.chassis)
    batch = predictor.predict(state, native, observation.dt, mode=mode, stop_only=True)
    clearance = float(swept_clearances(predictor.codec.poses(batch.states), batch.sweep, world,
                                       predictor.footprint, observation.bounds, batch.valid_time_mask)[0])
    verified = bool(batch.stopped[0] and np.isfinite(batch.states).all() and np.isfinite(clearance) and clearance > 0)
    safety = assessment or (SafetyAssessment.SAFE_UNDER_MODEL if verified else SafetyAssessment.BRAKING_UNVERIFIED)
    return MotionResult(command, MotionStatus.BRAKING if verified else MotionStatus.EMERGENCY_STOP,
                        safety, selected_trajectory(batch, 0, observation, predictor.codec),
                        Diagnostics(candidate_count=1, feasible_count=int(verified), clearance_m=clearance,
                                    selected_command=tuple(native[0]), reason=reason,
                                    extra={'safety_override': True, 'stop_verified': verified, **(extra or {})}),
                        SafetyAction.ESTOP if mode == 'estop' else SafetyAction.SOFT_STOP)
