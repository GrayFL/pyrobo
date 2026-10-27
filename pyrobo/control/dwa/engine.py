"""Shared local optimizer: one batch pipeline for navigation and assisted teleop."""
from dataclasses import replace
from time import perf_counter

import numpy as np

from ...core import (Diagnostics, MotionPolicy, MotionResult, MotionStatus,
                     SafetyAction, SafetyAssessment)
from pyrobo.models.dynamics.chassis import ChassisDynamics
from pyrobo.models.dynamics.config import DynamicsConfig
from pyrobo.models.kinematics.differential import DifferentialDrive
from ...geometry.cache import SnapshotIndex
from ...models.kinematics import command_space_for
from ...models.dynamics.prediction import BatchPredictor
from ...models.dynamics.reachability import ReachabilityEstimator
from ...safety.fallback import selected_trajectory, verified_stop
from ...safety.guard import SafetyConfig, SafetyGuard
from .config import DWAConfig
from .policies import NavigationPolicy, TeleopPolicy
from .rollout import MotionBatch, evaluate_safety
from .sampling import sample_window
from .scoring import navigation_scores, teleop_scores


class DWAEngine:
    def __init__(self, config=None, footprint=None, *, model=None, dynamics=None,
                 command_space=None, reachability=None, index=None, policies=None, input_limits=None):
        from pyrobo.core.footprint import Footprint
        base_config = config or DWAConfig()
        self.model = (model or getattr(dynamics, 'model', None) or getattr(command_space, 'model', None) or
                      DifferentialDrive(max_speed=base_config.max_speed, max_yaw_rate=base_config.max_yaw_rate))
        self.command_space = command_space or command_space_for(self.model)
        self.config = config or (DWAConfig() if self.command_space.can_spin else DWAConfig(prediction_time=2.4, lookahead=1.6))
        self.footprint = footprint or Footprint()
        self.plant = dynamics if hasattr(dynamics, 'integrate') else ChassisDynamics(self.model, dynamics or DynamicsConfig(
            drive_acceleration=self.config.linear_acceleration, yaw_acceleration=self.config.angular_acceleration))
        if getattr(self.plant, 'model', self.model) != self.model or self.command_space.model != self.model:
            raise ValueError('dynamics and command space must use the same model')
        self.predictor = BatchPredictor(self.plant, self.footprint, prediction_time=self.config.prediction_time,
                                        recording_dt=self.config.integration_dt, braking_timeout=self.config.braking_timeout)
        self.reachability = reachability or ReachabilityEstimator(self.plant, self.command_space)
        self.index = index or SnapshotIndex()
        self.input_guard = SafetyGuard(self.plant, self.command_space, self.footprint,
                                      index=self.index, config=input_limits or SafetyConfig())
        self.navigation_policy = NavigationPolicy()
        self.teleop_policy = TeleopPolicy()
        self.policies = {MotionPolicy.NAVIGATION: self.navigation_policy,
                         MotionPolicy.TELEOP_STOP: self.teleop_policy, MotionPolicy.TELEOP_AVOID: self.teleop_policy}
        if policies:
            self.policies.update(policies)
        self.previous = None
        self.last_window = None
        self._last_policy = None

    @property
    def tracker(self):
        return self.navigation_policy.tracker

    def reset(self):
        # Keep injected policies installed; reset bounded controller memory only.
        for policy in {id(p): p for p in (self.navigation_policy, self.teleop_policy, *self.policies.values())}.values():
            reset = getattr(policy, 'reset', None)
            if reset is not None:
                reset()
        self.previous, self.last_window, self._last_policy = None, None, None
        self.index.reset()

    def _validate_observation(self, observation):
        if observation.dt > self.config.prediction_time:
            raise ValueError('control dt must not exceed prediction_time')
        if observation.frame != observation.chassis.pose.wrd_frame:
            raise ValueError('pose and obstacles must have the same explicit frame')
        return self.plant.codec.pack(observation.chassis)

    def _prepare(self, observation, request):
        if request.policy == MotionPolicy.NAVIGATION:
            if request.path.frame != observation.frame or request.goal.wrd_frame != observation.frame:
                raise ValueError('goal and path frames must match the robot')
        if request.policy != self._last_policy:
            self.navigation_policy.turn_active = False
        self._last_policy = request.policy
        policy = self.policies[request.policy]
        return policy.prepare(observation, request, self.command_space, self.config, self.plant.codec, self.previous)

    def evaluate_candidates(self, observation, request, search, *, context=None, world=None):
        """Inspect one numerical batch. No candidate commands or poses become objects."""
        state = self._validate_observation(observation)
        context = context or self._prepare(observation, request)
        world = world or self.index.get(observation.obstacles)
        native, command_ok = self.command_space.to_native_batch(search)
        prediction, clearance, masks = evaluate_safety(self.predictor, state, native, observation.dt, world,
                                                       observation.bounds, diagnostics=self.config.collect_batch_diagnostics or
                                                       self.config.reject_lateral_violation)
        final = self.plant.codec.poses(prediction.states[:, prediction.prediction_end])
        policy = self.policies[request.policy]
        score = getattr(policy, 'score_batch', None)
        if score is not None:
            objective, policy_ok = score(context, search, final, self.config, self.command_space)
        elif request.policy == MotionPolicy.NAVIGATION:
            objective, policy_ok = navigation_scores(context, search, final, self.config, self.command_space, observation)
        else:
            objective, policy_ok = teleop_scores(context, search, final, self.config, self.command_space)
        masks['command_valid'] = command_ok
        masks['policy_valid'] = policy_ok & np.all((search >= context.lower - 1e-9) & (search <= context.upper + 1e-9), axis=1)
        trajectory_mask = getattr(self.policies[request.policy], 'trajectory_mask', None)
        if trajectory_mask is not None:
            end = prediction.prediction_end + 1
            masks['intent_deviation'] = trajectory_mask(context, self.plant.codec.poses(prediction.states[:, :end]),
                                                        prediction.times[:end], self.command_space)
        if self.config.reject_lateral_violation:
            masks['lateral_valid'] = ~prediction.constraint_flags['lateral_envelope_exceeded']
        feasible = np.logical_and.reduce(list(masks.values()))
        obstacle_cost = self.config.clearance_weight / (.1 + np.maximum(clearance, 0.))
        costs = objective + obstacle_cost
        costs[~feasible] = np.inf
        return MotionBatch(search, native, prediction, feasible, clearance, costs, masks,
                           {'objective': objective, 'clearance': obstacle_cost})

    def optimize(self, observation, request, *, collision_world=None):
        started = perf_counter()
        state = self._validate_observation(observation)
        world = collision_world or self.index.get(observation.obstacles)
        reason = self.input_guard.input_reason(observation, request)
        if reason:
            return verified_stop(observation, self.predictor, self.command_space, world, reason=reason,
                                 assessment=SafetyAssessment.INPUT_STALE)
        if request.policy == MotionPolicy.STOP:
            return verified_stop(observation, self.predictor, self.command_space, world, mode=request.stop_mode)
        context = self._prepare(observation, request)
        window = self.reachability.command_window(state, request, observation.dt, lower=context.lower,
                                                   upper=context.upper, anchors=context.anchors)
        self.last_window = window
        counts = (self.config.linear_samples,) + (self.config.angular_samples,)*(self.command_space.search_size-1)
        search = sample_window(window, counts)
        total, feasible_count = len(search), 0
        best, best_score = None, np.inf
        batch_size = self.config.batch_size or max(1, total)
        # Each iteration handles an ndarray block, never an individual candidate.
        for offset in range(0, total, batch_size):
            batch = self.evaluate_candidates(observation, request, search[offset:offset+batch_size], context=context, world=world)
            feasible_count += int(batch.feasible.sum())
            if batch.feasible.any():
                index = int(np.argmin(batch.costs))
                if batch.costs[index] < best_score - 1e-12:
                    best_score = float(batch.costs[index])
                    best = (batch, index, offset + index)
        if best is None:
            result = verified_stop(observation, self.predictor, self.command_space, world, reason='no_feasible_candidate',
                                   extra={'controller_mode': 'blocked', 'window_fallback': window.fallback_reason})
            return replace(result, diagnostics=replace(result.diagnostics, elapsed_s=perf_counter()-started,
                                                        candidate_count=total, feasible_count=0,
                                                        requested_command=tuple(context.desired)))
        batch, index, candidate_id = best
        chosen = batch.search_commands[index]
        self.previous = chosen.copy()
        status, action = MotionStatus.TRACKING, SafetyAction.NONE
        if request.policy != MotionPolicy.NAVIGATION:
            deviation = chosen - context.desired
            if request.policy == MotionPolicy.TELEOP_AVOID and len(deviation) > 1 and abs(deviation[1]) > .03:
                status = MotionStatus.OBSTACLE_AVOIDING
            elif np.linalg.norm(deviation) > 1e-5:
                status = MotionStatus.CONSTRAINED
            action = SafetyAction.NONE if status == MotionStatus.TRACKING else SafetyAction.LIMIT
            selected = getattr(self.policies[request.policy], 'selected', None)
            if selected is not None:
                selected(observation.timestamp, chosen, context.desired)
        command = self.command_space.decode_one(batch.native_commands[index])
        diagnostics = Diagnostics(perf_counter()-started, total, feasible_count, float(batch.clearance[index]),
                                  tuple(context.desired), tuple(batch.native_commands[index]),
                                  {key: float(values[index]) for key, values in batch.cost_terms.items()},
                                  extra={'controller_mode': context.mode, 'candidate_id': candidate_id,
                                         'search_command': tuple(chosen), 'window_fallback': window.fallback_reason,
                                         'envelope_strict': bool(window.envelope and window.envelope.strict_outer_bound),
                                         'constraint_flags': {key: bool(values[index]) for key, values in batch.prediction.constraint_flags.items()}})
        return MotionResult(command, status, SafetyAssessment.SAFE_UNDER_MODEL,
                            selected_trajectory(batch.prediction, index, observation, self.plant.codec), diagnostics, action)
