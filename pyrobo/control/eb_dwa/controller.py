"""Global-anchored geometry plus the existing finite-force DWA pipeline."""
from dataclasses import replace
from time import perf_counter

import numpy as np

from ...core import MotionPolicy, Path, SafetyAssessment
from ...core.geometry import path_poses, planar_state, pose2d, wrap_angle
from ...models.kinematics import Ackermann, DifferentialDrive
from ...safety.fallback import verified_stop
from ..dwa import DWAConfig, DWAEngine
from ..dwa.policies import NavigationPolicy
from ..dwa.scoring import navigation_scores
from .band import EBPathRepairer
from .config import EBDWAConfig
from .scoring import score_rollout
from .speed import SpeedPlanner


class _EBNavigationPolicy(NavigationPolicy):
    def __init__(self, footprint, config):
        self.eb_config = config
        super().__init__(footprint)

    def reset(self):
        super().reset()
        self.reference = self.execution_command = self.speed_reference = None
        self.preview_distance = 0.
        self.heading_error = None

    def prepare(self, observation, request, space, config, codec, previous):
        ref = self.reference
        if ref is not None:
            cfg = self.eb_config
            lookahead = min(ref.s[-1], max(cfg.track_lookahead_m,
                abs(observation.chassis.velocity.linear)*cfg.track_lookahead_time_s))
            self.preview_distance = lookahead
            target = ref.sample(min(lookahead, ref.s[-1]))[0]
            yaw = float(np.interp(min(lookahead, ref.s[-1]), ref.s, ref.yaw))
            request = replace(request, local_target=pose2d(*target, yaw, frame=observation.frame),
                              remaining_path=Path(path_poses(ref.xy, observation.frame)))
        context = super().prepare(observation, request, space, config, codec, previous)
        self.heading_error = float(context.signed_error)
        if ref is not None and ref.gear < 0 and context.mode == 'following':
            delta = context.target-context.state[:2]
            error = float(wrap_angle(np.arctan2(delta[1], delta[0])+np.pi-context.state[2]))
            speed = -min(-context.lower[0], space.model.max_reverse_speed)*max(.25, np.cos(error))
            turn = -space.turn_for_heading(error, max(np.linalg.norm(delta), .1))
            context.desired = np.array([speed, np.clip(turn, context.lower[1], context.upper[1])])
            context.anchors = np.vstack((context.anchors, context.desired))
            context.signed_error = error
        if ref is not None and context.mode == 'following' and self.speed_reference is not None:
            cap = abs(np.interp(lookahead, ref.s, self.speed_reference))
            context.desired[0] = np.sign(context.desired[0])*min(abs(context.desired[0]), cap)
            context.anchors = np.vstack((context.anchors, context.desired))
        if ref is not None and context.mode == 'following' and np.linalg.norm(ref.xy[-1]-context.goal[:2]) < 1e-6:
            # Put sub-grid terminal speeds into the ordinary sampler. Without
            # this anchor a long hold horizon can prefer steering away from a
            # straight goal just to avoid passing its endpoint.
            context.desired[0] = np.sign(context.desired[0])*min(abs(context.desired[0]), ref.s[-1]/config.prediction_time)
            context.anchors = np.vstack((context.anchors, context.desired))
        return context

    def score_rollout(self, context, search, prediction, config, space, *, observation, codec, world):
        if context.mode != 'following':
            final = codec.poses(prediction.states[:, prediction.prediction_end])
            clearance = world.clearances(final, self.footprint, observation.bounds) if context.mode == 'recovering' else None
            costs, valid = navigation_scores(context, search, final, config, space, observation,
                                             terminal_clearance=clearance)
            return costs, valid, {'maneuver': costs}
        if self.reference is None:
            raise ValueError('EB following requires a verified geometric reference')
        lower, upper = space.bounds()
        scores = score_rollout(self.reference, prediction, codec, self.eb_config,
                               desired_speed=context.desired[0], search=search,
                               previous_command=self.execution_command, command_scale=upper-lower,
                               speed_reference=self.speed_reference)
        if np.linalg.norm(self.reference.xy[-1]-context.goal[:2]) < 1e-6:
            final = codec.poses(prediction.states[:, prediction.prediction_end])
            # Goal orientation belongs to candidates that actually reach its
            # position tolerance. Applying it a whole lookahead early prevents
            # the lateral approach needed beside a wall.
            distance = np.linalg.norm(final[:, :2]-context.goal[:2], axis=1)
            weight = 2*np.clip(1-distance/self.eb_config.goal_tolerance, 0., 1.)
            body_yaw = self.reference.yaw+(np.pi if self.reference.gear < 0 else 0.)
            if np.max(abs(wrap_angle(body_yaw-context.goal[2]))) <= self.eb_config.yaw_tolerance:
                weight = np.maximum(weight, 2*max(0., 1-context.distance_to_goal/self.eb_config.track_lookahead_m))
            terminal = weight*(1-np.cos(final[:, 2]-context.goal[2]))
            scores.terms['terminal_heading'] = terminal
            scores.costs[:] += terminal
        return scores.costs, np.ones(len(search), dtype=bool), scores.terms


from pyrobo.registry import register_local_optimizer


@register_local_optimizer('eb_dwa', config=EBDWAConfig, label='EB-DWA · 几何修复与动态窗口', family='eb_dwa', teleop_config=True)
class ElasticBandDWA:
    """LocalMotionOptimizer; no duplicate candidate predictor or collision code.

    The optional ``repairs_global_path``/``consume_replan_request`` capability
    lets Navigator retain route memory while EB repairs local map conflicts.
    Every reference and every issued command still undergo separate validation.
    """
    repairs_global_path = True

    def __init__(self, config=None, footprint=None, *, model=None, dynamics=None,
                 command_space=None, index=None, input_limits=None, teleop_config=None):
        self.config = config or EBDWAConfig()
        model = model or getattr(dynamics, 'model', None) or getattr(command_space, 'model', None) or DifferentialDrive()
        if not isinstance(model, (DifferentialDrive, Ackermann)):
            raise TypeError('EB-DWA currently supports DifferentialDrive and Ackermann')
        cfg = self.config
        dwa = DWAConfig(prediction_time=cfg.prediction_time, integration_dt=cfg.integration_dt,
            braking_timeout=cfg.braking_timeout, linear_samples=cfg.linear_samples, angular_samples=cfg.angular_samples,
            batch_size=cfg.batch_size, lookahead=cfg.track_lookahead_m, goal_tolerance=cfg.goal_tolerance,
            yaw_tolerance=cfg.yaw_tolerance, allow_reverse=cfg.allow_reverse,
            clearance_weight=cfg.clearance_weight, reject_lateral_violation=cfg.reject_lateral_violation)
        self.policy = _EBNavigationPolicy(footprint, cfg)
        self._engine = DWAEngine(dwa, footprint, model=model, dynamics=dynamics,
            command_space=command_space, index=index, input_limits=input_limits,
            policies={MotionPolicy.NAVIGATION: self.policy})
        for name in ('model', 'footprint', 'plant', 'command_space', 'index', 'predictor', 'input_guard'):
            setattr(self, name, getattr(self._engine, name))
        self._teleop = DWAEngine(teleop_config or dwa, self.footprint, model=model, dynamics=self.plant,
                                command_space=self.command_space, index=self.index, input_limits=input_limits)
        self.repairer = EBPathRepairer(cfg, self.footprint, model=model, index=self.index)
        self.speed_planner = SpeedPlanner(self.plant, self.command_space, self.footprint, cfg)
        self.reset()

    @property
    def tracker(self):
        return self.repairer.tracker

    def reset(self):
        self._engine.reset()
        self._teleop.reset()
        self.repairer.reset()
        self.speed_planner.reset()
        self.last_reference = None
        self._executed = self._execution_stamp = None
        self._reachability = self._reachability_state = self._reachability_stamp = None
        self._replan_requested, self._failed_candidates = False, 0

    def record_execution(self, command, timestamp):
        """Supply the command actually applied, after any downstream override.

        Standalone/hardware callers should call this after execution feedback.
        A proposal is not feedback. Without recent feedback the smoothing term
        is disabled and its availability is exposed in diagnostics.
        """
        if not np.isfinite(timestamp):
            raise ValueError('execution timestamp must be finite')
        if self._execution_stamp is not None and timestamp < self._execution_stamp:
            raise ValueError('execution feedback must be monotonic; reset after replay')
        native = None if command is None else self.model.encode_command(command)[None]
        self._executed = (self.command_space.from_native_batch(native)[0]
                          if command is not None and command.mode == 'track' else None)
        self._execution_stamp = float(timestamp)

    def consume_replan_request(self):
        requested, self._replan_requested = self._replan_requested, False
        if requested:
            self._failed_candidates = 0
        return requested

    def _coverage(self, observation):
        """Representative shared-plant rollouts, not current speed times horizon."""
        state = self.plant.codec.pack(observation.chassis)
        coordinates = self.command_space.state_coordinates(state)
        reuse = (self._reachability_state is not None and self._reachability_stamp is not None
                 and 0 <= observation.timestamp-self._reachability_stamp < .5
                 and np.max(abs(coordinates-self._reachability_state)) < .15)
        if reuse:
            return self._reachability
        lower, upper = self.command_space.bounds()
        probes = np.array([[upper[0], 0.], [lower[0], 0.], coordinates])
        native, valid = self.command_space.to_native_batch(probes)
        prediction = self.predictor.predict(state, native[valid], observation.dt)
        end = prediction.prediction_end+1
        speed = abs(self.plant.codec.velocities(prediction.states)[..., 0])
        travel = (speed[:, :-1]+speed[:, 1:])*.5*np.diff(prediction.times)
        reachable = float(travel[:, :end-1].sum(axis=1).max())
        braking = self.predictor.predict(state, np.zeros((1, self.command_space.native_size)),
                                         observation.dt, mode='estop', stop_only=True)
        speed = abs(self.plant.codec.velocities(braking.states)[0, :, 0])
        stopping = float(np.sum((speed[:-1]+speed[1:])*.5*np.diff(braking.times)))
        self._reachability = (max(reachable, stopping)+self.config.coverage_margin_m,
                              {'eb_rollout_distance_m': reachable, 'eb_stopping_distance_m': stopping,
                               'eb_reachability_is_outer_bound': False})
        self._reachability_state, self._reachability_stamp = coordinates.copy(), observation.timestamp
        return self._reachability

    def optimize(self, observation, request, *, collision_world=None):
        started = perf_counter()
        self._engine._validate_observation(observation)
        world = collision_world or self.index.get(observation.obstacles)
        reason = self.input_guard.input_reason(observation, request)
        if reason:
            self.repairer.reset()
            self.last_reference = None
            return verified_stop(observation, self.predictor, self.command_space, world,
                                 reason=reason, assessment=SafetyAssessment.INPUT_STALE,
                                 extra={'optimizer': 'eb_dwa'})
        if request.policy in (MotionPolicy.TELEOP_STOP, MotionPolicy.TELEOP_AVOID):
            self.repairer.reset()
            self.policy.reset()
            self.last_reference = None
            result = self._teleop.optimize(observation, request, collision_world=world)
            return replace(result, diagnostics=replace(result.diagnostics,
                extra={**result.diagnostics.extra, 'optimizer': 'dwa', 'requested_optimizer': 'eb_dwa'}))
        fresh_feedback = (self._execution_stamp is not None and
                          0 <= observation.timestamp-self._execution_stamp <= self.config.prediction_time)
        self.policy.execution_command = self._executed if fresh_feedback else None
        self.policy.reference = None
        self.policy.speed_reference = None
        info = {}
        if request.policy == MotionPolicy.NAVIGATION:
            position = planar_state(observation.chassis.pose)[:2]
            near_goal = np.linalg.norm(position-planar_state(request.goal)[:2]) <= self.config.goal_tolerance
            if not near_goal:
                coverage, reachability = self._coverage(observation)
                repaired = self.repairer.update(observation, request.path, goal=request.goal,
                    coverage_m=coverage, collision_world=world, force=self._failed_candidates >= 3)
                info.update(repaired.diagnostics, **reachability)
                self.last_reference = repaired.reference
                if not repaired.success:
                    self._replan_requested = self.config.fallback_enabled and not info['eb_solver_pending']
                    result = verified_stop(observation, self.predictor, self.command_space, world,
                        reason=repaired.status, extra={**info, 'optimizer': 'eb_dwa', 'controller_mode': 'blocked',
                                                       'request_replan': self._replan_requested})
                    return replace(result, diagnostics=replace(result.diagnostics, candidate_count=0, feasible_count=0,
                                                                 elapsed_s=perf_counter()-started))
                self.policy.reference = repaired.reference
                terminal = np.linalg.norm(repaired.reference.xy[-1]-planar_state(request.goal)[:2]) < 1e-6
                if self.config.speed_profile:
                    profile = self.speed_planner.update(repaired.reference, observation, terminal=terminal)
                    self.policy.speed_reference = profile.values
                    info.update(profile.diagnostics)
        dwa_started = perf_counter()
        result = self._engine.optimize(observation, request, collision_world=world)
        dwa_duration = perf_counter()-dwa_started
        if request.policy == MotionPolicy.NAVIGATION:
            self._failed_candidates = self._failed_candidates+1 if result.diagnostics.feasible_count == 0 else 0
            if self._failed_candidates >= 6:
                self._replan_requested = self.config.fallback_enabled
        info.update(optimizer='eb_dwa', dwa_duration_s=dwa_duration,
                    eb_track_lookahead_m=self.policy.preview_distance,
                    eb_heading_error=self.policy.heading_error,
                    dwa_candidate_count=result.diagnostics.candidate_count,
                    dwa_feasible_count=result.diagnostics.feasible_count,
                    eb_execution_feedback=bool(fresh_feedback and self._executed is not None),
                    request_replan=self._replan_requested)
        return replace(result, diagnostics=replace(result.diagnostics, elapsed_s=perf_counter()-started,
                                                    extra={**result.diagnostics.extra, **info}))
