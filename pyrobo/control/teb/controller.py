"""Receding-horizon TEB with model-native, finite-force command verification."""
from dataclasses import replace
from time import perf_counter

import numpy as np

from ...core import Diagnostics, MotionPolicy, MotionResult, MotionStatus, SafetyAssessment
from ...core.geometry import wrap_angle
from ...models.kinematics import Ackermann, DifferentialDrive
from ...safety.fallback import selected_trajectory, verified_stop
from ..dwa import DWAConfig, DWAEngine
from ..dwa.path import path_lookahead
from .band import segment_motion, solve_band
from .config import TEBConfig


class TimedElasticBand:
    """LocalMotionOptimizer for differential and Ackermann vehicles.

    NAVIGATION optimizes a pose/time band and verifies a small set of tracking
    references with the shared plant. Assisted teleop explicitly delegates to
    the existing DWA policy engine, preserving its bounded manual intent.
    ``last_band`` is an optimization diagnostic, never an executable trajectory.
    """
    def __init__(self, config=None, footprint=None, *, model=None, dynamics=None,
                 command_space=None, index=None, input_limits=None, teleop_config=None):
        self.config = config or TEBConfig()
        model = model or getattr(dynamics, 'model', None) or getattr(command_space, 'model', None) or DifferentialDrive()
        if not isinstance(model, (DifferentialDrive, Ackermann)):
            raise TypeError('TEB currently supports DifferentialDrive and Ackermann')
        cfg = self.config
        tracking = DWAConfig(prediction_time=cfg.prediction_time, integration_dt=cfg.integration_dt,
            braking_timeout=cfg.braking_timeout, lookahead=cfg.lookahead, goal_tolerance=cfg.goal_tolerance,
            yaw_tolerance=cfg.yaw_tolerance, allow_reverse=cfg.allow_reverse,
            reject_lateral_violation=cfg.reject_lateral_violation)
        self._tracking = DWAEngine(tracking, footprint, model=model, dynamics=dynamics,
                                   command_space=command_space, index=index, input_limits=input_limits)
        for name in ('model', 'footprint', 'plant', 'command_space', 'index', 'predictor', 'input_guard'):
            setattr(self, name, getattr(self._tracking, name))
        self._teleop = DWAEngine(teleop_config or tracking, self.footprint, model=model, dynamics=self.plant,
                                command_space=self.command_space, index=self.index, input_limits=input_limits)
        self.reset()

    @property
    def tracker(self):
        return self._tracking.tracker

    def reset(self):
        self._tracking.reset()
        self._teleop.reset()
        self.last_band = None
        self._last_path = self._last_key = self._last_time = None

    def _seed(self, observation, context):
        cfg, space, c = self.config, self.command_space, context
        count = int(np.clip(np.ceil(cfg.prediction_time/cfg.dt_ref)+1, cfg.min_samples, cfg.max_samples))
        u = np.linspace(0., 1., count)
        if c.mode in ('turning', 'turn_braking', 'aligning'):
            angle = float(wrap_angle((c.goal[2] if c.mode == 'aligning' else c.direction)-c.state[2]))
            reference = np.column_stack((np.tile(c.state[:2], (count, 1)), c.state[2]+u*angle))
        elif c.mode in ('recovering', 'pose_aligning'):
            v = c.desired[0]
            if c.mode == 'pose_aligning':
                v = c.upper[0]
            w = space.yaw_reference(np.array([[v, c.desired[1]]]))[0]
            t = u*cfg.prediction_time
            yaw = c.state[2]+w*t
            travel = v*t*np.sinc(w*t/(2*np.pi))
            mid = c.state[2]+.5*w*t
            reference = np.column_stack((c.state[0]+travel*np.cos(mid), c.state[1]+travel*np.sin(mid), yaw))
        else:
            xy = np.vstack((c.state[:2], c.remaining))
            xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]]
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            length = min(distance[-1], max(cfg.lookahead, space.bounds()[1][0]*cfg.prediction_time))
            sample = u*length
            points = np.column_stack([np.interp(sample, distance, xy[:, j]) for j in range(2)])
            delta = np.diff(points, axis=0)
            yaw = np.r_[np.arctan2(delta[:, 1], delta[:, 0]), 0.]
            if (not space.can_spin and cfg.allow_reverse and space.bounds()[0][0] < 0.
                    and delta[0] @ np.array([np.cos(c.state[2]), np.sin(c.state[2])]) < 0.):
                yaw[:-1] += np.pi
            yaw[-1] = c.goal[2] if distance[-1] <= length+1e-8 else yaw[-2]
            reference = np.column_stack((points, np.unwrap(yaw)))
        reference[0] = c.state
        reference[:, 2] = c.state[2]+np.unwrap(wrap_angle(reference[:, 2]-c.state[2]))
        intervals = np.full(count-1, cfg.prediction_time/(count-1))
        key = (observation.frame, observation.bounds, tuple(c.goal), c.mode)
        elapsed = observation.timestamp-self._last_time if self._last_time is not None else np.inf
        warm = (cfg.warm_start and self.last_band is not None and self._last_key == key
                and 0 < elapsed <= cfg.prediction_time
                and np.array_equal(self._last_path, c.request.path.xy)
                and np.linalg.norm(c.state[:2]-self.last_band.poses[0, :2]) < cfg.lookahead)
        seed = reference.copy()
        if warm:
            old = self.last_band
            times = elapsed+u*max(old.times[-1]-elapsed, cfg.dt_min*(count-1))
            old_yaw = c.state[2]+np.unwrap(wrap_angle(old.poses[:, 2]-c.state[2]))
            previous = np.column_stack([np.interp(times, old.times, values) for values in
                                        (old.poses[:, 0], old.poses[:, 1], old_yaw)])
            # Re-anchor both ends to this observation and the current route horizon.
            previous += u[:, None]*(reference[-1]-previous[-1])
            seed[1:-1] = .7*previous[1:-1]+.3*reference[1:-1]
            intervals = np.diff(times)
        self._last_key, self._last_time = key, observation.timestamp
        self._last_path = c.request.path.xy.copy()
        return seed, np.clip(intervals, cfg.dt_min+1e-9, cfg.dt_max-1e-9), reference, bool(warm)

    def _commands(self, observation, context, band):
        """Continuous band velocities and feedback references; no velocity grid."""
        space, c = self.command_space, context
        v, w, _ = segment_motion(band.poses, band.intervals)
        commands = []
        # Several short time previews compensate for the finite-force plant's lag.
        for horizon in (.3, .6, 1.):
            duration = np.minimum(band.intervals, np.maximum(horizon-band.times[:-1], 0.))
            speed, yaw = np.array([v, w]) @ duration/max(duration.sum(), 1e-9)
            turn = yaw if space.can_spin else np.arctan(space.model.wheelbase*yaw*speed/(speed*speed+1e-8))
            commands.append([speed, turn])
        if c.mode == 'following':
            target, _ = path_lookahead(band.poses[:, :2], c.state[:2], self.config.lookahead)
            direction = np.arctan2(target[1]-c.state[1], target[0]-c.state[0])
            error = float(wrap_angle(direction-c.state[2]))
            turn = (error if space.can_spin else space.turn_for_heading(error, max(np.linalg.norm(target-c.state[:2]), .1)))
            commands.append([c.desired[0], turn])
        else:
            commands.append(c.desired)
        base = np.asarray(commands)
        # Slow driving arcs keep their curvature: differential yaw scales with
        # speed, while a car retains its steering angle. Keeping yaw fixed on a
        # slow arc would tighten the turn and trap the body at inside corners.
        # The zero-speed row is a separate spin maneuver; retain all four turn
        # references for heading correction, plus the existing recovery policy.
        curved = space.can_spin and c.mode == 'following'
        search = np.concatenate([base*np.array([scale, scale if curved and scale > 0. else 1.])
                                 for scale in (1., .65, .3, 0.)])
        search = np.clip(search, c.lower, c.upper)
        return np.unique(search, axis=0)

    def optimize(self, observation, request, *, collision_world=None):
        started = perf_counter()
        engine = self._tracking
        engine._validate_observation(observation)
        world = collision_world or self.index.get(observation.obstacles)
        reason = self.input_guard.input_reason(observation, request)
        if reason or request.policy == MotionPolicy.STOP:
            self.last_band = None
            return verified_stop(observation, self.predictor, self.command_space, world,
                mode=request.stop_mode, reason=reason or 'stop_requested',
                assessment=SafetyAssessment.INPUT_STALE if reason else None, extra={'optimizer': 'teb'})
        if request.policy != MotionPolicy.NAVIGATION:
            self.last_band = None
            # Clear autonomous intent without invalidating the shared immutable
            # point index on every manual frame (teleop and the final gate use it).
            engine.navigation_policy.reset()
            engine.previous, engine._last_policy = None, None
            result = self._teleop.optimize(observation, request, collision_world=world)
            return replace(result, diagnostics=replace(result.diagnostics,
                extra={**result.diagnostics.extra, 'optimizer': 'dwa', 'navigation_optimizer': 'teb'}))
        context = engine._prepare(observation, request)
        seed, intervals, reference, warm = self._seed(observation, context)
        band = solve_band(seed, intervals, reference, observation, world, self.footprint,
                          self.command_space, self.plant.config, self.config, warm_started=warm)
        self.last_band = band
        search = self._commands(observation, context, band)
        batch = engine.evaluate_candidates(observation, request, search, context=context, world=world)
        if context.mode == 'following':
            end = batch.prediction.prediction_end+1
            times = batch.prediction.times[:end]
            guide = np.column_stack([np.interp(times, band.times, band.poses[:, j]) for j in range(2)])
            predicted = self.plant.codec.poses(batch.prediction.states[:, :end])
            tracking_cost = .5*np.mean(np.sum((predicted[..., :2]-guide)**2, axis=2), axis=1)
            batch.costs += tracking_cost
            batch.cost_terms['band_tracking'] = tracking_cost
        extra = {'optimizer': 'teb', 'controller_mode': context.mode, 'teb_status': band.status,
                 'teb_nodes': len(band.poses), 'teb_evaluations': band.evaluations,
                 'teb_iterations': band.iterations, 'teb_cost': band.cost, 'teb_initial_cost': band.initial_cost,
                 'teb_duration_s': float(band.times[-1]), 'teb_warm_started': band.warm_started}
        if not batch.feasible.any():
            result = verified_stop(observation, self.predictor, self.command_space, world,
                reason='no_feasible_teb_command', extra={**extra, 'controller_mode': 'blocked'})
            return replace(result, diagnostics=replace(result.diagnostics, elapsed_s=perf_counter()-started,
                candidate_count=len(search), feasible_count=0, requested_command=tuple(context.desired)))
        # Follow the optimized geometry with the actual physical prediction.
        # Terminal navigation scoring retains the established arrival/recovery policy.
        best = float(np.min(batch.costs))
        tied = np.flatnonzero(batch.feasible & (batch.costs == best))
        scale = np.maximum(context.upper-context.lower, 1e-9)
        index = int(tied[np.argmin(np.sum(((search[tied]-context.desired)/scale)**2, axis=1))])
        engine.previous = search[index].copy()
        diagnostics = Diagnostics(perf_counter()-started, len(search), int(batch.feasible.sum()),
            float(batch.clearance[index]), tuple(context.desired), tuple(batch.native_commands[index]),
            {key: float(values[index]) for key, values in batch.cost_terms.items()},
            extra={**extra, 'search_command': tuple(search[index]), 'candidate_id': index})
        return MotionResult(self.command_space.decode_one(batch.native_commands[index]), MotionStatus.TRACKING,
            SafetyAssessment.SAFE_UNDER_MODEL,
            selected_trajectory(batch.prediction, index, observation, self.plant.codec), diagnostics)


TEBEngine = TimedElasticBand
