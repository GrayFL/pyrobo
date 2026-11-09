"""Incremental XY elastic-band repair with bounded local sparse optimization."""
from dataclasses import dataclass, replace
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ...core import Footprint, Path
from ...core.geometry import path_poses, planar_state, wrap_angle
from ...geometry.cache import SnapshotIndex
from ...models.kinematics import DifferentialDrive, command_space_for
from ...navigation.tracking import PathTracker
from .config import EBDWAConfig
from .conflict import find_rejoin, inspect_route
from .reference import CurveValidation, EBReference, GeometricCurve, cross, interpolate, polyline, validate_curve


@dataclass(frozen=True)
class RepairResult:
    reference: EBReference | None
    status: str
    diagnostics: dict

    @property
    def success(self):
        return self.reference is not None


class _BudgetExpired(Exception):
    pass


@lru_cache(maxsize=16)
def _jacobian_pattern(node_count, active_nodes, spans):
    """Bounded structural cache; sparsity depends on node indices, not geometry."""
    count = len(spans)
    rows = 7*count+(count-1)+2*len(active_nodes)+(node_count-1)
    pattern = lil_matrix((rows, 2*len(active_nodes)), dtype=np.int8)

    def mark(row, size, nodes):
        columns = [2*j+k for j, node in enumerate(active_nodes) if node in nodes for k in (0, 1)]
        pattern[row:row+size, columns] = 1

    stencils = []
    for i, span in enumerate(spans):
        nodes = set(range(max(0, span-1), min(node_count, span+3)))
        stencils.append(nodes)
        mark(7*i, 7, nodes)
    offset = 7*count
    for i in range(count-1):
        mark(offset+i, 1, stencils[i] | stencils[i+1])
    offset += count-1
    for i, node in enumerate(active_nodes):
        mark(offset+2*i, 2, {node})
    offset += 2*len(active_nodes)
    for i in range(node_count-1):
        mark(offset+i, 1, {i, i+1})
    return pattern.tocsr()


def optimize_band(seed, anchor, parameter, active, world, footprint, bounds, config,
                  *, start_direction, end_direction, gear, max_curvature, deadline):
    """Only active XY nodes are variables; safety is checked after solving."""
    ids = np.flatnonzero(active)
    started, evaluations, iterations = perf_counter(), 0, 0
    best_xy, best_cost, initial_cost = seed.copy(), np.inf, np.inf
    fractions = np.array([0., .25, .5, .75, 1.-1e-6])
    sample_u = np.r_[(parameter[:-1, None]+np.diff(parameter)[:, None]*fractions).ravel(), parameter[-1]]
    weights = np.r_[(np.diff(parameter)[:, None]*np.array([.125, .25, .25, .25, .125])).ravel(), 1e-10]
    def curve_of(xy):
        return GeometricCurve(xy, parameter, start_direction=start_direction, end_direction=end_direction)

    def residual(x):
        nonlocal evaluations, best_xy, best_cost, initial_cost
        if evaluations and (evaluations >= config.max_evaluations or
                            (deadline is not None and perf_counter() >= deadline)):
            raise _BudgetExpired
        evaluations += 1
        xy = seed.copy()
        xy[ids] = x.reshape(-1, 2)
        curve = curve_of(xy)
        p, v, a = (curve.evaluate(sample_u, d) for d in range(3))
        speed = np.maximum(np.linalg.norm(v, axis=1), 1e-4)
        yaw = np.arctan2(v[:, 1], v[:, 0])+(np.pi if gear < 0 else 0.)
        gap = world.clearances(np.column_stack((p, yaw)), footprint, bounds)
        k = cross(v, a)/speed**3
        ds = np.maximum(weights*speed, 1e-5)
        root_ds = np.sqrt(ds)
        hard_k = np.maximum(abs(k)-.94*max_curvature, 0.) if np.isfinite(max_curvature) else np.zeros_like(k)
        geometric = np.column_stack((
            np.sqrt(config.smooth_weight*weights)*a[:, 0],
            np.sqrt(config.smooth_weight*weights)*a[:, 1],
            np.sqrt(config.curvature_weight)*k*root_ds,
            np.sqrt(config.curvature_limit_weight)*hard_k*root_ds,
            np.sqrt(config.obstacle_weight)*np.maximum(config.preferred_clearance-gap, 0.)*root_ds,
            np.sqrt(config.hard_clearance_weight)*np.maximum(config.minimum_clearance+.012-gap, 0.)*root_ds,
            3*np.maximum(.25-speed, 0.)*np.sqrt(weights)))
        rate = np.sqrt(config.curvature_rate_weight)*np.diff(k)/np.sqrt(np.maximum((ds[:-1]+ds[1:])/2, 1e-5))
        # A weak, spatially weighted anchor only applies to free nodes. The rest
        # are fixed, including the rejoin/tail. No uniform full-path attraction.
        anchor_ds = np.r_[np.diff(parameter), parameter[-1]-parameter[-2]]
        guide = np.sqrt(config.global_anchor_weight*anchor_ds[ids])[:, None]*(xy[ids]-anchor[ids])
        # Prevent folded/vanishing segments from winning via a degenerate curve.
        separation = 10*np.maximum(.2*np.diff(parameter)-np.linalg.norm(np.diff(xy, axis=0), axis=1), 0.)
        result = np.r_[geometric.ravel(), rate, guide.ravel(), separation]
        cost = float(result @ result)
        if cost < best_cost:
            best_xy, best_cost = xy.copy(), cost
        if evaluations == 1:
            initial_cost = cost
        return result

    if not len(ids):
        return curve_of(seed), {'evaluations': 0, 'iterations': 0, 'status': 'unchanged',
                                'initial_cost': 0., 'cost': 0., 'duration_s': 0.}
    # Each Hermite span depends on at most four adjacent XY nodes. Curvature-rate
    # residuals cover the union of two neighbouring stencils.
    spans = np.minimum(np.searchsorted(parameter, sample_u, side='right')-1, len(seed)-2)
    pattern = _jacobian_pattern(len(seed), tuple(ids), tuple(spans))
    x0 = seed[ids].ravel()
    lower = np.maximum(anchor[ids]-config.max_deformation_m, [bounds.xmin, bounds.ymin]).ravel()
    upper = np.minimum(anchor[ids]+config.max_deformation_m, [bounds.xmax, bounds.ymax]).ravel()
    x0 = np.clip(x0, lower+1e-9, upper-1e-9)
    status = 'iteration_limit'
    def counted_iteration(x):
        nonlocal iterations
        iterations += 1

    try:
        result = least_squares(residual, x0, bounds=(lower, upper), jac_sparsity=pattern,
                               jac='3-point', max_nfev=config.max_iterations, ftol=1e-4, xtol=1e-4, gtol=1e-4,
                               tr_solver='lsmr', callback=counted_iteration)
        status = 'converged' if result.success else 'iteration_limit'
    except _BudgetExpired:
        status = 'time_budget' if deadline is not None and perf_counter() >= deadline else 'evaluation_limit'
    return curve_of(best_xy), {'evaluations': evaluations, 'iterations': iterations, 'status': status,
                              'initial_cost': initial_cost, 'cost': best_cost, 'duration_s': perf_counter()-started}


class EBPathRepairer:
    """One bounded route cache; no dynamics integration or command generation."""
    def __init__(self, config=None, footprint=None, *, model=None, index=None):
        self.config = config or EBDWAConfig()
        self.footprint = footprint or Footprint()
        self.model = model or DifferentialDrive()
        self.space = command_space_for(self.model)
        self.index = index or SnapshotIndex()
        self.reset()

    def reset(self):
        self.tracker, self._cache_tracker = PathTracker(), PathTracker()
        self._reference = self._anchor_reference = self._key = self._cloud = self._time = self._position = None
        self._cache_diagnostics = {}
        self._pending_curve, self._pending_gear, self._pending_attempt = None, 1, 0
        self._refinement_cycle = 0

    def _validate(self, curve, world, observation, gear):
        cfg = self.config
        if self.space.can_spin:
            pose = planar_state(observation.chassis.pose)
            tangent = curve.tangent[0]
            heading = np.arctan2(tangent[1], tangent[0])+(np.pi if gear < 0 else 0.)
            body = replace(self.footprint, margin=self.footprint.margin+cfg.minimum_clearance)
            if not world.rotation_free(pose[:2], pose[2], heading, body, observation.bounds):
                return CurveValidation(False, 'initial_rotation', 0., 0., 0)
        return validate_curve(curve, world, self.footprint, observation.bounds, gear=gear,
            minimum_clearance=cfg.minimum_clearance,
            max_curvature=np.inf if self.space.can_spin else 1/self.model.minimum_turn_radius,
            depth=cfg.validation_depth, max_intervals=cfg.max_validation_intervals)

    def update(self, observation, global_path, *, goal=None, coverage_m=None, collision_world=None, force=False):
        started, cfg = perf_counter(), self.config
        route = global_path if isinstance(global_path, Path) else Path(global_path)
        if route.frame != observation.frame or (goal is not None and goal.wrd_frame != observation.frame):
            raise ValueError('EB route, goal and observation frames must match')
        state = planar_state(observation.chassis.pose)
        world = collision_world or self.index.get(observation.obstacles)
        key = (observation.frame, observation.bounds, self.footprint, self.model, cfg,
               world.point_radius, tuple(planar_state(goal)) if goal is not None else None)
        changed = key != self._key or not np.array_equal(route.xy, self.tracker.xy)
        jumped = (self._time is not None and (observation.timestamp < self._time or
                  (self._position is not None and np.linalg.norm(state[:2]-self._position) > cfg.min_coverage_m)))
        if changed or jumped:
            self.reset()
            self.tracker.set_path(route.poses)
        self._key, self._time, self._position = key, observation.timestamp, state[:2].copy()
        progress = self.tracker.update(observation.chassis.pose, 0.)
        desired_coverage = min(cfg.max_coverage_m, max(cfg.min_coverage_m, coverage_m or 0.))
        info = {'eb_status': '', 'eb_nodes': 0, 'eb_active_nodes': 0, 'eb_conflict_count': 0,
                'eb_rejoin_s': None, 'eb_coverage_m': 0., 'eb_warm_started': False,
                'eb_cache_hit': False, 'eb_optimized': False, 'eb_evaluations': 0,
                'eb_iterations': 0, 'eb_attempts': [], 'eb_progress_s': progress.distance}

        def finish(reference, status):
            if status in ('no_safe_rejoin', 'invalid_start', 'node_budget', 'at_path_end'):
                self._pending_curve = None
            info.update(eb_status=status, eb_duration_s=perf_counter()-started,
                        eb_solver_pending=self._pending_curve is not None,
                        eb_refinement_cycle=self._refinement_cycle)
            if reference is not None:
                info['eb_coverage_m'] = float(reference.s[-1])
                info['eb_reference_curvature_max'] = float(np.max(abs(reference.curvature)))
                self._reference, self._cloud = reference, observation.obstacles
                if status != 'cached':
                    # Keep a stable optimized curve. Re-anchoring the already
                    # re-anchored output each tick accumulates tracking error
                    # into the route and can fold a perfectly valid bend.
                    self._anchor_reference = reference
                    self._cache_tracker.set_path(path_poses(reference.xy, observation.frame))
                    self._cache_tracker.update(observation.chassis.pose, 0.)
                self._cache_diagnostics = dict(info)
            else:
                # A failed reference must never remain publicly available for use.
                self._reference = None
            return RepairResult(reference, status, dict(info))

        if len(route) < 2 or progress.remaining_distance < 1e-6:
            return finish(None, 'at_path_end')
        if world.clearances(state[None], self.footprint, observation.bounds)[0] <= cfg.minimum_clearance:
            return finish(None, 'invalid_start')

        if cfg.warm_start and self._reference is not None and self._pending_curve is None and not force:
            old = self._anchor_reference
            cached_progress = self._cache_tracker.update(observation.chassis.pose, 0.)
            u = float(np.interp(cached_progress.distance, old.s, old.global_s))
            tail_available = old.s[-1]-cached_progress.distance
            reaches_goal = old.global_s[-1] >= self.tracker.cumulative[-1]-1e-7
            if tail_available >= desired_coverage-.8 or reaches_goal:
                # A short connection absorbs measured lateral tracking error;
                # do not attach to a node only centimetres ahead of the car.
                distance = np.linalg.norm(state[:2]-old.sample(cached_progress.distance)[0])
                join = max(cfg.node_spacing_m, 2*distance)
                if not self.space.can_spin:
                    # A car cannot erase lateral/heading error in a tiny
                    # connector. Reserve arc proportional to its turn radius;
                    # the complete curve is still checked below.
                    tangent = old.curve.evaluate(u, 1)
                    heading_error = abs(float(wrap_angle(state[2]+(np.pi if old.gear < 0 else 0.)-
                                                         np.arctan2(tangent[1], tangent[0]))))
                    radius = self.model.minimum_turn_radius
                    join = max(join, np.sqrt(6*distance*radius), 2*heading_error*radius)
                keep = old.curve.parameter > min(u+join, old.curve.parameter[-1]-1e-8)
                if np.any(keep):
                    parameter = np.r_[u, old.curve.parameter[keep]]
                    xy = np.vstack((state[:2], old.curve.xy[keep]))
                    first = old.curve.evaluate(u, 1)
                    direction = np.arctan2(first[1], first[0]) if self.space.can_spin else state[2]+(np.pi if old.gear < 0 else 0.)
                    last = old.curve.tangent[-1]
                    terminal = np.arctan2(last[1], last[0])
                    if not self.space.can_spin and reaches_goal and len(xy) == 2:
                        # With only the goal left, retaining the old end tangent
                        # overconstrains a centimetre-scale lateral correction.
                        # A circular-arc tangent avoids forcing a second bend
                        # into this short connector; Navigator requires actual goal yaw
                        # and a stopped chassis before declaring arrival.
                        chord = xy[-1]-xy[0]
                        terminal = 2*np.arctan2(chord[1], chord[0])-direction
                    curve = GeometricCurve(xy, parameter, start_direction=direction, end_direction=terminal)
                    validation = self._validate(curve, world, observation, old.gear)
                    if validation.valid:
                        reference = EBReference.from_curve(curve, world, self.footprint, observation.bounds, cfg, gear=old.gear)
                        new_min = float(min(reference.tube_left))
                        previous_width = np.interp(reference.global_s, old.global_s, old.tube_left)
                        invaded = np.any(reference.tube_left < np.minimum(previous_width, cfg.preferred_clearance)-.02)
                        distorted = np.max(abs(reference.curvature)) > max(cfg.preferred_curvature, 2*np.max(abs(old.curvature)))+.1
                        if not distorted and (self._cloud is observation.obstacles or not invaded):
                            info.update(eb_cache_hit=True, eb_warm_started=True, eb_nodes=len(xy),
                                        eb_rejoin_s=self._cache_diagnostics.get('eb_rejoin_s'),
                                        eb_surface_clearance_m=new_min, eb_verified_clearance_m=validation.clearance_m)
                            return finish(reference, 'cached')

        full_xy, full_s = polyline(route.xy)
        s0, scan_end = progress.distance, min(full_s[-1], progress.distance+cfg.max_coverage_m)
        scan_s = np.unique(np.r_[s0, np.arange(s0, scan_end, cfg.obstacle_node_spacing_m),
                                full_s[(full_s > s0) & (full_s < scan_end)], scan_end])
        scan_s = scan_s[np.r_[True, np.diff(scan_s) > 1e-9]]
        scan_xy = interpolate(full_xy, full_s, scan_s)
        forward = scan_xy[min(2, len(scan_xy)-1)]-state[:2]
        reverse = (not self.space.can_spin and cfg.allow_reverse and
                   forward @ np.array([np.cos(state[2]), np.sin(state[2])]) < 0.)
        gear = -1 if reverse else 1
        max_k = np.inf if self.space.can_spin else 1/self.model.minimum_turn_radius
        inspected = inspect_route(scan_xy, scan_s, world, self.footprint, observation.bounds, cfg, gear=gear, max_curvature=max_k)
        end = min(scan_end, s0+desired_coverage)
        soft_end = min(scan_end, end+cfg.active_padding_m)
        rejoin = None
        # Extend through every defect that enters the requested window, including
        # a second obstacle introduced by the first obstacle's required tail.
        for _ in range(len(inspected.active)+1):
            relevant = [(a, b) for a, b in inspected.active if a < end]
            if not relevant:
                break
            # Long low-clearance corridors are one connected soft defect.
            # They must not drag the entire scan window into every solve.
            # Hard conflicts below still extend through their complete rejoin.
            needed = min(soft_end, max(b for _, b in relevant))
            hard = [(a, b) for a, b in inspected.conflicts if a < end]
            if hard:
                # Surface clouds do not label solid interiors. A long wall can
                # appear as separate entry/exit conflicts with a clear centre.
                # Once repair is needed, rejoin after the last conflict in the
                # bounded scan; never accept that apparent interior as a tail.
                rejoin = find_rejoin(inspected, max(b for _, b in inspected.conflicts)+cfg.node_spacing_m,
                                     world, self.footprint, observation.bounds, cfg, gear=gear, max_curvature=max_k)
                if rejoin is None:
                    info['eb_conflict_count'] = len(hard)
                    return finish(None, 'no_safe_rejoin')
                needed = max(needed, rejoin+cfg.rejoin_tail_m)
            new_end = min(scan_end, max(end, needed))
            if new_end <= end+1e-8:
                break
            end = new_end
        active_intervals = [(a, b) for a, b in inspected.active if a < end]
        if not self.space.can_spin:
            delta = scan_xy[1]-scan_xy[0]
            heading = np.arctan2(delta[1], delta[0])+(np.pi if gear < 0 else 0.)
            error = abs(float(wrap_angle(state[2]-heading)))
            lateral = np.linalg.norm(state[:2]-scan_xy[0])
            if error > 1e-3 or lateral > .01:
                # Rejoining a line from a different heading needs an S-shaped
                # transition, not just one turn. Reserve room for both arcs
                # before selecting free nodes, including after a global replan.
                radius = self.model.minimum_turn_radius
                transition = max(cfg.active_padding_m, 4*radius*error, np.sqrt(6*lateral*radius))
                active_intervals.append((s0, min(end, s0+transition)))
        nodes = [s0, end]
        cursor = s0
        while cursor < end-1e-8:
            active_here = any(a-cfg.node_spacing_m <= cursor <= b for a, b in active_intervals)
            cursor = min(end, cursor+(cfg.obstacle_node_spacing_m if active_here else cfg.node_spacing_m))
            nodes.append(cursor)
        parameter = np.unique(nodes)
        # Repeated floating additions can put the last regular node only
        # 1e-14 m before the explicit endpoint. Preserve the exact endpoint and
        # remove numerical zero-length spans before forming cubic derivatives.
        parameter = parameter[np.r_[np.diff(parameter) > 1e-8, True]]
        if not self.space.can_spin and end >= full_s[-1]-1e-8 and end-s0 < cfg.node_spacing_m:
            parameter = np.array([s0, end])
        if len(parameter) > cfg.max_nodes:
            info['eb_required_nodes'] = len(parameter)
            return finish(None, 'node_budget')
        anchor = interpolate(full_xy, full_s, parameter)
        anchor[0] = state[:2]
        active = np.array([any(a <= u <= b for a, b in active_intervals) for u in parameter])
        if np.linalg.norm(state[:2]-scan_xy[0]) > .01:
            active |= parameter < s0+cfg.active_padding_m
        active[[0, -1]] = False
        # Keep a rejoin tail completely fixed, including its tangent-producing
        # neighbour. Smoothing may continue up to the rejoin region.
        if rejoin is not None:
            active[parameter >= rejoin] = False
        first_delta, last_delta = anchor[1]-anchor[0], anchor[-1]-anchor[-2]
        direction = float(np.arctan2(first_delta[1], first_delta[0]))
        body_heading = direction+(np.pi if gear < 0 else 0.)
        hard_body = replace(self.footprint, margin=self.footprint.margin+cfg.minimum_clearance)
        if not self.space.can_spin or not world.rotation_free(state[:2], state[2], body_heading, hard_body, observation.bounds):
            needs_alignment = abs(float(wrap_angle(state[2]-body_heading))) > 1e-3
            direction = state[2]+(np.pi if gear < 0 else 0.)
            if needs_alignment:
                active[1:-1] |= parameter[1:-1] < s0+cfg.active_padding_m
        terminal = float(np.arctan2(last_delta[1], last_delta[0]))
        if not self.space.can_spin and len(anchor) == 2 and end >= full_s[-1]-1e-8:
            terminal = 2*terminal-direction
        info.update(eb_nodes=len(parameter), eb_active_nodes=int(active.sum()),
                    eb_conflict_count=int(sum(a < end for a, _ in inspected.conflicts)), eb_rejoin_s=rejoin,
                    eb_triggers={name: int(mask[inspected.s <= end].sum()) for name, mask in inspected.triggers.items()},
                    eb_active_intervals=active_intervals)
        seed = anchor.copy()
        old = self._reference
        pending = self._pending_curve if cfg.warm_start and self._pending_gear == gear else None
        if pending is not None:
            overlap = active & (parameter >= pending.parameter[0]) & (parameter <= pending.parameter[-1])
            seed[overlap] = pending.evaluate(parameter[overlap])
            info['eb_warm_started'] = bool(overlap.any())
        elif cfg.warm_start and old is not None and old.gear == gear:
            overlap = active & (parameter >= old.global_s[0]) & (parameter <= old.global_s[-1])
            seed[overlap] = old.curve.evaluate(parameter[overlap])
            info['eb_warm_started'] = bool(overlap.any())
        initial = GeometricCurve(seed, parameter, start_direction=direction, end_direction=terminal)
        initial_valid = self._validate(initial, world, observation, gear)
        if not active.any() and initial_valid.valid:
            self._pending_curve, self._refinement_cycle = None, 0
            reference = EBReference.from_curve(initial, world, self.footprint, observation.bounds, cfg, gear=gear)
            info.update(eb_verified_clearance_m=initial_valid.clearance_m,
                        eb_surface_clearance_m=float(reference.tube_left.min()))
            return finish(reference, 'unchanged')
        deadline = None if cfg.time_budget_s is None else started+cfg.time_budget_s
        info['eb_optimized'] = True
        best, reason = None, initial_valid.reason
        # The unchanged seed is a permissible budget fallback only after the same
        # complete sweep and curvature validation as every optimized candidate.
        if initial_valid.valid:
            best = initial
        attempts = cfg.topology_attempts if cfg.fallback_enabled else 1
        first_attempt = self._pending_attempt if pending is not None else 0
        self._pending_curve = None
        self._refinement_cycle = self._refinement_cycle+1 if pending is not None else 1

        def make_proposal(attempt):
            proposal = seed.copy() if attempt == first_attempt else anchor.copy()
            if attempt and not (pending is not None and attempt == first_attempt):
                sign = 1. if attempt % 2 else -1.
                amplitude = cfg.topology_offset_m*(1+(attempt-1)//2)
                delta = np.gradient(anchor, parameter, axis=0)
                normal = np.column_stack((-delta[:, 1], delta[:, 0]))/np.maximum(np.linalg.norm(delta, axis=1)[:, None], 1e-9)
                envelope = np.zeros(len(parameter))
                for a, b in active_intervals:
                    ramp = np.clip(np.minimum(parameter-a, b-parameter)/max(cfg.active_padding_m, .1), 0., 1.)
                    envelope = np.maximum(envelope, ramp*ramp*(3-2*ramp))
                proposal[active] += (sign*amplitude*envelope[:, None]*normal)[active]
            return proposal

        def retain(curve, attempt):
            if cfg.warm_start and self._refinement_cycle < cfg.max_refinement_cycles:
                self._pending_curve, self._pending_attempt, self._pending_gear = curve, attempt, gear

        for attempt in range(first_attempt, attempts):
            proposal = make_proposal(attempt)
            remaining_evaluations = cfg.max_evaluations-info['eb_evaluations']
            if remaining_evaluations <= 0 or (attempt > first_attempt and deadline is not None and perf_counter() >= deadline):
                retain(GeometricCurve(proposal, parameter, start_direction=direction, end_direction=terminal), attempt)
                break
            attempt_config = replace(cfg, max_evaluations=remaining_evaluations)
            curve, diagnostics = optimize_band(proposal, anchor, parameter, active, world, self.footprint,
                observation.bounds, attempt_config, start_direction=direction, end_direction=terminal,
                gear=gear, max_curvature=max_k, deadline=deadline)
            validation = self._validate(curve, world, observation, gear)
            diagnostics.update(topology='nominal' if not attempt else ('left' if attempt % 2 else 'right'),
                               validated=validation.valid, validation_reason=validation.reason)
            info['eb_attempts'].append(diagnostics)
            info['eb_evaluations'] += diagnostics['evaluations']
            info['eb_iterations'] += diagnostics['iterations']
            info['eb_geometry_failure'] = validation.reason
            reason = (diagnostics['status'] if diagnostics['status'] in ('time_budget', 'evaluation_limit')
                      else validation.reason or diagnostics['status'])
            if validation.valid:
                best = curve
                if diagnostics['status'] != 'converged':
                    retain(curve, attempt)
                break
            if diagnostics['status'] in ('time_budget', 'evaluation_limit'):
                retain(curve, attempt)
                break
        if best is None:
            return finish(None, 'no_safe_rejoin' if rejoin is None and inspected.conflicts else reason or 'geometric_infeasible')
        validation = self._validate(best, world, observation, gear)
        reference = EBReference.from_curve(best, world, self.footprint, observation.bounds, cfg, gear=gear)
        info.update(eb_verified_clearance_m=validation.clearance_m,
                    eb_surface_clearance_m=float(reference.tube_left.min()),
                    eb_curvature_bound=validation.curvature_bound)
        return finish(reference, 'repaired' if best is not initial else 'validated_fallback')
