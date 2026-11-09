"""Spatial polyline projection and trajectory-integral costs, without a plant.

Candidate rows remain ndarray batches. Time steps are processed in order so
each projection starts from the preceding arc position, including reversals.
This module never turns geometric scores into collision/safety acceptance.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionBatch:
    s: np.ndarray
    xy: np.ndarray
    lateral: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    tube_left: np.ndarray
    tube_right: np.ndarray
    endpoint_excess: np.ndarray


def project_trajectories(reference, points, *, start_s=0., slack_m=.25):
    """Exact segment projection within a displacement-dependent arc window.

    ``start_s`` is the known branch at rollout start, normally zero because an
    EB reference begins at the measured robot position. Negative start-extension
    progress distinguishes real retreat from standing still; there is no reward
    for going beyond the reference's final endpoint.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 3 or points.shape[2] != 2 or not np.isfinite(points).all():
        raise ValueError('projection requires finite (N,T,2) positions')
    count, steps, _ = points.shape
    if not steps or not np.isfinite(slack_m) or slack_m <= 0:
        raise ValueError('projection requires samples and positive window slack')
    previous = np.broadcast_to(np.asarray(start_s, dtype=float), (count,)).copy()
    if not np.isfinite(previous).all():
        raise ValueError('start_s must be finite')
    arcs, projected = np.empty((count, steps)), np.empty_like(points)
    xy, s = reference.xy, reference.s
    vectors = np.diff(xy, axis=0)
    length = np.diff(s)
    squared = np.einsum('ij,ij->i', vectors, vectors)
    origin = reference.sample(previous)
    for tick in range(steps):
        position = points[:, tick]
        if len(xy) == 1:
            chosen_s = np.zeros(count)
            chosen_xy = np.broadcast_to(xy[0], position.shape).copy()
        else:
            travel = np.linalg.norm(position-(origin if tick == 0 else points[:, tick-1]), axis=1)
            window = 2*travel+slack_m
            lo, hi = np.maximum(0., previous-window), np.minimum(s[-1], previous+window)
            # A trajectory can start outside the first endpoint when reversing.
            hi = np.maximum(hi, 0.)
            offset = position[:, None, :]-xy[None, :-1, :]
            fraction = np.clip(np.einsum('nki,ki->nk', offset, vectors)/np.maximum(squared, 1e-30), 0., 1.)
            candidate = s[:-1]+fraction*length
            candidate = np.maximum(s[:-1], np.minimum(s[1:], np.clip(candidate, lo[:, None], hi[:, None])))
            valid = (candidate >= lo[:, None]-1e-10) & (candidate <= hi[:, None]+1e-10)
            fraction = (candidate-s[:-1])/np.maximum(length, 1e-30)
            projected_all = xy[None, :-1]+fraction[..., None]*vectors
            error = np.sum((projected_all-position[:, None])**2, axis=2)
            # Resolve equal-distance intersections using the prior branch. This
            # is deliberately much smaller than the spatial distance term.
            error += 1e-6*(candidate-previous[:, None])**2
            error[~valid] = np.inf
            selected = np.argmin(error, axis=1)
            chosen_s = candidate[np.arange(count), selected]
            chosen_xy = projected_all[np.arange(count), selected]
        first_tangent = np.array([np.cos(reference.yaw[0]), np.sin(reference.yaw[0])])
        behind = (position-xy[0]) @ first_tangent
        chosen_s = np.where((chosen_s <= 1e-9) & (behind < 0), behind, chosen_s)
        arcs[:, tick], projected[:, tick] = chosen_s, chosen_xy
        previous = chosen_s
    def sample(values):
        return np.interp(arcs, s, values)
    yaw = sample(reference.yaw)
    delta = points-projected
    lateral = -np.sin(yaw)*delta[..., 0]+np.cos(yaw)*delta[..., 1]
    along = np.cos(yaw)*delta[..., 0]+np.sin(yaw)*delta[..., 1]
    excess = np.where(arcs >= s[-1]-1e-9, np.maximum(along, 0.), 0.)
    return ProjectionBatch(arcs, projected, lateral, yaw, sample(reference.curvature),
                           sample(reference.tube_left), sample(reference.tube_right), excess)


@dataclass(frozen=True)
class SpatialScores:
    costs: np.ndarray
    terms: dict[str, np.ndarray]
    projection: ProjectionBatch


def trajectory_scores(reference, poses, speeds, times, config, *, desired_speed,
                      search=None, previous_command=None, command_scale=None,
                      speed_reference=None, valid_time_mask=None, start_s=0.):
    """Geometric terms integrate over positive arc travel; speed over time.

    ``poses`` and ``speeds`` are the measured-state plant's predicted hold phase,
    not a constant-speed synthetic rollout. Brake tails remain the responsibility
    of the shared physical evaluator. ``previous_command`` is explicitly supplied
    execution feedback, not an internally assumed previous winning candidate.
    """
    poses, speeds, times = map(lambda x: np.asarray(x, dtype=float), (poses, speeds, times))
    if (poses.ndim != 3 or poses.shape[2] != 3 or speeds.shape != poses.shape[:2] or
            times.shape != (poses.shape[1],) or not len(times) or np.any(np.diff(times) <= 0) or
            not all(np.isfinite(x).all() for x in (poses, speeds, times))):
        raise ValueError('scores require finite aligned poses (N,T,3), speeds (N,T), increasing times (T,)')
    count, steps = speeds.shape
    valid = np.ones_like(speeds, dtype=bool) if valid_time_mask is None else np.asarray(valid_time_mask, dtype=bool)
    if valid.shape != speeds.shape or not valid[:, 0].all() or np.any(np.diff(valid.astype(int), axis=1) > 0):
        raise ValueError('valid samples must be a nonempty contiguous prefix per candidate')
    projection = project_trajectories(reference, poses[..., :2], start_s=start_s)
    differences = np.diff(projection.s, axis=1)
    advance = np.maximum(differences, 0.)*valid[:, 1:]
    retreat = np.maximum(-differences, 0.)*valid[:, 1:]
    dt = np.diff(times)[None]*valid[:, 1:]
    end = valid.sum(axis=1)-1
    rows = np.arange(count)

    def integral(values, increments):
        return np.sum((values[:, :-1]+values[:, 1:])*.5*increments, axis=1)

    heading = poses[..., 2]-projection.yaw-(np.pi if reference.gear < 0 else 0.)
    outside = np.maximum.reduce((projection.lateral-projection.tube_left,
                                 -projection.lateral-projection.tube_right, np.zeros_like(speeds)))
    if speed_reference is None:
        vref = np.broadcast_to(desired_speed, speeds.shape)
        if not np.isfinite(vref).all():
            raise ValueError('desired_speed must be finite')
    else:
        speed_reference = np.asarray(speed_reference, dtype=float)
        if speed_reference.shape != reference.s.shape or not np.isfinite(speed_reference).all():
            raise ValueError('speed reference must be finite and aligned with the EB arc samples')
        vref = np.interp(projection.s, reference.s, speed_reference)
    terms = {
        'cross_track': config.cross_track_weight*integral(projection.lateral**2, advance),
        'heading': config.heading_weight*integral(1-np.cos(heading), advance),
        'tube': config.tube_weight*integral(outside**2, advance),
        'progress': -config.progress_weight*(projection.s[rows, end]-projection.s[:, 0]),
        'backward': config.backward_weight*np.sum(retreat, axis=1),
        'speed': config.speed_weight*integral((speeds-vref)**2, dt),
        'endpoint': config.endpoint_weight*(projection.endpoint_excess[rows, end]**2 +
            np.where(projection.s[rows, end] >= reference.s[-1]-1e-9, projection.lateral[rows, end]**2, 0.)),
        'control': np.zeros(count),
    }
    if previous_command is not None:
        search, previous, scale = (np.asarray(x, dtype=float) for x in (search, previous_command, command_scale))
        if (search.ndim != 2 or len(search) != count or previous.shape != search.shape[1:] or
                scale.shape != previous.shape or np.any(scale <= 0) or
                not all(np.isfinite(x).all() for x in (search, previous, scale))):
            raise ValueError('control smoothing requires matching finite search, executed command and positive scales')
        terms['control'] = config.control_weight*np.sum(((search-previous)/scale)**2, axis=1)
    return SpatialScores(np.sum(list(terms.values()), axis=0), terms, projection)


def score_rollout(reference, prediction, codec, config, **kwargs):
    """Score only the shared predictor's hold phase; retain its measured dynamics."""
    end = prediction.prediction_end+1
    return trajectory_scores(reference, codec.poses(prediction.states[:, :end]),
                             codec.velocities(prediction.states[:, :end])[..., 0],
                             prediction.times[:end], config,
                             valid_time_mask=prediction.valid_time_mask[:, :end], **kwargs)
