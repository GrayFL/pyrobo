"""Global-route defects and safe rejoin intervals, in global arc coordinates."""
from dataclasses import dataclass, replace

import numpy as np

from ...core.geometry import wrap_angle
from .reference import interpolate


@dataclass(frozen=True)
class RouteInspection:
    s: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray
    clearance: np.ndarray
    triggers: dict[str, np.ndarray]
    conflicts: tuple[tuple[float, float], ...]
    active: tuple[tuple[float, float], ...]


def intervals(s, mask, padding=0., merge_distance=0.):
    ids = np.flatnonzero(mask)
    result = []
    for i in ids:
        lo, hi = max(s[0], s[max(0, i-1)]-padding), min(s[-1], s[min(i+1, len(s)-1)]+padding)
        if result and lo <= result[-1][1]+merge_distance:
            result[-1] = (result[-1][0], max(hi, result[-1][1]))
        else:
            result.append((float(lo), float(hi)))
    return tuple(result)


def inspect_route(xy, s, world, footprint, bounds, config, *, gear=1, max_curvature=np.inf):
    """Inspect complete edges and corner rotation envelopes, not just vertices."""
    delta = np.diff(xy, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    headings = np.arctan2(delta[:, 1], delta[:, 0])
    yaw = np.r_[headings, headings[-1]]
    gap = world.clearances(np.column_stack((xy, yaw+(np.pi if gear < 0 else 0.))), footprint, bounds)
    # Reversing a rectangle is equivalent to negating its longitudinal offset
    # when using the shared segment-aligned translation/rotation predicates.
    body = replace(footprint, offset_x=footprint.offset_x*gear,
                   margin=footprint.margin+config.minimum_clearance)
    collision = gap <= config.minimum_clearance
    for i in range(len(delta)):
        if not world.rectangle_segment_free(xy[i], xy[i+1], body, bounds):
            collision[i:i+2] = True
        if i and not world.rotation_free(xy[i], headings[i-1], headings[i], body, bounds):
            collision[i] = True
    angle = np.r_[0., wrap_angle(np.diff(headings)), 0.]
    scale = np.r_[max(lengths[0], 1e-9), (lengths[:-1]+lengths[1:])/2, max(lengths[-1], 1e-9)]
    curvature = angle/np.maximum(scale, 1e-9)
    rate = np.r_[0., abs(np.diff(curvature))/np.maximum(np.diff(s), 1e-9)]
    zigzag = np.zeros(len(s), dtype=bool)
    zigzag[1:] = (angle[:-1]*angle[1:] < 0) & (np.maximum(abs(angle[:-1]), abs(angle[1:])) > .1)
    triggers = {
        'collision': collision,
        'insufficient_clearance': gap < config.preferred_clearance,
        'sharp_corner': abs(angle) > config.sharp_corner_angle,
        'curvature_violation': abs(curvature) > min(max_curvature, config.preferred_curvature),
        'curvature_discontinuity': rate > config.preferred_curvature_rate,
        'zigzag': zigzag,
    }
    active = intervals(s, np.logical_or.reduce(list(triggers.values())), config.active_padding_m)
    return RouteInspection(s, xy, yaw, gap, triggers,
                           intervals(s, collision, merge_distance=config.active_padding_m), active)


def find_rejoin(inspection, after, world, footprint, bounds, config, *, gear=1, max_curvature=np.inf):
    """Require a whole safe tail and a compatible entry tangent after conflicts."""
    s, xy = inspection.s, inspection.xy
    body = replace(footprint, offset_x=footprint.offset_x*gear,
                   margin=footprint.margin+config.minimum_clearance)
    for candidate in s[s >= after-1e-9]:
        end = candidate+config.rejoin_validation_m
        if end > s[-1]+1e-9:
            break
        ids = (s > candidate) & (s < end)
        tail = np.vstack((interpolate(xy, s, [candidate]), xy[ids], interpolate(xy, s, [end])))
        if not world.path_free(tail, body, bounds):
            continue
        if np.isfinite(max_curvature):
            delta = np.diff(tail, axis=0)
            length = np.linalg.norm(delta, axis=1)
            angles = wrap_angle(np.diff(np.arctan2(delta[:, 1], delta[:, 0])))
            curvature = abs(angles)/np.maximum((length[:-1]+length[1:])/2, 1e-9)
            if np.any(curvature > max_curvature):
                continue
        return float(candidate)
    return None
