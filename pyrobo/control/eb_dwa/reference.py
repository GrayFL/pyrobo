"""C1 geometric references, certified using the shared rectangular distance query.

Curve parameters are not time and are not assumed to be arc length. Derivative
bounds certify the intervals between queries; the circumscribed radius is used
only to bound corner travel, never as the collision shape.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicHermiteSpline


def readonly(values):
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def polyline(xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('reference requires a finite nonempty (N,2) path')
    xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-9]]
    return xy, np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]


def interpolate(xy, s, query):
    return np.column_stack([np.interp(query, s, xy[:, axis]) for axis in range(2)])


class GeometricCurve:
    """Local Hermite stencils retain fixed endpoints and prescribed tangents."""
    def __init__(self, xy, parameter=None, *, start_direction=None, end_direction=None):
        xy = np.asarray(xy, dtype=float)
        if parameter is None:
            xy, parameter = polyline(xy)
        parameter = np.asarray(parameter, dtype=float)
        if len(xy) < 2 or parameter.shape != (len(xy),) or np.any(np.diff(parameter) <= 0):
            raise ValueError('a curve requires >= 2 points and strictly increasing parameters')
        delta = np.diff(xy, axis=0)/np.diff(parameter)[:, None]
        tangent = np.vstack((delta[0], (delta[:-1]+delta[1:])/2, delta[-1]))
        for index, direction in ((0, start_direction), (-1, end_direction)):
            if direction is not None:
                tangent[index] = np.array([np.cos(direction), np.sin(direction)])*max(np.linalg.norm(tangent[index]), .1)
        self.xy, self.parameter, self.tangent = map(readonly, (xy, parameter, tangent))
        self.spline = CubicHermiteSpline(parameter, xy, tangent)

    def evaluate(self, parameter, derivative=0):
        return self.spline(parameter, nu=derivative)

    def samples(self, spacing):
        counts = np.maximum(1, np.ceil(np.linalg.norm(np.diff(self.xy, axis=0), axis=1)/spacing).astype(int))
        return np.r_[np.concatenate([np.linspace(a, b, count, endpoint=False)
                                     for a, b, count in zip(self.parameter[:-1], self.parameter[1:], counts)]),
                     self.parameter[-1]]

    def interval_bounds(self, lo, hi):
        """Bound speed, yaw travel and curvature inside individual cubic spans.

        Acceleration is linear on each span; jerk is constant. The speed lower
        bound follows from the midpoint velocity and the acceleration bound.
        d cross(velocity, acceleration)/du = cross(velocity, jerk).
        """
        mid, half = (lo+hi)/2, (hi-lo)/2
        spans = np.clip(np.searchsorted(self.parameter, mid, side='right')-1, 0, len(self.xy)-2)
        c = self.spline.c[:, spans]
        u = mid-self.parameter[spans]
        v = 3*c[0]*u[:, None]**2+2*c[1]*u[:, None]+c[2]
        a = 6*c[0]*u[:, None]+2*c[1]
        jerk = 6*c[0]
        amax = np.maximum(np.linalg.norm(a-jerk*half[:, None], axis=1),
                          np.linalg.norm(a+jerk*half[:, None], axis=1))
        vmid = np.linalg.norm(v, axis=1)
        lower = np.maximum(vmid-amax*half, 1e-14)
        upper = vmid+amax*half
        numerator = abs(cross(v, a))+upper*np.linalg.norm(jerk, axis=1)*half
        return upper, amax/lower, numerator/lower**3, vmid, lower


@dataclass(frozen=True)
class CurveValidation:
    valid: bool
    reason: str
    clearance_m: float
    curvature_bound: float
    queries: int


def validate_curve(curve, world, footprint, bounds, *, gear=1, minimum_clearance=0.,
                   max_curvature=np.inf, depth=12, max_intervals=8192):
    """Prove the complete oriented curve or explicitly reject ambiguity."""
    lo, hi = curve.parameter[:-1].copy(), curve.parameter[1:].copy()
    minimum, curvature_bound, queries = np.inf, 0., 0
    yaw_offset = np.pi if gear < 0 else 0.
    for _ in range(depth+1):
        if len(lo) > max_intervals:
            return CurveValidation(False, 'validation_budget', minimum, curvature_bound, queries)
        mid = (lo+hi)/2
        parameter = np.concatenate((lo, hi, mid))
        xy, v, a = (curve.evaluate(parameter, i) for i in range(3))
        speed = np.linalg.norm(v, axis=1)
        if np.any(speed < 1e-9):
            return CurveValidation(False, 'curve_cusp', minimum, np.inf, queries)
        poses = np.column_stack((xy, np.arctan2(v[:, 1], v[:, 0])+yaw_offset))
        gap = world.clearances(poses, footprint, bounds)
        queries += len(poses)
        if np.any(gap <= minimum_clearance):
            return CurveValidation(False, 'collision', float(gap.min()), curvature_bound, queries)
        k = abs(cross(v, a))/speed**3
        if np.any(k > max_curvature+1e-8):
            return CurveValidation(False, 'curvature_violation', float(gap.min()), float(k.max()), queries)
        vmax, yaw_rate, kmax, _, vmin = curve.interval_bounds(lo, hi)
        sweep = (vmax+footprint.radius*yaw_rate)*(hi-lo)/2
        continuous_gap = np.minimum(gap[:len(lo)], gap[len(lo):2*len(lo)])-sweep
        proved = (continuous_gap > minimum_clearance) & (kmax <= max_curvature) & (vmin > 1e-9)
        if proved.any():
            minimum = min(minimum, float(continuous_gap[proved].min()))
            curvature_bound = max(curvature_bound, float(kmax[proved].max()))
        if proved.all():
            return CurveValidation(True, '', minimum, curvature_bound, queries)
        lo, hi, mid = lo[~proved], hi[~proved], mid[~proved]
        lo, hi = np.r_[lo, mid], np.r_[mid, hi]
    return CurveValidation(False, 'unresolved_sweep', minimum, curvature_bound, queries)


@dataclass(frozen=True)
class EBReference:
    xy: np.ndarray
    s: np.ndarray
    yaw: np.ndarray                 # geometric tangent, not reverse body heading
    curvature: np.ndarray
    tube_left: np.ndarray           # positive widths; positive lateral error is left
    tube_right: np.ndarray
    global_s: np.ndarray
    gear: int = 1
    curve: GeometricCurve | None = None

    def __post_init__(self):
        xy = np.asarray(self.xy)
        if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or self.gear not in (-1, 1):
            raise ValueError('invalid EB reference shape or gear')
        for name in ('xy', 's', 'yaw', 'curvature', 'tube_left', 'tube_right', 'global_s'):
            value = np.asarray(getattr(self, name), dtype=float)
            if not np.isfinite(value).all() or (name != 'xy' and value.shape != (len(xy),)):
                raise ValueError(f'invalid reference {name}')
            object.__setattr__(self, name, readonly(value))
        if np.any(np.diff(self.s) <= 0) or abs(self.s[0]) > 1e-8:
            raise ValueError('reference arc length must start at zero and strictly increase')
        if np.any(self.tube_left < 0) or np.any(self.tube_right < 0):
            raise ValueError('tube widths must be nonnegative')

    @classmethod
    def from_curve(cls, curve, world, footprint, bounds, config, *, gear=1):
        u = curve.samples(config.reference_spacing_m)
        xy, first, second = (curve.evaluate(u, d) for d in range(3))
        length = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        keep = np.r_[True, length > 1e-10]
        xy, first, second, u = xy[keep], first[keep], second[keep], u[keep]
        speed = np.maximum(np.linalg.norm(first, axis=1), 1e-12)
        yaw = np.unwrap(np.arctan2(first[:, 1], first[:, 0]))
        poses = np.column_stack((xy, yaw+(np.pi if gear < 0 else 0.)))
        clearance = world.clearances(poses, footprint, bounds)
        # Distance is 1-Lipschitz under translation. This conservative symmetric
        # tube is a soft scoring allowance; it never replaces rollout collision.
        width = np.clip(clearance-config.minimum_clearance, 0., config.max_tube_width_m)
        s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        return cls(xy, s, yaw, cross(first, second)/speed**3, width, width, u, gear, curve)

    def sample(self, distance):
        distance = np.atleast_1d(distance)
        return interpolate(self.xy, self.s, distance)
