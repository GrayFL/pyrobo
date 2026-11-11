"""Sparse SE(2) pose/time optimization, with only local graph dependencies.

The mathematical formulation follows Timed Elastic Band; this is an independent
NumPy/SciPy implementation. References and differences are in this directory's README.
"""
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ...core.geometry import wrap_angle


def segment_motion(poses, intervals):
    """Signed exact-arc speed, yaw rate and lateral nonholonomic residual."""
    angle = wrap_angle(np.diff(poses[:, 2]))
    mid = poses[:-1, 2] + .5*angle
    delta = np.diff(poses[:, :2], axis=0)
    c, s = np.cos(mid), np.sin(mid)
    along = delta[:, 0]*c + delta[:, 1]*s
    lateral = delta[:, 1]*c - delta[:, 0]*s
    distance = along/np.sinc(angle/(2*np.pi))
    return distance/intervals, angle/intervals, lateral


@dataclass(frozen=True)
class Band:
    poses: np.ndarray
    intervals: np.ndarray
    cost: float
    initial_cost: float
    evaluations: int
    iterations: int
    status: str
    warm_started: bool

    @property
    def times(self):
        return np.r_[0., np.cumsum(self.intervals)]


class _BudgetExpired(Exception):
    pass


@lru_cache(maxsize=24)
def sparsity(count, car):
    """Fixed local stencil, independent of obstacle count and active penalties."""
    n = count-1
    # Node obstacle, segment NH/v/w/turn/time/mid-obstacle,
    # acceleration (measured start + optional stopped goal), steering-rate residuals.
    rows = count + 6*n + 2*(n+1) + (n if car else 0)
    pattern = lil_matrix((rows, 3*(count-2)+n), dtype=int)
    def mark(row, size, nodes=(), times=()):
        cols = [3*(j-1)+k for j in nodes if 0 < j < count-1 for k in range(3)]
        cols += [3*(count-2)+j for j in times if 0 <= j < n]
        pattern[row:row+size, cols] = 1
    for j in range(count):
        mark(j, 1, (j,))
    row = count
    for j in range(n):
        mark(row+6*j, 6, (j, j+1), (j,))
    row += 6*n
    for j in range(n):
        mark(row+2*j, 2, (j-1, j, j+1), (j-1, j))
    mark(row+2*n, 2, (n-1, n), (n-1,))
    if car:
        row += 2*(n+1)
        for j in range(n):
            mark(row+j, 1, (j-1, j, j+1), (j-1, j))
    return pattern.tocsr()


def solve_band(seed, intervals, observation, world, footprint, space, dynamics, config,
               *, goal_velocity=None):
    """Optimize interior poses and positive dt; endpoints remain fixed.

    Obstacles use the complete current point index and the actual rectangle,
    including offset/margin. Soft penalties guide optimization; execution is
    independently certified by the finite-force rollout and braking checks.
    """
    count = len(seed)
    size = 3*(count-2)
    car = not space.can_spin
    lo, hi = space.bounds()
    reverse = lo[0] if config.allow_reverse else 0.
    wmax = space.model.max_yaw_rate
    if not car:
        wmax = min(wmax, hi[1], -lo[1])
    curvature = np.tan(min(hi[1], -lo[1]))/space.model.wheelbase if car else 0.
    weights = {name: np.sqrt(getattr(config, name+'_weight')) for name in
               ('time', 'obstacle', 'kinematics', 'velocity', 'acceleration', 'turning')}
    started = perf_counter()
    evaluations = 0
    best_x = np.r_[seed[1:-1].ravel(), intervals]
    best_cost = np.inf

    def unpack(x):
        poses = np.vstack((seed[0], x[:size].reshape(-1, 3), seed[-1]))
        return poses, x[size:]

    def residual(x):
        nonlocal evaluations, best_cost, best_x
        if evaluations and config.time_budget is not None and perf_counter()-started >= config.time_budget:
            raise _BudgetExpired
        evaluations += 1
        poses, dt = unpack(x)
        v, w, lateral = segment_motion(poses, dt)
        middle = (poses[:-1]+poses[1:])*.5
        middle[:, 2] = poses[:-1, 2] + .5*wrap_angle(np.diff(poses[:, 2]))
        clear = world.clearances(np.vstack((poses, middle)), footprint, observation.bounds)
        obstacle = weights['obstacle']*np.maximum(config.obstacle_clearance-clear, 0.)
        node = obstacle[:count]
        speed_error = np.maximum(v-hi[0], 0.) + np.minimum(v-reverse, 0.)
        yaw_error = np.maximum(np.abs(w)-wmax, 0.)
        if car:
            turn = weights['turning']*np.maximum(np.abs(w)-np.abs(v)*curvature, 0.)
        else:
            wheel = np.abs(v) + .5*space.model.track*np.abs(w)
            turn = weights['velocity']*np.maximum(wheel-space.model.max_wheel_speed, 0.)
        segment = np.column_stack((weights['kinematics']*lateral,
            weights['velocity']*speed_error, weights['velocity']*yaw_error, turn,
            weights['time']*dt, obstacle[count:]))
        # Velocities are centered on segments, including the observed start.
        elapsed = np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2]
        a = np.diff(np.r_[observation.chassis.velocity.linear, v])/elapsed
        alpha = np.diff(np.r_[observation.chassis.velocity.angular, w])/elapsed
        acceleration = weights['acceleration']*np.column_stack((
            np.maximum(np.abs(a)-dynamics.drive_acceleration, 0.),
            np.maximum(np.abs(alpha)-dynamics.yaw_acceleration, 0.)))
        # Standard endpoint acceleration edge; free velocity at a local horizon.
        final_a = (np.array(goal_velocity)-np.array([v[-1], w[-1]]))/(dt[-1]/2) if goal_velocity is not None else np.zeros(2)
        final_error = weights['acceleration']*np.maximum(np.abs(final_a)-
            [dynamics.drive_acceleration, dynamics.yaw_acceleration], 0.)
        parts = [node.ravel(), segment.ravel(), acceleration.ravel(), final_error]
        if car:
            # atan of signed curvature keeps reverse steering consistent.
            steering = np.arctan(space.model.wheelbase*w*v/(v*v+1e-8))
            rate = np.diff(np.r_[observation.chassis.steering, steering])/elapsed
            parts.append(weights['turning']*np.maximum(np.abs(rate)-space.model.max_steering_rate, 0.))
        result = np.concatenate(parts)
        cost = float(result @ result)
        if np.isfinite(cost) and cost < best_cost:
            best_cost, best_x = cost, x.copy()
        return result

    x0 = best_x.copy()
    residual(x0)
    initial_cost = best_cost
    lower = np.r_[np.full(size, -np.inf), np.full(count-1, config.dt_min)]
    upper = np.r_[np.full(size, np.inf), np.full(count-1, config.dt_max)]
    iterations, status = 0, 'iteration_limit'
    try:
        result = least_squares(residual, x0, bounds=(lower, upper),
            jac_sparsity=sparsity(count, car), max_nfev=config.max_iterations,
            ftol=1e-3, xtol=1e-3, gtol=1e-3, tr_solver='lsmr')
        iterations = result.nfev
        status = 'converged' if result.success else 'iteration_limit'
    except _BudgetExpired:
        status = 'time_budget'
    poses, dt = unpack(best_x)
    poses.setflags(write=False)
    dt = dt.copy()
    dt.setflags(write=False)
    return Band(poses, dt, best_cost, initial_cost, evaluations, iterations, status, False)
