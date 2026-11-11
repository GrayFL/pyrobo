"""Private DWA candidate feasibility; not a contract for other algorithms."""
import numpy as np

from ...geometry.sweep import swept_clearances


def evaluate_commands(predictor, state, native, dt, world, bounds):
    """DWA holds each candidate over its horizon, then checks braking.

    Also check braking immediately after the next control cycle. These are
    DWA's sampling semantics, not requirements imposed on trajectory solvers.
    """
    predicted = predictor.predict(state, native, dt)
    immediate = predictor.predict(state, native, dt, stop_only=True)
    full_clearance = swept_clearances(predictor.codec.poses(predicted.states), predicted.sweep,
        world, predictor.footprint, bounds, predicted.valid_time_mask)
    immediate_clearance = swept_clearances(predictor.codec.poses(immediate.states), immediate.sweep,
        world, predictor.footprint, bounds, immediate.valid_time_mask)
    clearance = np.minimum(full_clearance, immediate_clearance)
    feasible = (np.isfinite(clearance) & (clearance > 0.) & predicted.stopped & immediate.stopped &
                np.isfinite(predicted.states).all(axis=(1, 2)) & np.isfinite(immediate.states).all(axis=(1, 2)))
    return predicted, clearance, feasible
