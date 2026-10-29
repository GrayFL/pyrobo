"""DWA feasibility combines full-horizon and one-cycle brake assessments."""
from dataclasses import dataclass
import numpy as np

from ...geometry.sweep import swept_clearances


@dataclass(slots=True)
class MotionBatch:
    search_commands: np.ndarray
    native_commands: np.ndarray
    prediction: object
    feasible: np.ndarray
    clearance: np.ndarray
    costs: np.ndarray
    constraint_masks: dict[str, np.ndarray]
    cost_terms: dict[str, np.ndarray]


def immediate_safety(predictor, state, native, dt, world, bounds):
    """Prove one executed cycle + braking before spending on a full rollout."""
    batch = predictor.predict(state, native, dt, stop_only=True)
    clearance = swept_clearances(predictor.codec.poses(batch.states), batch.sweep, world,
                                 predictor.footprint, bounds, batch.valid_time_mask)
    return clearance, batch.stopped, np.isfinite(batch.states).all(axis=(1, 2))


def evaluate_safety(predictor, state, native, dt, world, bounds, *, diagnostics=False, immediate_check=None):
    predicted = predictor.predict(state, native, dt, diagnostics=diagnostics)
    immediate_check = (immediate_check if immediate_check is not None else
                       immediate_safety(predictor, state, native, dt, world, bounds))
    clearance = swept_clearances(predictor.codec.poses(predicted.states), predicted.sweep, world,
                                 predictor.footprint, bounds, predicted.valid_time_mask)
    clear = np.minimum(clearance, immediate_check[0])
    masks = {'collision_free': clear > 0., 'stop_reachable': predicted.stopped & immediate_check[1],
             'model_finite': np.isfinite(predicted.states).all(axis=(1, 2)) & immediate_check[2]}
    return predicted, clear, masks
