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


def evaluate_safety(predictor, state, native, dt, world, bounds, *, diagnostics=False):
    predicted = predictor.predict(state, native, dt, diagnostics=diagnostics)
    immediate = predictor.predict(state, native, dt, stop_only=True)
    def clearance(batch):
        return swept_clearances(predictor.codec.poses(batch.states), batch.sweep, world,
                                predictor.footprint, bounds, batch.valid_time_mask)
    clear = np.minimum(clearance(predicted), clearance(immediate))
    masks = {'collision_free': clear > 0., 'stop_reachable': predicted.stopped & immediate.stopped,
             'model_finite': np.isfinite(predicted.states).all(axis=(1, 2))}
    return predicted, clear, masks
