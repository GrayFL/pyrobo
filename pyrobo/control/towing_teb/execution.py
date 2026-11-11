"""Private TEB check of the next command, independent of band optimization."""
import numpy as np

from ...geometry.sweep import swept_clearances


def check_command(predictor, state, native, dt, world, bounds):
    """Execute the extracted command for dt, then check a physical brake.

    Future TEB segments can have different speeds/curvatures. Holding the first
    command for the whole band duration would check a different trajectory.
    The optimized band remains a reference, not a predicted physical rollout.
    """
    batch = predictor.predict(state, native[None], dt, stop_only=True)
    clearance = float(swept_clearances(predictor.codec.poses(batch.states), batch.sweep,
        world, predictor.footprint, bounds, batch.valid_time_mask)[0])
    verified = bool(batch.stopped[0] and np.isfinite(batch.states).all() and
                    np.isfinite(clearance) and clearance > 0.)
    return batch, clearance, verified
