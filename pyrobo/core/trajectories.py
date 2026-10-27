"""Selected, timestamped trajectory at a public API boundary (never one per candidate)."""
from dataclasses import dataclass

import numpy as np
from graphmap.pose import PoseArray
from .paths import Path


@dataclass(frozen=True)
class Trajectory:
    poses: PoseArray
    times: np.ndarray
    start_time: float = 0.
    states: np.ndarray | None = None
    prediction_end: int = 0

    def __post_init__(self):
        Path(self.poses)  # shared public frame/planarity validation
        times = np.array(self.times, dtype=float, copy=True)
        if (times.shape != (len(self.poses.position),) or not np.isfinite(times).all()
                or not np.isfinite(self.start_time) or np.any(times < 0) or np.any(np.diff(times) <= 0)):
            raise ValueError('trajectory times must match poses and strictly increase from t>=0')
        if not 0 <= self.prediction_end < max(1, len(times)):
            raise ValueError('invalid prediction endpoint')
        times.setflags(write=False)
        object.__setattr__(self, 'times', times)
        if self.states is not None:
            states = np.array(self.states, dtype=float, copy=True)
            if states.ndim != 2 or len(states) != len(times) or not np.isfinite(states).all():
                raise ValueError('trajectory states must be a finite (T,S) array')
            states.setflags(write=False)
            object.__setattr__(self, 'states', states)

    @property
    def timestamps(self):
        return self.start_time + self.times
