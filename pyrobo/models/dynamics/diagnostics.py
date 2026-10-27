"""Optional row-aligned diagnostics. Limiting flags are not automatic rejections."""
from dataclasses import dataclass
import numpy as np

FLAGS = ('reference_limited', 'effort_limited', 'steering_stop_reaction',
         'speed_hard_projection', 'lateral_envelope_exceeded',
         'linear_jerk_relaxed', 'angular_jerk_relaxed', 'tire_envelope_exceeded')


@dataclass(slots=True)
class BatchDiagnostics:
    flags: dict[str, np.ndarray]
    force: np.ndarray
    torque: np.ndarray

    @classmethod
    def empty(cls, count):
        return cls({name: np.zeros(count, dtype=bool) for name in FLAGS}, np.zeros(count), np.zeros(count))

    def merge(self, masks):
        for name, value in masks.items():
            self.flags[name] |= value
