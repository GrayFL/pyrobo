"""Shared batched command prediction and physically integrated braking tails.

Used by optimization and independent final command validation. This module has
no dependency on a planning algorithm, navigation task or runtime authority.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class RolloutBatch:
    states: np.ndarray              # (N,T,S)
    times: np.ndarray               # relative seconds, (T,)
    sweep: np.ndarray               # half interval corner travel, (N,T)
    prediction_end: int
    stopped: np.ndarray             # (N,)
    valid_time_mask: np.ndarray     # (N,T), false only after each row stops
    constraint_flags: dict[str, np.ndarray]


class BatchPredictor:
    def __init__(self, dynamics, footprint, *, prediction_time=1.8, recording_dt=.08, braking_timeout=4.):
        self.dynamics, self.codec, self.footprint = dynamics, dynamics.codec, footprint
        self.prediction_time, self.recording_dt, self.braking_timeout = prediction_time, recording_dt, braking_timeout

    def predict(self, state, commands, dt, *, stop_only=False, mode='track', diagnostics=False):
        count = len(commands)
        values = np.broadcast_to(state, (count, len(state))).copy()
        trajectory, sweep, times = [values.copy()], [np.zeros(count)], [0.]
        valid = [np.ones(count, dtype=bool)]
        active = np.ones(count, dtype=bool)
        masks = {}

        def phase(duration, phase_mode, tail=False):
            nonlocal values, active
            if duration <= 1e-12:
                return
            steps = max(1, int(np.ceil(duration / self.dynamics.config.substep - 1e-10)))
            h = duration / steps
            stride = max(1, int(np.floor(self.recording_dt / h + 1e-10)))
            travel = np.zeros(count)
            recorded_time = 0.
            for i in range(steps):
                previous = values
                if diagnostics:
                    values, report = self.dynamics.integrate(values, commands, h, mode=phase_mode, diagnostics=True)
                    for key, mask in report.flags.items():
                        masks[key] = masks.get(key, np.zeros(count, dtype=bool)) | mask
                else:
                    values = self.dynamics.integrate(values, commands, h, mode=phase_mode)
                mid = (self.codec.velocities(previous) + self.codec.velocities(values)) / 2
                travel += (np.abs(mid[:, 0]) + self.footprint.radius * np.abs(mid[:, 1])) * h
                recorded_time += h
                if (i + 1) % stride == 0 or i == steps - 1:
                    trajectory.append(values.copy())
                    sweep.append(travel / 2)
                    times.append(times[-1] + recorded_time)
                    valid.append(active.copy() if tail else np.ones(count, dtype=bool))
                    if tail:
                        active &= ~self.codec.stopped(values)
                    travel, recorded_time = np.zeros(count), 0.

        phase(dt, mode)
        if not stop_only and mode == 'track':
            phase(max(0., self.prediction_time - dt), mode)
        endpoint = len(trajectory) - 1
        active = ~self.codec.stopped(values)
        elapsed = 0.
        while active.any() and elapsed < self.braking_timeout - 1e-12:
            h = min(self.recording_dt, self.braking_timeout - elapsed)
            phase(h, 'estop', tail=True)
            elapsed += h
        return RolloutBatch(np.stack(trajectory, axis=1), np.array(times), np.stack(sweep, axis=1), endpoint,
                            self.codec.stopped(values), np.stack(valid, axis=1), masks)
