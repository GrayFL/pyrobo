"""Response-aware command proposals using the authoritative track integrator.

Finite probes produce an empirical reachable-state estimate, NOT a certified
outer envelope. They shape sampling anchors but never reject a command/state.
Full model command bounds remain available, including saturation plateaus: a
reference need not be attained within dt to be useful over the prediction horizon.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(slots=True)
class ReachableEnvelope:
    lower: np.ndarray
    upper: np.ndarray
    native_probes: np.ndarray
    next_states: np.ndarray
    mode: str
    strict_outer_bound: bool = False


@dataclass(slots=True)
class CommandWindow:
    lower: np.ndarray
    upper: np.ndarray
    anchors: np.ndarray
    envelope: ReachableEnvelope | None = None
    fallback_reason: str = ''


class ReachabilityEstimator:
    def __init__(self, dynamics, command_space):
        self.dynamics, self.space = dynamics, command_space

    def state_envelope(self, packed_state, dt, mode='track', *, search=None):
        if search is None:
            lo, hi = self.space.bounds()
            current = np.clip(self.space.state_coordinates(packed_state), lo, hi)
            search = np.array(list(product(*[np.unique([a, b, c]) for a, b, c in zip(lo, current, hi)])))
        native, valid = self.space.to_native_batch(search)
        native = native[valid]
        if not len(native):
            raise ValueError('no model-valid envelope probes')
        states = np.broadcast_to(packed_state, (len(native), len(packed_state)))
        predicted = self.dynamics.integrate(states, native, dt, mode=mode)
        coordinates = self.space.state_coordinates(predicted)
        return ReachableEnvelope(coordinates.min(axis=0), coordinates.max(axis=0), native, predicted, mode)

    def command_window(self, state, request, dt, *, lower=None, upper=None, anchors=None):
        lo, hi = self.space.bounds()
        if lower is not None:
            lo = np.maximum(lo, lower)
        if upper is not None:
            hi = np.minimum(hi, upper)
        dim = len(lo)
        anchors = np.empty((0, dim)) if anchors is None else np.asarray(anchors).reshape(-1, dim)
        if np.any(lo > hi):
            return CommandWindow(lo, hi, anchors, fallback_reason='empty_policy_domain')
        current = np.clip(self.space.state_coordinates(state), lo, hi)
        probes = np.array(list(product(*[np.unique([a, b, c]) for a, b, c in zip(lo, current, hi)])))
        probes = np.unique(np.vstack((probes, np.clip(anchors, lo, hi))), axis=0)
        try:
            envelope = self.state_envelope(state, dt, 'track', search=probes)
            responses = self.space.state_coordinates(envelope.next_states)
            commands = self.space.from_native_batch(envelope.native_probes)
            # Retain references spanning actual track responses. These include
            # servo lag, jerk history, drag and coupled wheel/tire saturation.
            chosen = []
            for axis in range(dim):
                levels = np.linspace(envelope.lower[axis], envelope.upper[axis], 5)
                indices = np.argmin(abs(responses[:, axis, None] - levels), axis=0)
                chosen.append(commands[indices])
            anchors = np.vstack((anchors, current, *chosen))
            return CommandWindow(lo, hi, anchors, envelope)
        except (ValueError, FloatingPointError, OverflowError) as error:
            # Do not turn an uncertain estimator into a hard safety filter.
            return CommandWindow(lo, hi, np.vstack((anchors, current)), fallback_reason=type(error).__name__)
