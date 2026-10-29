"""Explicit dictionaries and arrays for RSim's public shared-memory transport.

Pose objects stay inside each process. State layout belongs to StateCodec, not
to the wire adapter. Point clouds are materialized once per map revision here.
"""
import numpy as np
from time import monotonic_ns, time_ns

from pyrobo import Bounds2D, ChassisState, PointCloud, RobotObservation, pose2d
from pyrobo.runtime.leases import CommandProposal

CLOCK = 'host:monotonic'


def latest(signal):
    frames = signal.frames
    return frames[-1] if frames else None


class ObservationReader:
    def __init__(self, plant, *, clock=CLOCK):
        self.plant = plant
        self.clock = clock
        self.revision = None
        self._map_key = None
        self.cloud = self.bounds = None

    def map(self, frame):
        if frame.clock != self.clock:
            raise ValueError('navigation map clock does not match the configured environment clock')
        data = frame.data
        revision = data['revision']
        stamp = None if data.get('static', False) else frame.stamp_ns/1e9
        key = (revision, stamp, getattr(frame, 'metadata', {}).get('episode', 0))
        if key != self._map_key:
            self.cloud = PointCloud(data['points'], frame=data['frame'], timestamp=stamp)
            self.bounds = Bounds2D(*data['bounds'])
            self.revision, self._map_key = revision, key
        return self.cloud, self.bounds

    def observation(self, state_frame, map_frame, dt, *, now=None):
        cloud, bounds = self.map(map_frame)
        if state_frame.clock != self.clock or state_frame.data['frame'] != cloud.frame:
            raise ValueError('state/map clock or spatial frame mismatch')
        if not map_frame.data.get('static', False) and episode(state_frame) != episode(map_frame):
            raise ValueError('dynamic map and state episode mismatch')
        stamp = state_frame.stamp_ns/1e9
        source = ChassisState(pose2d(0, 0, frame=cloud.frame), timestamp=stamp)
        state = self.plant.codec.unpack(np.asarray(state_frame.data['state']), source, 0.)
        return RobotObservation(state, cloud, bounds, dt, timestamp=stamp if now is None else now)


def episode(frame):
    return getattr(frame, 'metadata', {}).get('episode', frame.data.get('episode', 0)
                                              if isinstance(frame.data, dict) else 0)


def source_session(frame):
    return getattr(frame, 'metadata', {}).get('session',
        getattr(getattr(frame, 'sample_id', None), 'producer_instance_id', ''))


def source_host_ns(frame):
    """Same-host source evidence, preserved through every conversion.

    Older producers expose only wall receipt time. Convert its age
    conservatively; callers retain the result with the proposal rather than
    treating a later receive/conversion as a new acquisition.
    """
    explicit = getattr(frame, 'metadata', {}).get('host_monotonic_ns')
    if explicit is not None:
        if not isinstance(explicit, int) or explicit < 0:
            raise ValueError('host_monotonic_ns must be a nonnegative integer')
        return explicit
    if frame.clock == CLOCK:
        return frame.stamp_ns
    return monotonic_ns() - max(0, time_ns() - frame.received_ns)


def proposal_packet(proposal, model):
    return {'native': model.encode_command(proposal.command), 'mode': proposal.command.mode,
            'issued_at': proposal.issued_at, 'valid_until': proposal.valid_until,
            'sequence': proposal.sequence, 'task_epoch': proposal.task_epoch,
            'map_revision': proposal.map_revision}


def decode_proposal(data, space):
    return CommandProposal(space.decode_one(data['native'], data['mode']), data['issued_at'],
                           data['valid_until'], data['sequence'], data['task_epoch'], data['map_revision'])
