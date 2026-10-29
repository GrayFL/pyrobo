"""Explicit dictionaries and arrays for RSim's public shared-memory transport.

Pose objects stay inside each process. State layout belongs to StateCodec, not
to the wire adapter. Point clouds are materialized once per map revision here.
"""
import numpy as np

from pyrobo import Bounds2D, ChassisState, PointCloud, RobotObservation, pose2d
from pyrobo.runtime.leases import CommandProposal

CLOCK = 'host:monotonic'


def latest(signal):
    frames = signal.frames
    return frames[-1] if frames else None


class ObservationReader:
    def __init__(self, plant):
        self.plant = plant
        self.revision = None
        self._map_key = None
        self.cloud = self.bounds = None

    def map(self, frame):
        if frame.clock != CLOCK:
            raise ValueError('navigation requires an explicit host monotonic clock transform')
        data = frame.data
        revision = data['revision']
        stamp = None if data.get('static', False) else frame.stamp_ns/1e9
        key = (revision, stamp)
        if key != self._map_key:
            self.cloud = PointCloud(data['points'], frame=data['frame'], timestamp=stamp)
            self.bounds = Bounds2D(*data['bounds'])
            self.revision, self._map_key = revision, key
        return self.cloud, self.bounds

    def observation(self, state_frame, map_frame, dt, *, now=None):
        cloud, bounds = self.map(map_frame)
        if state_frame.clock != CLOCK or state_frame.data['frame'] != cloud.frame:
            raise ValueError('state/map clock or spatial frame mismatch')
        stamp = state_frame.stamp_ns/1e9
        source = ChassisState(pose2d(0, 0, frame=cloud.frame), timestamp=stamp)
        state = self.plant.codec.unpack(np.asarray(state_frame.data['state']), source, 0.)
        return RobotObservation(state, cloud, bounds, dt, timestamp=stamp if now is None else now)


def proposal_packet(proposal, model):
    return {'native': model.encode_command(proposal.command), 'mode': proposal.command.mode,
            'issued_at': proposal.issued_at, 'valid_until': proposal.valid_until,
            'sequence': proposal.sequence, 'task_epoch': proposal.task_epoch,
            'map_revision': proposal.map_revision}


def decode_proposal(data, space):
    return CommandProposal(space.decode_one(data['native'], data['mode']), data['issued_at'],
                           data['valid_until'], data['sequence'], data['task_epoch'], data['map_revision'])
