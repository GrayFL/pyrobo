"""A small different model verifies that the batch engine does not assume S=10/U=2."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from pyrobo import ChassisState, Bounds2D, PointCloud, Twist2D, pose2d
from pyrobo.core import MotionPolicy, MotionRequest, RobotObservation
from pyrobo.control.dwa import DWAConfig, DWAEngine
from pyrobo.control.dwa.policies import PolicyContext
from pyrobo.models.dynamics.diagnostics import BatchDiagnostics


@dataclass(frozen=True)
class TripleCommand:
    references: tuple
    mode: str = 'track'


class Codec:
    state_size = 7
    def pack(self, state):
        return np.array([*state.pose.translation[:2], 0., state.velocity.linear, state.velocity.angular, 0., 1.])
    def poses(self, states):
        return states[..., :3]
    def velocities(self, states):
        return states[..., 3:5]
    def stopped(self, states):
        return np.max(abs(states[..., 3:6]), axis=-1) < 1e-4


class ThreeAxisSpace:
    search_size = native_size = 3
    can_spin = True
    def __init__(self):
        self.model, self.decoded = self, 0
    def bounds(self):
        return np.full(3, -.6), np.full(3, .6)
    def to_native_batch(self, search):
        return search.copy(), np.all(abs(search) <= .6, axis=1)
    def from_native_batch(self, native):
        return native.copy()
    def state_coordinates(self, states):
        return states[..., 3:6]
    def desired_coordinates(self, request):
        return np.array([.4, .2, .1])
    def encode_command(self, command):
        return np.array(command.references)
    def decode_one(self, native, mode='track'):
        self.decoded += 1
        return TripleCommand(tuple(native), mode)


class ThreeAxisDynamics:
    codec = Codec()
    config = SimpleNamespace(substep=.05)
    def integrate(self, states, commands, dt, *, mode='track', diagnostics=False):
        assert states.shape == (len(commands), 7) and commands.shape[1] == 3
        result = states.copy()
        result[:, 3:6] = commands if mode == 'track' else 0.
        result[:, 0] += (states[:, 3] + result[:, 3]) * dt/2
        result[:, 2] += (states[:, 4] + result[:, 4]) * dt/2
        return (result, BatchDiagnostics.empty(len(commands))) if diagnostics else result


class ThreeAxisPolicy:
    def prepare(self, observation, request, space, config, codec, previous):
        low, high = space.bounds()
        desired = space.desired_coordinates(request)
        return PolicyContext(request, codec.poses(codec.pack(observation.chassis)), low, high,
                             desired[None], desired, previous=previous)
    def score_batch(self, context, search, final, config, space):
        return np.sum((search-context.desired)**2, axis=1), np.ones(len(search), dtype=bool)


def test_different_command_and_state_dimensions_keep_rows_aligned_and_decode_once():
    space, policy = ThreeAxisSpace(), ThreeAxisPolicy()
    engine = DWAEngine(DWAConfig(linear_samples=3, angular_samples=3, batch_size=7), model=space,
                       dynamics=ThreeAxisDynamics(), command_space=space,
                       policies={MotionPolicy.TELEOP_AVOID: policy})
    engine.reset()
    assert engine.policies[MotionPolicy.TELEOP_AVOID] is policy
    o = RobotObservation(ChassisState(pose2d(0, 0)), PointCloud(np.empty((0, 2))), Bounds2D(-4, 4, -4, 4))
    request = MotionRequest(MotionPolicy.TELEOP_AVOID, 0., desired=Twist2D(.4, .2))
    search = np.array([[.4, .2, .1], [1., 0., 0.], [.1, -.3, .5]])
    batch = engine.evaluate_candidates(o, request, search)
    assert space.decoded == 0
    assert batch.prediction.states.shape[0] == 3 and batch.prediction.states.shape[-1] == 7
    np.testing.assert_array_equal(batch.native_commands, search)
    assert batch.constraint_masks['command_valid'].tolist() == [True, False, True]
    assert np.isinf(batch.costs[1]) and batch.feasible.tolist() == [True, False, True]
    result = engine.optimize(o, request)
    assert result.safe and result.command.references == (.4, .2, .1)
    assert result.trajectory.states.shape[1] == 7 and space.decoded == 1
