"""Batch search-to-actuator mappings. Only the selected row becomes an object."""
import numpy as np
from ..dynamics.codec import ChassisCodec

from pyrobo.models.kinematics.ackermann import Ackermann
from pyrobo.models.kinematics.differential import DifferentialDrive


def matrix(values, width):
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != width or not np.isfinite(result).all():
        raise ValueError(f'expected a finite (N,{width}) matrix')
    return result


class DifferentialSpace:
    search_size = native_size = 2
    can_spin = True
    coordinate_names = ('speed_mps', 'yaw_rate_radps')

    def __init__(self, model):
        self.model = model

    def bounds(self):
        m = self.model
        return np.array([-m.max_reverse_speed, -m.max_yaw_rate]), np.array([m.max_speed, m.max_yaw_rate])

    def to_native_batch(self, search):
        values = matrix(search, 2)
        v, w = values.T
        native = np.column_stack((v - self.model.track * w/2, v + self.model.track * w/2))
        lo, hi = self.bounds()
        mask = np.all((values >= lo - 1e-10) & (values <= hi + 1e-10), axis=1)
        mask &= np.all(np.abs(native) <= self.model.max_wheel_speed + 1e-10, axis=1)
        return native, mask

    def from_native_batch(self, native):
        left, right = matrix(native, 2).T
        return np.column_stack(((left + right)/2, (right - left)/self.model.track))

    def state_coordinates(self, states):
        return ChassisCodec.differential_coordinates(states)

    def desired_coordinates(self, request):
        return np.array([request.desired.linear, request.desired.angular])

    def request_error_batch(self, search, native, request):
        desired = self.desired_coordinates(request)
        _, scale = self.bounds()
        return np.sum(((search - desired) / scale)**2, axis=1)

    def decode_one(self, native, mode='track'):
        return self.model.decode_command(native, mode)

    def yaw_reference(self, search):
        return np.asarray(search)[..., 1]

    def turn_for_heading(self, error, distance):
        return error

    def goal_approach_distance(self, lookahead):
        return 0.


class AckermannSpace(DifferentialSpace):
    can_spin = False
    coordinate_names = ('speed_mps', 'steering_rad')

    def bounds(self):
        m = self.model
        return np.array([-m.max_reverse_speed, -m.max_steering]), np.array([m.max_speed, m.max_steering])

    def to_native_batch(self, search):
        values = matrix(search, 2)
        lo, hi = self.bounds()
        mask = np.all((values >= lo - 1e-10) & (values <= hi + 1e-10), axis=1)
        mask &= np.abs(self.model.yaw_rate(values[:, 0], values[:, 1])) <= self.model.max_yaw_rate + 1e-10
        return values.copy(), mask

    def from_native_batch(self, native):
        return matrix(native, 2).copy()

    def state_coordinates(self, states):
        return ChassisCodec.ackermann_coordinates(states)

    def desired_coordinates(self, request):
        if request.steering is not None:
            return np.array([request.desired.linear, request.steering])
        command = self.model.from_twist(request.desired)
        return self.model.encode_command(command)

    def yaw_reference(self, search):
        values = np.asarray(search)
        return self.model.yaw_rate(values[..., 0], values[..., 1])

    def turn_for_heading(self, error, distance):
        return np.arctan2(2*self.model.wheelbase*np.sin(error), distance)

    def goal_approach_distance(self, lookahead):
        return max(lookahead, 1.5 * self.model.minimum_turn_radius)


def command_space_for(model):
    # Model selection belongs here, never in navigation or teleop policies.
    if isinstance(model, DifferentialDrive):
        return DifferentialSpace(model)
    if isinstance(model, Ackermann):
        return AckermannSpace(model)
    factory = getattr(model, 'command_space', None)
    if factory is not None:
        return factory()
    raise TypeError('supply a CommandSpace for this model')
