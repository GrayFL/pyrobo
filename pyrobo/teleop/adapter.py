"""Authorized input becomes a timed desire; it never overwrites actuator output."""
import numpy as np

from ..core import MotionPolicy, MotionRequest


class TeleopAdapter:
    def __init__(self, command_space, *, policy=MotionPolicy.TELEOP_STOP, ttl=.3, decode_speed=.02,
                 max_turn_deviation=.8, max_lateral_deviation=.8, max_heading_deviation=1.2):
        self.space = command_space
        self.policy = MotionPolicy(policy)
        if self.policy not in (MotionPolicy.TELEOP_STOP, MotionPolicy.TELEOP_AVOID):
            raise ValueError('teleop policy must be stop or avoid')
        if not np.isfinite(decode_speed) or decode_speed <= 0:
            raise ValueError('decode_speed must be positive')
        self.ttl, self.decode_speed = ttl, decode_speed
        self.options = dict(max_turn_deviation=max_turn_deviation, max_lateral_deviation=max_lateral_deviation,
                            max_heading_deviation=max_heading_deviation)
        self._latest, self._last_steering = None, None

    def submit(self, desired, timestamp, *, steering=None, source='operator', policy=None):
        chosen_policy = MotionPolicy(policy or self.policy)
        if chosen_policy not in (MotionPolicy.TELEOP_STOP, MotionPolicy.TELEOP_AVOID):
            raise ValueError('manual source cannot submit a navigation or stop override')
        if self._latest is not None and timestamp < self._latest.timestamp:
            raise ValueError('teleop timestamps cannot go backwards')
        if not self.space.can_spin:
            if steering is None:
                if abs(desired.linear) < self.decode_speed:
                    if self._last_steering is None and abs(desired.angular) > 1e-12:
                        raise ValueError('Ackermann low-speed turn requires explicit or previous steering intent')
                    steering = self._last_steering or 0.
                else:
                    provisional = MotionRequest(chosen_policy, timestamp, desired=desired)
                    steering = float(self.space.desired_coordinates(provisional)[1])
            if not np.isfinite(steering):
                raise ValueError('steering must be finite')
            self._last_steering = steering
        request = MotionRequest(chosen_policy, timestamp, desired=desired, steering=steering, ttl=self.ttl,
                                source=source, **self.options)
        self._latest = request
        return request

    def latest_request(self, timestamp):
        return self._latest if self._latest is not None and self._latest.fresh(timestamp) else None

    def clear(self):
        self._latest = None
