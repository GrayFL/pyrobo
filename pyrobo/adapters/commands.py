"""Explicit external command capabilities, without a ROS dependency."""
import numpy as np

from pyrobo.core.state import Twist2D


class TwistCommandAdapter:
    def __init__(self, command_space):
        self.space = command_space

    def convert(self, command):
        if command.mode != 'track':
            raise ValueError('Twist cannot express coast/soft_stop/estop semantics; use a native command interface')
        raw = self.space.model.encode_command(command)[None]
        search = self.space.from_native_batch(raw)
        _, valid = self.space.to_native_batch(search)
        if not valid[0]:
            raise ValueError('command is outside device/model capability')
        speed = float(search[0, 0])
        if not self.space.can_spin and abs(speed) < 1e-12 and abs(search[0, 1]) > 1e-12:
            raise ValueError('Twist cannot express independent Ackermann steering at rest')
        return Twist2D(speed, float(self.space.yaw_reference(search)[0]))
