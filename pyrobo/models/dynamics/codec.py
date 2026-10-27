"""Single authority for the current chassis array layout; other codecs may differ."""
from enum import IntEnum
import numpy as np


class StateColumn(IntEnum):
    X = 0
    Y = 1
    YAW = 2
    SPEED = 3
    YAW_RATE = 4
    STEERING = 5
    STEERING_RATE = 6
    ACCELERATION = 7
    YAW_ACCELERATION = 8
    STEERING_ACCELERATION = 9


class ChassisCodec:
    state_size = 10

    def __init__(self, plant):
        self.plant = plant

    def pack(self, state):
        return self.plant.pack(state)

    def unpack(self, values, source, dt):
        return self.plant.unpack(values, source, dt)

    def poses(self, states):
        return np.asarray(states)[..., :3]

    def velocities(self, states):
        return np.asarray(states)[..., 3:5]

    @staticmethod
    def differential_coordinates(states):
        return np.asarray(states)[..., [StateColumn.SPEED, StateColumn.YAW_RATE]]

    @staticmethod
    def ackermann_coordinates(states):
        return np.asarray(states)[..., [StateColumn.SPEED, StateColumn.STEERING]]

    def stopped(self, states, epsilon=1e-4):
        return np.max(np.abs(self.velocities(states)), axis=-1) <= epsilon
