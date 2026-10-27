from enum import StrEnum


class ControlMode(StrEnum):
    AUTO = 'auto'
    MANUAL = 'manual'
    PAUSED = 'paused'
    ESTOP = 'estop'
