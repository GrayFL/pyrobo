"""Task-level settings independent of a particular local optimizer."""
from dataclasses import dataclass
from pyrobo.core.geometry import positive


@dataclass(frozen=True)
class NavigationConfig:
    lookahead: float = 1.3
    goal_tolerance: float = .18
    yaw_tolerance: float = .1

    def __post_init__(self):
        for name, value in vars(self).items():
            positive(value, name)
