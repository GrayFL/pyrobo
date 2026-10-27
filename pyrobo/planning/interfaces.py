from typing import Protocol


class GlobalPlanner(Protocol):
    def plan(self, start, goal, obstacles, bounds): ...
