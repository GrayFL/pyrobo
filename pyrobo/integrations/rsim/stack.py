"""Reusable graph composition; sources and the actuator remain caller-owned."""
from rsim import ProcessPlacement

from .components import Execution, GlobalPlanning, LocalNavigation
from .config import StackConfig


class NavigationStack:
    def __init__(self, states, maps, missions, config=None, *, local_options=None, global_options=None):
        self.config = config or StackConfig()
        self.local = LocalNavigation(states, maps, missions, self.config, **(local_options or {}))
        self.global_planning = GlobalPlanning(self.local.requests, maps, self.config, **(global_options or {}))
        self.local.connect_routes(self.global_planning.routes)
        self.execution = Execution(self.local.proposals, states, maps, missions, self.config)
        self.commands = self.execution.commands
        self.diagnostics = self.local.diagnostics

    def placement(self, prefix='navigation'):
        """Three distinct workers; the caller may also supply another placement."""
        return {self.global_planning: ProcessPlacement(f'{prefix}_global'),
                self.local: ProcessPlacement(f'{prefix}_local'),
                self.execution: ProcessPlacement(f'{prefix}_execution')}
