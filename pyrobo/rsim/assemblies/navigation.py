"""Reusable graph composition; sources and the actuator remain caller-owned."""
from rsim import ProcessPlacement

from ..components import Execution, GlobalPlanning, LocalNavigation
from ..config import StackConfig
from ..scheduling import snapshot_from


class NavigationStack:
    def __init__(self, states, maps, missions, config=None, *, local_options=None, global_options=None,
                 tick_source=None):
        self.config = config or StackConfig()
        self.local = LocalNavigation(states, maps, missions, self.config,
                                     tick_source=tick_source, **(local_options or {}))
        self.global_planning = GlobalPlanning(self.local.requests, maps, self.config, **(global_options or {}))
        self.local.connect_routes(self.global_planning.routes)
        self.execution = Execution(self.local.proposals, states, maps, missions, self.config,
                                   tick_source=tick_source)
        self.commands = self.execution.commands
        self.diagnostics = self.local.diagnostics

    async def step_once(self, context, snapshot=None, *, actuator_horizon=None):
        """Compute one explicit navigation stage without autonomous tasks.

        The first frame can request a route and produce a safe stop. That route
        is consumed at the next new source frame, preserving Navigator's single
        observation per source timestamp. Command acceptance/physics advancement
        remain separate operations owned by the caller.
        """
        if self.config.scheduling != 'lockstep':
            raise RuntimeError('step_once requires scheduling="lockstep" to avoid concurrent algorithms')
        snapshot = snapshot or snapshot_from(self.local.states, self.local.maps, self.local.missions)
        if snapshot is None:
            raise RuntimeError('navigation source/map/mission snapshot is not ready')
        snapshot.require_context(context)
        await self.local.step_once(context, snapshot)
        await self.global_planning.step_once()
        return await self.execution.step_once(context, snapshot, actuator_horizon=actuator_horizon)

    def placement(self, prefix='navigation'):
        """Three distinct workers; the caller may also supply another placement."""
        if self.config.environment_policy == 'every':
            raise ValueError('every navigation requires co-located source and consumers; remote delivery is latest only')
        return {self.global_planning: ProcessPlacement(f'{prefix}_global'),
                self.local: ProcessPlacement(f'{prefix}_local'),
                self.execution: ProcessPlacement(f'{prefix}_execution')}
