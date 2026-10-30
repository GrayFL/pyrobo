"""Bounded search over a preferred extra clearance, never over body safety.

Concrete planner configurations retain their hard-clearance semantics. This
explicit composition treats their clearance as a preference and keeps a
separate hard floor, in addition to the unchanged footprint and point radius.
"""
from dataclasses import replace
from time import perf_counter

import numpy as np

from ..core.geometry import planar_state, positive


class AdaptiveClearancePlanner:
    def __init__(self, factory, config, footprint, *, minimum_clearance=0.):
        if config.time_budget is None:
            raise ValueError('adaptive clearance requires a wall time_budget; use a fixed-clearance planner for logical budgets')
        positive(minimum_clearance, 'minimum_clearance', zero=True)
        if minimum_clearance > config.clearance:
            raise ValueError('minimum clearance cannot exceed preferred clearance')
        self.factory, self.config, self.footprint = factory, config, footprint
        self.minimum_clearance = minimum_clearance
        self.reset()

    def reset(self):
        self._planners = {}

    @property
    def safety_footprint(self):
        return replace(self.footprint, margin=self.footprint.margin+self.minimum_clearance)

    def plan(self, start, goal, obstacles, bounds):
        started = perf_counter()
        deadline = started+self.config.time_budget
        endpoint_room = float(obstacles.clearances([planar_state(start), planar_state(goal)],
                                                   self.footprint, bounds).min())
        # Reserve half the endpoint's available space for tracking. This is an
        # extra preference only; all actual body/point margins remain unchanged.
        preferred = max(self.minimum_clearance, min(self.config.clearance, max(0., endpoint_room/2)))
        levels = list(dict.fromkeys([preferred, (preferred+self.minimum_clearance)/2, self.minimum_clearance]))
        attempts, total_extensions = [], 0
        for i, clearance in enumerate(levels):
            left = deadline-perf_counter()
            if left <= 0 and attempts:
                break
            share = (1. if len(levels) == 1 else .7 if i == 0 else .15)*self.config.time_budget
            budget = max(1e-6, min(left, share))
            key = float(clearance)
            cfg = replace(self.config, clearance=clearance, time_budget=budget)
            planner = self._planners.get(key)
            if planner is None:
                planner = self.factory(cfg, self.footprint)
                if len(self._planners) >= 4:
                    self._planners.pop(next(iter(self._planners)))
                self._planners[key] = planner
            else:
                planner.config = cfg
            result = planner.plan(start, goal, obstacles, bounds)
            total_extensions += result.extensions
            attempts.append({'clearance': clearance, 'status': str(result.status), 'elapsed_s': result.elapsed})
            if result.success or result.status in ('invalid_start', 'invalid_goal') and clearance == self.minimum_clearance:
                break
        diagnostics = {**result.diagnostics, 'preferred_clearance': self.config.clearance,
                       'effective_clearance': attempts[-1]['clearance'], 'minimum_clearance': self.minimum_clearance,
                       'clearance_attempts': attempts}
        return replace(result, elapsed=perf_counter()-started, extensions=total_extensions, diagnostics=diagnostics)
