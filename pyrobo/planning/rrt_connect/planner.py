"""Bounded bidirectional RRT-Connect in continuous XY space."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from graphmap.pose import Pose, PoseArray

from ...pointcloud import PointObstacles
from ...types import Bounds2D, Footprint, path_poses, planar_state, positive


@dataclass(frozen=True)
class RRTConnectConfig:
    step_size: float = 0.65
    max_extensions: int = 1800
    time_budget: float = 0.045
    goal_bias: float = 0.15
    shortcut_attempts: int = 60
    path_spacing: float = 0.15
    clearance: float = 0.12
    seed: int = 7

    def __post_init__(self):
        for name in ("step_size", "time_budget", "path_spacing"):
            positive(getattr(self, name), name)
        positive(self.clearance, "clearance", zero=True)
        if not np.isfinite(self.goal_bias) or not 0 <= self.goal_bias <= 1:
            raise ValueError("goal_bias must be in [0, 1]")
        for name, lower in (("max_extensions", 1), ("shortcut_attempts", 0)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")


@dataclass(frozen=True)
class PlanResult:
    path: PoseArray
    status: str
    extensions: int
    elapsed: float

    @property
    def success(self) -> bool:
        return self.status == "success"


class _Tree:
    def __init__(self, root, capacity):
        self.nodes = np.empty((capacity + 1, 2))
        self.parent = np.full(capacity + 1, -1, dtype=int)
        self.nodes[0] = root
        self.size = 1

    def branch(self, index):
        result = []
        while index >= 0:
            result.append(self.nodes[index])
            index = self.parent[index]
        return np.array(result[::-1])


class RRTConnect:
    """Each call builds fresh trees. The RNG advances between calls; reset replays it."""
    def __init__(self, config: RRTConnectConfig | None = None, footprint: Footprint | None = None):
        self.config = config or RRTConnectConfig()
        self.footprint = footprint or Footprint()
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.config.seed)

    @property
    def radius(self):
        return self.footprint.radius + self.config.clearance

    def plan(self, start: Pose, goal: Pose, obstacles: PointObstacles, bounds: Bounds2D) -> PlanResult:
        started = perf_counter()
        a, b = planar_state(start)[:2], planar_state(goal)[:2]
        frame = obstacles.cloud.frame
        if start.wrd_frame != frame or goal.wrd_frame != frame:
            raise ValueError("start, goal and obstacle frames must agree and be explicit")
        cfg = self.config
        extensions = 0

        def finish(xy, status):
            return PlanResult(path_poses(xy, frame, start.ego_frame), status, extensions, perf_counter() - started)

        def free(p, q):
            return obstacles.segment_free(p, q, self.radius, bounds)

        if not free(a, a):
            return finish([], "invalid_start")
        if not free(b, b):
            return finish([], "invalid_goal")
        if free(a, b):
            return finish(self._densify(np.array([a, b])), "success")
        trees = [_Tree(a, cfg.max_extensions), _Tree(b, cfg.max_extensions)]
        deadline = started + cfg.time_budget

        def available():
            return extensions < cfg.max_extensions and perf_counter() < deadline

        def extend(tree, target):
            nonlocal extensions
            extensions += 1
            nodes = tree.nodes[:tree.size]
            nearest = int(np.argmin(np.sum((nodes - target) ** 2, axis=1)))
            diff = target - nodes[nearest]
            distance = np.linalg.norm(diff)
            if distance < 1e-10:
                return nearest, True
            new = nodes[nearest] + diff * min(1.0, cfg.step_size / distance)
            if not free(nodes[nearest], new):
                return None, False
            index = tree.size
            tree.nodes[index] = new
            tree.parent[index] = nearest
            tree.size += 1
            return index, distance <= cfg.step_size

        active = 0
        while available():
            first, second = trees[active], trees[1 - active]
            target = (second.nodes[0] if self.rng.random() < cfg.goal_bias else
                      self.rng.uniform([bounds.xmin + self.radius, bounds.ymin + self.radius],
                                       [bounds.xmax - self.radius, bounds.ymax - self.radius]))
            i, _ = extend(first, target)
            if i is not None:
                while available():
                    j, reached = extend(second, first.nodes[i])
                    if j is None:
                        break
                    if reached:
                        left, right = first.branch(i), second.branch(j)
                        if active == 1:
                            left, right = right, left
                        xy = np.concatenate((left, right[-2::-1]))
                        for _ in range(cfg.shortcut_attempts):
                            if len(xy) <= 2 or perf_counter() >= deadline:
                                break
                            lo, hi = np.sort(self.rng.choice(len(xy), 2, replace=False))
                            if hi > lo + 1 and free(xy[lo], xy[hi]):
                                xy = np.concatenate((xy[:lo + 1], xy[hi:]))
                        return finish(self._densify(xy), "success")
            active = 1 - active
        return finish([], "budget_exhausted")

    def _densify(self, xy):
        parts = []
        for a, b in zip(xy[:-1], xy[1:]):
            count = max(1, int(np.ceil(np.linalg.norm(b - a) / self.config.path_spacing)))
            parts.append(np.linspace(a, b, count, endpoint=False))
        return np.concatenate([*parts, xy[-1:]])
