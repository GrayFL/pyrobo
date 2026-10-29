"""Bounded bidirectional RRT-Connect in continuous XY space."""
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
from graphmap.pose import Pose

from ..common import PlanResult, densify_path
from pyrobo.geometry.collision import PointObstacles
from pyrobo.core.obstacles import Bounds2D
from pyrobo.core.footprint import Footprint
from pyrobo.core.geometry import path_poses, planar_state, positive


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
    bridge_sampling: bool = True

    def __post_init__(self):
        for name in ("step_size", "time_budget", "path_spacing"):
            positive(getattr(self, name), name)
        positive(self.clearance, "clearance", zero=True)
        if not isinstance(self.bridge_sampling, bool):
            raise ValueError('bridge_sampling must be boolean')
        if not np.isfinite(self.goal_bias) or not 0 <= self.goal_bias <= 1:
            raise ValueError("goal_bias must be in [0, 1]")
        for name, lower in (("max_extensions", 1), ("shortcut_attempts", 0)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")


class _Tree:
    def __init__(self, root, capacity, yaw, reverse=False):
        self.nodes = np.empty((capacity + 1, 2))
        self.parent = np.full(capacity + 1, -1, dtype=int)
        self.nodes[0] = root
        self.heading = np.empty(capacity + 1)
        self.heading[0] = yaw
        self.reverse = reverse
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
        """Broad-phase bound only; collision checks use safety_footprint."""
        return self.footprint.radius + self.config.clearance

    @property
    def safety_footprint(self):
        return replace(self.footprint, margin=self.footprint.margin + self.config.clearance)

    def plan(self, start: Pose, goal: Pose, obstacles: PointObstacles, bounds: Bounds2D) -> PlanResult:
        started = perf_counter()
        initial, final = planar_state(start), planar_state(goal)
        a, b = initial[:2], final[:2]
        frame = obstacles.cloud.frame
        if start.wrd_frame != frame or goal.wrd_frame != frame:
            raise ValueError("start, goal and obstacle frames must agree and be explicit")
        cfg, footprint = self.config, self.safety_footprint
        extensions = 0
        bridge_samples = 0

        def finish(xy, status):
            return PlanResult(path_poses(xy, frame, start.ego_frame), status, extensions, perf_counter() - started,
                              {'bridge_samples': bridge_samples})

        def free(p, q):
            return (obstacles.rectangle_segment_free(p, q, footprint, bounds)
                    and obstacles.rectangle_segment_free(q, p, footprint, bounds))

        def path_free(xy):
            return obstacles.path_free(xy, footprint, bounds,
                                       start_yaw=initial[2], goal_yaw=final[2])

        if obstacles.clearances([initial], footprint, bounds)[0] <= 0:
            return finish([], "invalid_start")
        if obstacles.clearances([final], footprint, bounds)[0] <= 0:
            return finish([], "invalid_goal")
        if path_free([a, b]):
            return finish(self._densify(np.array([a, b])), "success")
        trees = [_Tree(a, cfg.max_extensions, initial[2]), _Tree(b, cfg.max_extensions, final[2], True)]
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
            direction = -diff if tree.reverse else diff
            heading = float(np.arctan2(direction[1], direction[0]))
            if not obstacles.rotation_free(nodes[nearest], tree.heading[nearest], heading,
                                           footprint, bounds):
                return None, False
            index = tree.size
            tree.nodes[index] = new
            tree.heading[index] = heading
            tree.parent[index] = nearest
            tree.size += 1
            return index, distance <= cfg.step_size

        active = 0
        while available():
            first, second = trees[active], trees[1 - active]
            if self.rng.random() < cfg.goal_bias:
                target = second.nodes[0]
            else:
                target = None
                if cfg.bridge_sampling and len(obstacles.xy) and self.rng.random() < .35:
                    # Obstacle-surface bridge sampling: the midpoint of nearby
                    # wall samples can lie in a narrow passage. It only biases
                    # targets; every tree edge/rotation still gets exact checks.
                    span = 2*(max(footprint.length, footprint.width)+2*(footprint.margin+obstacles.point_radius))
                    for _ in range(3):
                        p = obstacles.xy[self.rng.integers(len(obstacles.xy))]
                        q = obstacles.xy[obstacles.tree.query(p+self.rng.normal(size=2)*span)[1]]
                        vector = q-p
                        distance = np.linalg.norm(vector)
                        if distance <= footprint.width+2*(footprint.margin+obstacles.point_radius) or distance > 2*span:
                            continue
                        midpoint = (p+q)/2
                        yaw = np.arctan2(vector[1], vector[0])+np.pi/2
                        if obstacles.clearances([[*midpoint, yaw]], footprint, bounds)[0] > 0:
                            target = midpoint
                            bridge_samples += 1
                            break
                if target is None:
                    target = self.rng.uniform([bounds.xmin, bounds.ymin], [bounds.xmax, bounds.ymax])
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
                        if not path_free(xy):
                            break
                        for _ in range(cfg.shortcut_attempts):
                            if len(xy) <= 2 or perf_counter() >= deadline:
                                break
                            lo, hi = np.sort(self.rng.choice(len(xy), 2, replace=False))
                            candidate = np.concatenate((xy[:lo + 1], xy[hi:]))
                            if hi > lo + 1 and path_free(candidate):
                                xy = candidate
                        return finish(self._densify(xy), "success")
            active = 1 - active
        return finish([], "budget_exhausted")

    def _densify(self, xy):
        return densify_path(xy, self.config.path_spacing)
