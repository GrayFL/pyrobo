"""Point-cloud Voronoi roadmap and bounded D* Lite search; no occupancy grid."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from graphmap.pose import Pose
from scipy.spatial import QhullError, Voronoi, cKDTree

from ..common import PlanResult, densify_path
from ...pointcloud import PointObstacles
from ...types import Bounds2D, Footprint, path_poses, planar_state, positive
from .dstar_lite import DStarLiteGraph


@dataclass(frozen=True)
class VoronoiConfig:
    boundary_step: float = 0.5
    max_edge_length: float = 15.0
    connection_neighbors: int = 32
    path_spacing: float = 0.15
    clearance: float = 0.12
    safety_distance: float = 0.5
    safety_weight: float = 1.0
    time_budget: float = 0.2
    max_sites: int = 20000
    max_edge_checks: int = 50000
    max_expansions: int = 10000

    def __post_init__(self):
        for name in ("boundary_step", "max_edge_length", "path_spacing", "safety_distance", "time_budget"):
            positive(getattr(self, name), name)
        for name in ("clearance", "safety_weight"):
            positive(getattr(self, name), name, zero=True)
        for name, lower in (("connection_neighbors", 1), ("max_sites", 4),
                            ("max_edge_checks", 1), ("max_expansions", 1)):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")


class _BudgetExhausted(Exception):
    pass


class VoronoiPlanner:
    """Cache one roadmap; rebuild on obstacle/goal/bounds/frame changes.

    Same-map replans reconnect the measured start and reuse D* Lite's costs.
    Point-cloud snapshots replace the map; points are never accumulated or
    implicitly downsampled. Qhull is one non-interruptible, size-limited call.
    """
    def __init__(self, config: VoronoiConfig | None = None, footprint: Footprint | None = None):
        self.config = config or VoronoiConfig()
        self.footprint = footprint or Footprint()
        self.reset()

    def reset(self):
        self._signature = None
        self._points = None
        self._nodes = np.empty((0, 2))
        self._dstar = None

    @property
    def radius(self):
        return self.footprint.radius + self.config.clearance

    def plan(self, start: Pose, goal: Pose, obstacles: PointObstacles, bounds: Bounds2D) -> PlanResult:
        started = perf_counter()
        a, b = planar_state(start)[:2], planar_state(goal)[:2]
        frame = obstacles.cloud.frame
        if start.wrd_frame != frame or goal.wrd_frame != frame:
            raise ValueError("start, goal and obstacle frames must agree and be explicit")
        cfg, radius = self.config, self.radius
        deadline = started + cfg.time_budget
        diagnostics = {"reused_graph": False, "sites": 0, "vertices": 0, "edges": 0, "edge_checks": 0}
        expansions = 0

        def finish(xy, status):
            return PlanResult(path_poses(xy, frame, start.ego_frame), status, expansions,
                              perf_counter() - started, diagnostics)

        def check_budget():
            if perf_counter() >= deadline:
                raise _BudgetExhausted

        def edge_cost(p, q):
            check_budget()
            if diagnostics["edge_checks"] >= cfg.max_edge_checks:
                raise _BudgetExhausted
            diagnostics["edge_checks"] += 1
            length = float(np.linalg.norm(q - p))
            if length > cfg.max_edge_length or not obstacles.segment_free(p, q, radius, bounds):
                return np.inf
            # Samples influence preference only. Safety uses the exact whole edge above.
            sample = np.array([p, (p + q) / 2, q])
            distance = obstacles.tree.query(sample)[0] - obstacles.point_radius
            boundary_distance = np.minimum.reduce([sample[:, 0] - bounds.xmin, bounds.xmax - sample[:, 0],
                                                   sample[:, 1] - bounds.ymin, bounds.ymax - sample[:, 1]])
            clearance = float(np.minimum(distance, boundary_distance).min()) - radius
            penalty = max(0.0, 1.0 - clearance / cfg.safety_distance)
            return max(length, 1e-12) * (1.0 + cfg.safety_weight * penalty)

        if not obstacles.segment_free(a, a, radius, bounds):
            return finish([], "invalid_start")
        if not obstacles.segment_free(b, b, radius, bounds):
            return finish([], "invalid_goal")
        if obstacles.segment_free(a, b, radius, bounds):
            self.reset()
            return finish(densify_path(np.array([a, b]), cfg.path_spacing), "success")

        signature = (bounds, frame, obstacles.point_radius, tuple(b), cfg, self.footprint)
        reuse = self._signature == signature and np.array_equal(self._points, obstacles.xy)
        diagnostics["reused_graph"] = reuse
        try:
            check_budget()
            if reuse:
                nodes = self._nodes.copy()
                graph = self._dstar
            else:
                self.reset()
                sites = self._sites(obstacles.xy, bounds)
                diagnostics["sites"] = len(sites)
                check_budget()
                vor = Voronoi(sites)
                check_budget()
                valid = np.isfinite(vor.vertices).all(axis=1) & bounds.contains(vor.vertices, radius)
                indices = np.flatnonzero(valid)
                valid[indices] &= obstacles.tree.query(vor.vertices[indices])[0] > radius + obstacles.point_radius
                vertices = vor.vertices[valid]
                mapping = np.full(len(valid), -1, dtype=int)
                mapping[valid] = np.arange(len(vertices))
                nodes = np.vstack((vertices, a, b))
                edges = {i: {} for i in range(len(nodes))}
                for ridge in vor.ridge_vertices:
                    check_budget()
                    # -1 means an infinite ridge, never an index into the final vertex.
                    if len(ridge) != 2 or min(ridge) < 0:
                        continue
                    u, v = mapping[ridge]
                    if u < 0 or v < 0 or u == v:
                        continue
                    cost = edge_cost(nodes[u], nodes[v])
                    if np.isfinite(cost):
                        edges[u][v] = edges[v][u] = cost
                graph = DStarLiteGraph()
                graph.init_graph(len(nodes) - 2, len(nodes) - 1,
                                 dict(enumerate(map(tuple, nodes))), edges)

            diagnostics["vertices"] = len(nodes) - 2
            start_id, goal_id = len(nodes) - 2, len(nodes) - 1
            tree = cKDTree(nodes[:start_id])

            def connections(position):
                if start_id == 0:
                    return {}
                ids = np.atleast_1d(tree.query(position, k=min(cfg.connection_neighbors, start_id))[1])
                result = {}
                for node_id in ids:
                    cost = edge_cost(position, nodes[node_id])
                    if np.isfinite(cost):
                        result[int(node_id)] = cost
                return result

            # Prepare both sets before mutating an existing incremental search.
            start_edges = connections(a)
            goal_edges = None if reuse else connections(b)
            check_budget()
            nodes[start_id] = a
            graph.move_start(start_id, a)
            for node_id in set(graph.edges[start_id]) | start_edges.keys():
                graph.update_edge(start_id, node_id, start_edges.get(node_id, np.inf))
            if goal_edges is not None:
                for node_id, cost in goal_edges.items():
                    graph.update_edge(goal_id, node_id, cost)
            self._nodes, self._dstar = nodes, graph
            self._signature, self._points = signature, obstacles.xy.copy()
            diagnostics["edges"] = sum(bool(np.isfinite(cost)) for row in graph.edges.values() for cost in row.values()) // 2
            status = graph.compute_shortest_path(max_expansions=cfg.max_expansions, deadline=deadline)
            expansions = graph.expansions
            if status != "success":
                return finish([], status)
            path_ids = graph.get_path()
            if not path_ids:
                return finish([], "no_path")
            check_budget()
            return finish(densify_path(nodes[path_ids], cfg.path_spacing), "success")
        except _BudgetExhausted:
            return finish([], "budget_exhausted")
        except QhullError:
            self.reset()
            return finish([], "degenerate_geometry")

    def _sites(self, xy, bounds):
        cfg = self.config
        # A physical rectangular workspace replaces the old inferred convex hull.
        # Its samples close the diagram; boundary collision checks remain analytic.
        corners = np.array([[bounds.xmin, bounds.ymin], [bounds.xmax, bounds.ymin],
                            [bounds.xmax, bounds.ymax], [bounds.xmin, bounds.ymax]])
        lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        counts = np.maximum(2, np.ceil(lengths / cfg.boundary_step))
        if len(xy) + counts.sum() > cfg.max_sites:
            raise _BudgetExhausted
        boundary = [np.linspace(a, b, int(n), endpoint=False)
                    for a, b, n in zip(corners, np.roll(corners, -1, axis=0), counts)]
        # Exact duplicates have identical geometry; all original points stay in the collision index.
        return np.unique(np.vstack((xy, *boundary)), axis=0)
