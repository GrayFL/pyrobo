"""D* Lite for an undirected geometric graph, adapted from the historical project.

Edges must have positive costs at least their Euclidean length (or infinity).
The planner owns graph validation. Removed edges stay at infinity so that they
can be restored. Lazy queue entries are versioned and periodically compacted.
"""
import heapq
import math
from time import perf_counter


class DStarLiteGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.g = {}
        self.rhs = {}
        self._queue = []
        self._pending = {}
        self._serial = 0
        self.km = 0.0
        self.expansions = 0
        self.s_start = self.s_goal = None

    def init_graph(self, start_id, goal_id, nodes, edges):
        self.nodes = {u: tuple(xy) for u, xy in nodes.items()}
        self.edges = {u: dict(edges.get(u, {})) for u in nodes}
        self.s_start, self.s_goal = start_id, goal_id
        self.km = 0.0
        self.g = dict.fromkeys(nodes, math.inf)
        self.rhs = dict.fromkeys(nodes, math.inf)
        self.rhs[goal_id] = 0.0
        self._queue, self._pending, self._serial = [], {}, 0
        self.expansions = 0
        self._push(goal_id)

    def heuristic(self, u, v):
        return math.dist(self.nodes[u], self.nodes[v])

    def calculate_key(self, u):
        cost = min(self.g[u], self.rhs[u])
        return cost + self.heuristic(self.s_start, u) + self.km, cost

    def _push(self, u):
        self._serial += 1
        entry = (*self.calculate_key(u), self._serial, u)
        self._pending[u] = entry
        heapq.heappush(self._queue, entry)

    def _top(self):
        while self._queue and self._pending.get(self._queue[0][3]) != self._queue[0]:
            heapq.heappop(self._queue)
        return self._queue[0][:2] if self._queue else (math.inf, math.inf)

    def update_vertex(self, u):
        if u != self.s_goal:
            self.rhs[u] = min((cost + self.g[v] for v, cost in self.edges[u].items()), default=math.inf)
        self._pending.pop(u, None)
        if self.g[u] != self.rhs[u]:
            self._push(u)
        if len(self._queue) > 4 * len(self.nodes) + 32:
            self._queue = list(self._pending.values())
            heapq.heapify(self._queue)

    def move_start(self, start_id, position=None):
        old_position = self.nodes[self.s_start]
        if position is not None:
            self.nodes[start_id] = tuple(position)
        self.km += math.dist(old_position, self.nodes[start_id])
        self.s_start = start_id

    def update_edge(self, u, v, cost):
        if cost <= 0 or math.isnan(cost):
            raise ValueError("edge cost must be positive or infinity")
        self.edges[u][v] = self.edges[v][u] = cost
        self.update_vertex(u)
        self.update_vertex(v)

    def compute_shortest_path(self, *, max_expansions=10000, deadline=math.inf):
        self.expansions = 0
        while self._top() < self.calculate_key(self.s_start) or self.rhs[self.s_start] != self.g[self.s_start]:
            if self.expansions >= max_expansions or perf_counter() >= deadline:
                return "budget_exhausted"
            if not self._queue:
                return "no_path"
            old_key = self._top()
            _, _, _, u = heapq.heappop(self._queue)
            self._pending.pop(u, None)
            new_key = self.calculate_key(u)
            self.expansions += 1
            if old_key < new_key:
                self._push(u)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for v in self.edges[u]:
                    self.update_vertex(v)
            else:
                self.g[u] = math.inf
                for v in (*self.edges[u], u):
                    self.update_vertex(v)
        return "success" if math.isfinite(self.g[self.s_start]) else "no_path"

    def get_path(self):
        if self.s_start is None or not math.isfinite(self.g[self.s_start]):
            return []
        path, visited = [self.s_start], {self.s_start}
        while path[-1] != self.s_goal:
            current = path[-1]
            best = min(self.edges[current], key=lambda v: self.edges[current][v] + self.g[v], default=None)
            if best is None or best in visited or not math.isfinite(self.edges[current][best] + self.g[best]):
                return []
            path.append(best)
            visited.add(best)
        return path
