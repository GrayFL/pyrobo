import heapq

import numpy as np
import pytest

from pyrobo import (Bounds2D, NavigationInput, Navigator, PointCloud, PointObstacles,
                    Twist2D, VoronoiConfig, VoronoiPlanner, pose2d)
from pyrobo.planning.voronoi.dstar_lite import DStarLiteGraph
from pyrobo.simulation import Box, World2D, demo_world


def assert_safe_path(result, planner, obstacles, bounds, start, goal):
    assert result.success, (result.status, result.diagnostics)
    xy = result.path.position[:, :2]
    np.testing.assert_allclose(xy[0], start.translation[:2])
    np.testing.assert_allclose(xy[-1], goal.translation[:2])
    assert np.all(bounds.contains(xy, planner.radius))
    # Independent brute-force segment distances, including the endpoint connectors.
    for a, b in zip(xy[:-1], xy[1:]):
        delta = b - a
        t = np.clip((obstacles.xy - a) @ delta / max(delta @ delta, 1e-30), 0, 1)
        distance = np.linalg.norm(obstacles.xy - a - t[:, None] * delta, axis=1)
        assert np.all(distance > planner.radius + obstacles.point_radius - 1e-10)
    return xy


@pytest.mark.parametrize("points", [np.empty((0, 2)), [[0, 0]],
                                    [[0, 0], [0, 0], [0, 0]],
                                    np.column_stack((np.zeros(15), np.linspace(-1, 1, 15)))])
def test_sparse_duplicate_and_collinear_clouds(points):
    obstacles = PointObstacles(PointCloud(points))
    start, goal = pose2d(-2, 0), pose2d(2, 0)
    bounds = Bounds2D(-5, 5, -5, 5)
    planner = VoronoiPlanner(VoronoiConfig(time_budget=2))
    result = planner.plan(start, goal, obstacles, bounds)
    assert_safe_path(result, planner, obstacles, bounds, start, goal)
    np.testing.assert_array_equal(obstacles.cloud.points[:, :2], np.asarray(points))


def test_detour_and_relocated_start_reuse_one_roadmap():
    world, start, goal = demo_world()
    obstacles = PointObstacles(world.point_cloud())
    planner = VoronoiPlanner(VoronoiConfig(time_budget=3))
    first = planner.plan(start, goal, obstacles, world.bounds)
    assert_safe_path(first, planner, obstacles, world.bounds, start, goal)
    assert not first.diagnostics["reused_graph"]
    new_start = pose2d(1.5, 1.7)
    second = planner.plan(new_start, goal, obstacles, world.bounds)
    assert second.diagnostics["reused_graph"]
    assert_safe_path(second, planner, obstacles, world.bounds, new_start, goal)
    planner.reset()
    replay = planner.plan(start, goal, obstacles, world.bounds)
    assert not replay.diagnostics["reused_graph"]
    np.testing.assert_allclose(replay.path.position, first.path.position)


def test_obstacle_addition_removal_and_goal_change_invalidate_graph():
    bounds = Bounds2D(-5, 5, -5, 5)
    planner = VoronoiPlanner(VoronoiConfig(time_budget=2))
    start, goal = pose2d(-2, 0), pose2d(2, 0)
    point = PointObstacles(PointCloud([[0, 0]]))
    assert planner.plan(start, goal, point, bounds).success
    blocked = PointObstacles(PointCloud(np.column_stack((np.zeros(501), np.linspace(-5, 5, 501)))))
    failed = planner.plan(start, goal, blocked, bounds)
    assert failed.status == "no_path" and failed.path.shape == (0,)
    reopened = planner.plan(start, goal, point, bounds)
    assert not reopened.diagnostics["reused_graph"]
    assert_safe_path(reopened, planner, point, bounds, start, goal)
    new_goal = pose2d(2, .1)
    changed = planner.plan(start, new_goal, point, bounds)
    assert not changed.diagnostics["reused_graph"]
    assert_safe_path(changed, planner, point, bounds, start, new_goal)


@pytest.mark.parametrize("limits", [{"max_sites": 4}, {"max_edge_checks": 1},
                                    {"max_expansions": 1}, {"time_budget": 1e-12}])
def test_budgets_fail_without_returning_a_partial_path(limits):
    cfg = VoronoiConfig(**({"time_budget": 2} | limits))
    planner = VoronoiPlanner(cfg)
    result = planner.plan(pose2d(-2, 0), pose2d(2, 0), PointObstacles(PointCloud([[0, 0]])),
                          Bounds2D(-5, 5, -5, 5))
    assert result.status == "budget_exhausted" and result.path.shape == (0,)
    assert result.extensions <= cfg.max_expansions
    assert result.diagnostics["edge_checks"] <= cfg.max_edge_checks


def test_endpoint_validation_frames_and_height_filter():
    planner = VoronoiPlanner(VoronoiConfig(time_budget=2))
    bounds = Bounds2D(-5, 5, -5, 5)
    cloud = PointCloud([[0, 0, 4], [0, 1, .5]], frame="map")
    obstacles = PointObstacles(cloud, z_range=(0, 1))
    start, goal = pose2d(-2, 0, frame="map", body="chassis"), pose2d(2, 0, frame="map")
    result = planner.plan(start, goal, obstacles, bounds)
    assert result.success
    assert result.path.wrd_frame == "map" and result.path.ego_frame == "chassis"
    assert len(cloud.points) == 2
    assert planner.plan(pose2d(-5, 0, frame="map"), goal, obstacles, bounds).status == "invalid_start"
    assert planner.plan(start, pose2d(0, 1, frame="map"), obstacles, bounds).status == "invalid_goal"
    with pytest.raises(ValueError, match="frames"):
        planner.plan(pose2d(-2, 0), goal, obstacles, bounds)


def test_navigator_voronoi_replans_and_brakes_on_disconnected_map():
    nav = Navigator(planner=VoronoiConfig(time_budget=2))
    bounds = Bounds2D(-3, 5, -3, 3)
    def step(timestamp, cloud):
        return nav.step(NavigationInput(timestamp, pose2d(0, 0), Twist2D(.3, 0), pose2d(3, 0), cloud, bounds))
    first = step(0, PointCloud(np.empty((0, 2))))
    second = step(.1, PointCloud([[1.5, 0]]))
    assert first.safe and second.safe and second.diagnostics["replanned"]
    assert second.diagnostics["planner"] == "VoronoiPlanner"
    assert second.diagnostics["planner_status"] == "success"
    wall = World2D(bounds, [Box(1.5, 1.6, -3, 3)]).point_cloud()
    failed = step(.2, wall)
    assert failed.status == "no_path" and failed.safe
    assert failed.command.linear == pytest.approx(.21)
    assert failed.path.shape == (0,)
    # A later complete snapshot removes the wall and retry succeeds.
    assert step(.8, PointCloud(np.empty((0, 2)))).diagnostics["planner_status"] == "success"


def test_dstar_incremental_repairs_match_dijkstra():
    rng = np.random.default_rng(17)
    nodes = dict(enumerate(map(tuple, rng.uniform(-5, 5, (24, 2)))))
    edges = {i: {} for i in nodes}
    for u in nodes:
        for v in nodes:
            if u < v:
                cost = float(np.linalg.norm(np.array(nodes[u]) - nodes[v]))
                edges[u][v] = edges[v][u] = cost
    graph = DStarLiteGraph()
    graph.init_graph(0, 23, nodes, edges)

    def shortest(start):
        queue, seen = [(0., start)], set()
        while queue:
            cost, u = heapq.heappop(queue)
            if u == 23:
                return cost
            if u in seen:
                continue
            seen.add(u)
            for v, length in graph.edges[u].items():
                if v not in seen and np.isfinite(length):
                    heapq.heappush(queue, (cost + length, v))
        return np.inf

    for step in range(30):
        start = step % 10
        graph.move_start(start)
        # Remove both new and previously useful edges; restore edges on alternate rounds.
        for _ in range(8):
            u, v = map(int, rng.choice(len(nodes), 2, replace=False))
            cost = np.inf if step % 2 else float(np.linalg.norm(np.array(nodes[u]) - nodes[v]))
            graph.update_edge(u, v, cost)
        graph.update_edge(start, 23, np.inf)
        expected = shortest(start)
        assert graph.compute_shortest_path() == ("success" if np.isfinite(expected) else "no_path")
        path = graph.get_path()
        actual = sum(graph.edges[u][v] for u, v in zip(path[:-1], path[1:])) if path else np.inf
        assert actual == pytest.approx(expected)
    graph.init_graph(0, 1, {0: (0, 0), 1: (1, 0)}, {0: {1: 1.}, 1: {0: 1.}})
    assert graph.km == 0 and set(graph.g) == {0, 1}
    assert graph.compute_shortest_path() == "success" and graph.get_path() == [0, 1]
