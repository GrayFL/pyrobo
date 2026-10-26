import numpy as np
import pytest

from pyrobo import Bounds2D, NavigationInput, Navigator, PointCloud, Twist2D, pose2d
from pyrobo.control import DWAConfig
from pyrobo.planning import RRTConnectConfig, VoronoiConfig
from pyrobo.simulation import Box, Simulator2D, World2D, demo_world, run_navigation


def observation(timestamp=0., **kwargs):
    data = dict(timestamp=timestamp, pose=pose2d(0, 0), velocity=Twist2D(), goal=pose2d(3, 0),
                obstacles=PointCloud(np.empty((0, 2))), bounds=Bounds2D(-5, 5, -5, 5))
    data.update(kwargs)
    return NavigationInput(**data)


def test_timestamp_validation_and_reset():
    nav = Navigator()
    output = nav.step(observation())
    assert output.timestamp == 0 and output.safe
    with pytest.raises(ValueError, match="strictly increase"):
        nav.step(observation())
    nav.reset()
    assert nav.step(observation()).command == output.command
    with pytest.raises(ValueError, match="cloud timestamp"):
        nav.step(observation(1., obstacles=PointCloud(np.empty((0, 2)), timestamp=0)))
    with pytest.raises(ValueError, match="cloud timestamp"):
        nav.step(observation(1., obstacles=PointCloud(np.empty((0, 2)), timestamp=2)))
    assert nav.step(observation(1.)).safe  # invalid calls did not consume the timestamp


def test_frame_validation_and_goal_change_replan():
    nav = Navigator()
    with pytest.raises(ValueError, match="explicit world frame"):
        nav.step(observation(obstacles=PointCloud([[2, 2]], frame="lidar")))
    nav.step(observation())
    out = nav.step(observation(.1, goal=pose2d(0, 3)))
    assert out.diagnostics['replanned']
    np.testing.assert_allclose(out.path.position[-1], [0, 3, 0])


def test_output_preserves_custom_robot_frame_labels():
    nav = Navigator()
    out = nav.step(observation(pose=pose2d(0, 0, frame="map", body="chassis"),
                               goal=pose2d(3, 0, frame="map", body="target"),
                               obstacles=PointCloud(np.empty((0, 2)), frame="map")))
    assert out.path.wrd_frame == out.trajectory.wrd_frame == "map"
    assert out.path.ego_frame == out.trajectory.ego_frame == "chassis"


def test_new_obstacle_invalidates_cached_path():
    nav = Navigator(planner=RRTConnectConfig(time_budget=1.0))
    nav.step(observation())
    output = nav.step(observation(.1, obstacles=PointCloud([[1.5, 0]])))
    assert output.diagnostics['replanned']
    assert output.diagnostics['planner_status'] == 'success'
    assert np.max(np.abs(output.path.position[:, 1])) > .5


def test_at_goal_must_stop_and_align():
    nav = Navigator()
    out = nav.step(observation(goal=pose2d(0, 0), velocity=Twist2D(.3, 0)))
    assert out.status == "braking" and out.command.mode == "estop"
    out = nav.step(observation(.1, goal=pose2d(0, 0)))
    assert out.status == "arrived" and out.command.mode == "estop"
    out = nav.step(observation(.2, goal=pose2d(0, 0, np.pi / 2)))
    assert out.status == "aligning" and out.command.right > out.command.left


def test_no_route_returns_attainable_braking_command():
    world = World2D(Bounds2D(-3, 5, -3, 3), [Box(1.5, 1.6, -3, 3)])
    nav = Navigator(planner=RRTConnectConfig(max_extensions=20, time_budget=1))
    out = nav.step(observation(velocity=Twist2D(.3, 0), obstacles=world.point_cloud(), bounds=world.bounds))
    assert out.status == "no_path" and out.safe
    assert out.diagnostics["planner_status"] == "budget_exhausted"
    assert out.command.mode == "estop"


def test_simulator_detects_thin_solid_during_a_long_step():
    world = World2D(Bounds2D(-2, 3, -2, 2), [Box(1, 1.01, -1, 1)])
    sim = Simulator2D(world, pose2d(0, 0))
    for _ in range(40):
        state = sim.step(Twist2D(.9, 0))
        if sim.collided:
            break
    assert sim.collided and state.pose.translation[0] < .75
    assert state.velocity == Twist2D()
    with pytest.raises(RuntimeError, match="terminated"):
        sim.step(Twist2D())


def test_runner_reports_step_budget_exhaustion_without_false_arrival():
    world = World2D(Bounds2D(-5, 5, -5, 5))
    run = run_navigation(world, pose2d(0, 0), pose2d(3, 0), max_steps=1)
    assert not run.arrived and not run.collided
    assert len(run.outputs) == len(run.timestamps) == len(run.poses.position) == 1
    assert run.summary()["simulation_seconds"] == 0


@pytest.mark.parametrize("scene", ["detour", "u_trap", "empty"])
@pytest.mark.parametrize("planner", [RRTConnectConfig(time_budget=.5), VoronoiConfig(time_budget=2)])
def test_closed_loop_reaches_goal_without_solid_collision(scene, planner):
    if scene == "detour":
        world, start, goal = demo_world()
    elif scene == "u_trap":
        world = World2D(Bounds2D(0, 10, 0, 10), [Box(3, 3.3, 2, 7), Box(6.7, 7, 2, 7), Box(3, 7, 2, 2.3)])
        start, goal = pose2d(5, 4, -np.pi / 2), pose2d(5, 1)
    else:
        world = World2D(Bounds2D(-5, 5, -5, 5))
        start, goal = pose2d(-3, 0, np.pi), pose2d(3, 0, -np.pi / 2)
    nav = Navigator(planner=planner)
    sim = Simulator2D(world, start)
    cloud = world.point_cloud()
    for _ in range(1000):
        s = sim.state
        out = nav.step(NavigationInput(s.timestamp, s.pose, s.velocity, goal, cloud, world.bounds, chassis=s))
        assert out.safe, (scene, out.status)
        if out.status == "arrived":
            break
        sim.step(out.command)
        assert not sim.collided
    assert out.status == "arrived", (scene, sim.state.pose.translation)
    assert np.linalg.norm(sim.state.pose.translation[:2] - goal.translation[:2]) <= DWAConfig().goal_tolerance
