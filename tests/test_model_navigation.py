import numpy as np
import pytest

from pyrobo import (Ackermann, AckermannCommand, Bounds2D, ChassisState, DifferentialDrive,
                    Drag, DWAConfig, DynamicsConfig, DynamicWindow, Footprint, NavigationInput, Navigator, PointCloud,
                    PointObstacles, RRTConnectConfig, Twist2D, VoronoiConfig, WheelCommand, pose2d)
from pyrobo.simulation import Simulator2D, World2D, demo_world, run_navigation
from pyrobo.types import path_poses, planar_state


@pytest.mark.parametrize("model", [DifferentialDrive(), Ackermann()])
def test_prediction_first_control_period_matches_execution(model):
    dwa = DynamicWindow(model=model)
    initial = ChassisState(pose2d(0, 0))
    command = model.command(.6, .3)
    state = dwa.plant.step(initial, command, .2).state
    packed = dwa.plant.pack(state)
    commands = model.encode_command(command)[None]
    trajectory, _, endpoint, _ = dwa._rollout(packed, commands, .1, stop_only=True)
    executed = dwa.plant.step(state, command, .1).state
    np.testing.assert_allclose(trajectory[0, endpoint], dwa.plant.pack(executed), atol=1e-10)


def test_ackermann_dwa_samples_steering_and_never_publishes_a_spin_twist():
    dwa = DynamicWindow(model=Ackermann())
    state = ChassisState(pose2d(0, 0))
    result = dwa.step(state.pose, state.velocity, pose2d(2, 1, .5), path_poses([[0, 0], [2, 1]]),
                      PointObstacles(PointCloud(np.empty((0, 2)))), Bounds2D(-4, 4, -4, 4), .1, chassis=state)
    assert isinstance(result.command, AckermannCommand) and result.safe
    executed = dwa.plant.step(state, result.command, .1).state
    assert executed.velocity.angular == pytest.approx(dwa.model.yaw_rate(executed.velocity.linear, executed.steering))
    assert result.mode != "turning"


@pytest.mark.parametrize("model", [DifferentialDrive(), Ackermann()])
def test_manual_reverse_then_dwa_restores_progress_and_arrives(model):
    world = World2D(Bounds2D(-2, 9, -3, 3))
    nav = Navigator(model=model)
    def manual(k, state, output):
        return model.command(-.35, 0.) if 35 <= k < 65 else None
    run = run_navigation(world, pose2d(0, 0), pose2d(7, 0), navigator=nav,
                         command_override=manual, max_steps=300)
    assert run.arrived and not run.collided
    assert run.manual.sum() == 30 and run.summary()['reverse_steps'] > 10
    progress = np.array([o.diagnostics['progress_m'] for o in run.outputs[:75]])
    assert np.diff(progress[45:65]).min() < -.01
    assert all(np.array_equal(o.path.position, run.outputs[0].path.position) for o in run.outputs[:75])
    assert sum(o.diagnostics['replanned'] for o in run.outputs) == 1
    assert all(isinstance(c, AckermannCommand if isinstance(model, Ackermann) else WheelCommand) for c in run.commands)


@pytest.mark.parametrize("planner_name", ["rrt", "voronoi"])
@pytest.mark.parametrize("footprint", [Footprint.compact(), Footprint.long()])
def test_ackermann_global_dwa_dynamics_detour(planner_name, footprint):
    model = Ackermann()
    clearance = .12 + .4 * model.minimum_turn_radius
    planner = (RRTConnectConfig(clearance=clearance, time_budget=2) if planner_name == "rrt"
               else VoronoiConfig(clearance=clearance, time_budget=2))
    world, start, goal = demo_world()
    run = run_navigation(world, start, goal, navigator=Navigator(model=model, footprint=footprint, planner=planner),
                         max_steps=650)
    assert run.arrived and not run.collided, run.summary()
    np.testing.assert_allclose(run.velocities[:, 1], model.yaw_rate(run.velocities[:, 0], run.steering), atol=1e-9)
    assert abs(float(planar_state(run.poses[-1])[2] - planar_state(goal)[2])) < .11
    assert all(o.safe for o in run.outputs)


def test_ackermann_does_not_report_arrival_at_wrong_heading():
    model = Ackermann()
    nav = Navigator(model=model)
    state = ChassisState(pose2d(0, 0))
    result = nav.step(NavigationInput(0, state.pose, state.velocity, pose2d(0, 0, 1.),
                                     PointCloud(np.empty((0, 2))), Bounds2D(-5, 5, -5, 5), chassis=state))
    assert result.status != 'arrived'
    assert result.diagnostics['controller_mode'] != 'turning'


@pytest.mark.parametrize("model", [DifferentialDrive(), Ackermann()])
def test_sparse_right_angle_path_is_blended_while_moving(model):
    world = World2D(Bounds2D(-3, 6, -3, 6))
    dwa = DynamicWindow(model=model)
    sim = Simulator2D(world, pose2d(0, 0), model=model, dynamics=dwa.plant.config)
    path = path_poses([[0, 0], [3, 0], [3, 3]])
    goal = pose2d(3, 3, np.pi/2)
    obstacles = PointObstacles(world.point_cloud())
    corner_speeds = []
    for _ in range(200):
        state = sim.state
        if np.linalg.norm(state.pose.translation[:2] - [3, 0]) < .8:
            corner_speeds.append(state.velocity.linear)
        result = dwa.step(state.pose, state.velocity, goal, path, obstacles, world.bounds, .1, chassis=state)
        assert result.safe and result.mode not in ('turning', 'turn_braking')
        sim.step(result.command)
        assert not sim.collided
        if sim.state.pose.translation[1] > 1.5:
            break
    assert sim.state.pose.translation[1] > 1.5
    assert corner_speeds and min(corner_speeds) > .1


def test_no_braking_authority_cannot_claim_safe_stop_in_empty_space():
    config = DynamicsConfig(brake_force=0., longitudinal_drag=Drag(0., 0., 0.))
    dwa = DynamicWindow(DWAConfig(braking_timeout=.2), dynamics=config)
    state = ChassisState(pose2d(0, 0), Twist2D(.5, 0))
    result = dwa.stop(state.pose, state.velocity, PointObstacles(PointCloud(np.empty((0, 2)))),
                      Bounds2D(-5, 5, -5, 5), .1, chassis=state)
    assert result.clearance > 0
    assert not result.safe


def test_final_alignment_does_not_discard_global_path():
    nav = Navigator()
    cloud, bounds = PointCloud(np.empty((0, 2))), Bounds2D(-5, 5, -5, 5)
    goal = pose2d(3, 0, np.pi/2)
    first = nav.step(NavigationInput(0., pose2d(0, 0), Twist2D(), goal, cloud, bounds))
    last = nav.step(NavigationInput(1., pose2d(2.9, 0), Twist2D(), goal, cloud, bounds))
    assert last.status == 'aligning'
    np.testing.assert_array_equal(last.path.position, first.path.position)
