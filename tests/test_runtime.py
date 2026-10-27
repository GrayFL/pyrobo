from dataclasses import replace

import numpy as np
import pytest

from pyrobo import (Ackermann, Bounds2D, ChassisState, Navigator, PointCloud, Twist2D, pose2d)
from pyrobo.core import (Diagnostics, MotionPolicy, MotionResult, MotionStatus, RobotObservation,
                         SafetyAssessment)
from pyrobo.runtime import ControlMode, Supervisor
from pyrobo.safety import SafetyConfig, SafetyGuard
from pyrobo.simulation import World2D, run_navigation
from pyrobo.planning.common import PlanResult
from pyrobo.types import path_poses


def obs(t=0, *, x=0, cloud=None):
    return RobotObservation(ChassisState(pose2d(x, 0), timestamp=t),
                            cloud or PointCloud(np.empty((0, 2))), Bounds2D(-5, 9, -5, 5))


class CountingOptimizer:
    def __init__(self, engine):
        self.engine, self.requests = engine, []

    def optimize(self, observation, request):
        self.requests.append(request)
        return self.engine.optimize(observation, request)


def test_manual_observes_progress_without_running_auto_optimizer_and_recovers_reverse():
    nav = Navigator()
    counter = CountingOptimizer(nav.optimizer)
    runtime = Supervisor(nav, optimizer=counter)
    runtime.set_goal(pose2d(7, 0), 0.)
    runtime.step(obs())
    full_path = nav.path.xy.copy()
    runtime.step(obs(.1, x=2.))
    forward = nav.progress.distance
    runtime.set_mode(ControlMode.MANUAL)
    for t, x in ((.2, 1.8), (.3, 1.5), (.4, 1.2)):
        runtime.submit_teleop(Twist2D(-.3, 0), t)
        runtime.step(obs(t, x=x))
        np.testing.assert_array_equal(nav.path.xy, full_path)
    assert len(counter.requests) == 5
    assert all(r.policy == MotionPolicy.TELEOP_STOP for r in counter.requests[2:])
    assert nav.progress.distance < forward
    runtime.set_mode(ControlMode.AUTO)
    runtime.step(obs(.5, x=1.))
    assert counter.requests[-1].policy == MotionPolicy.NAVIGATION
    np.testing.assert_array_equal(nav.path.xy, full_path)
    assert not nav._replanned


def test_authority_timeout_and_estop_latch():
    nav = Navigator()
    runtime = Supervisor(nav, mode=ControlMode.MANUAL, authorized_sources=('joystick',))
    with pytest.raises(PermissionError):
        runtime.submit_teleop(Twist2D(.5, 0), 0, source='rogue')
    runtime.submit_teleop(Twist2D(.5, 0), 0, source='joystick')
    assert runtime.step(obs()).command.mode == 'track'
    expired = runtime.step(obs(.31))
    assert expired.command.mode == 'estop' and expired.safety_assessment == SafetyAssessment.INPUT_STALE
    assert expired.diagnostics.reason == 'teleop_timeout'
    runtime.set_mode(ControlMode.ESTOP, source='joystick')
    runtime.submit_teleop(Twist2D(.9, 0), .4, source='joystick')
    assert runtime.step(obs(.4)).command.mode == 'estop'
    with pytest.raises(RuntimeError, match='latched'):
        runtime.set_mode(ControlMode.AUTO, source='joystick')
    with pytest.raises(PermissionError):
        runtime.release_estop(source='rogue')
    runtime.release_estop(source='joystick')
    assert runtime.mode == ControlMode.PAUSED
    runtime.set_mode(ControlMode.MANUAL, source='joystick')
    assert runtime.step(obs(.5)).diagnostics.reason == 'teleop_timeout'


def test_automatic_task_timeout_requires_explicit_renewal():
    runtime = Supervisor(Navigator(), auto_task_ttl=.2)
    runtime.set_goal(pose2d(4, 0), 0)
    assert runtime.step(obs()).command.mode == 'track'
    stopped = runtime.step(obs(.3))
    assert stopped.command.mode == 'estop' and stopped.diagnostics.reason == 'auto_task_timeout'
    runtime.renew_task(.4)
    assert runtime.step(obs(.4)).command.mode == 'track'


def test_final_gate_does_not_trust_optimizer_and_detects_bad_adapters():
    nav = Navigator()
    o = obs(cloud=PointCloud([[.42, 0]]))
    o = replace(o, chassis=replace(o.chassis, velocity=Twist2D(.6, 0)))
    bogus = MotionResult(nav.model.command(.9, 0), MotionStatus.TRACKING,
                         SafetyAssessment.SAFE_UNDER_MODEL, None, Diagnostics())
    result = nav.safety.enforce(o, bogus)
    assert result.command.mode == 'estop' and result.diagnostics.reason == 'final_gate_rejected'
    invalid = nav.safety.enforce(obs(), replace(bogus, command=Ackermann().command(.5, .3)))
    assert invalid.command.mode == 'estop' and invalid.safety_assessment == SafetyAssessment.MODEL_INVALID


@pytest.mark.parametrize('case', ['map', 'state', 'dt', 'clock'])
def test_stale_input_and_control_period_cannot_reach_optimizer(case):
    nav = Navigator()
    counter = CountingOptimizer(nav.optimizer)
    gate = SafetyGuard(nav.dynamics, nav.command_space, nav.footprint, config=SafetyConfig(expected_dt=.1))
    runtime = Supervisor(nav, optimizer=counter, safety=gate)
    runtime.set_goal(pose2d(4, 0), 0)
    runtime.step(obs())
    current = obs(.2)
    if case == 'map':
        current = replace(current, obstacles=PointCloud(np.empty((0, 2)), timestamp=-1))
    elif case == 'state':
        current = replace(current, timestamp=.5)
    elif case == 'dt':
        current = replace(current, dt=.2)
    else:
        current = obs()
    result = runtime.step(current)
    assert result.command.mode == 'estop' and result.safety_assessment == SafetyAssessment.INPUT_STALE
    assert len(counter.requests) == 1


def test_navigation_accepts_independent_planner_and_optimizer():
    class StraightPlanner:
        def __init__(self):
            self.calls = 0
        def plan(self, start, goal, obstacles, bounds):
            self.calls += 1
            return PlanResult(path_poses([start.translation[:2], goal.translation[:2]]), 'success', 0, 0)
    engine = Navigator().optimizer
    planner, optimizer = StraightPlanner(), CountingOptimizer(engine)
    nav = Navigator(global_planner=planner, local_optimizer=optimizer)
    current = obs()
    nav.observe(current, goal=pose2d(4, 0))
    assert planner.calls == 0 and not optimizer.requests
    request = nav.make_motion_request(current)
    assert planner.calls == 1 and request.policy == MotionPolicy.NAVIGATION
    assert not optimizer.requests
    result = optimizer.optimize(current, request)
    assert result.safe


@pytest.mark.parametrize('model', [None, Ackermann()])
def test_supervised_manual_reverse_handoff_closed_loop(model):
    world = World2D(Bounds2D(-2, 9, -3, 3))
    def inputs(k, state, runtime):
        runtime.set_mode(ControlMode.MANUAL if 35 <= k < 65 else ControlMode.AUTO)
        if runtime.mode == ControlMode.MANUAL:
            runtime.submit_teleop(Twist2D(-.35, 0), state.timestamp)
    run = run_navigation(world, pose2d(0, 0), pose2d(7, 0), navigator=Navigator(model=model),
                         control_input=inputs, max_steps=320)
    assert run.arrived and not run.collided
    assert run.manual.sum() == 30 and run.summary()['reverse_steps'] > 10
    assert run.summary()['replans'] == 1
    progress = [o.diagnostics['progress_m'] for o in run.outputs[:75]]
    assert np.min(np.diff(progress[45:65])) < -.01
    assert all(np.array_equal(o.path.position, run.outputs[0].path.position) for o in run.outputs[:75])


def test_model_state_error_stops_without_publishing_an_unchecked_motion():
    runtime = Supervisor(Navigator())
    runtime.set_goal(pose2d(4, 0), 0)
    current = obs()
    current = replace(current, chassis=replace(current.chassis, velocity=Twist2D(2., 0)))
    result = runtime.step(current)
    assert result.command.mode == 'estop' and not result.safe
    assert result.safety_assessment == SafetyAssessment.MODEL_INVALID


def test_wrong_global_frame_is_a_guarded_planning_error():
    class WrongFrame:
        def plan(self, *args):
            return PlanResult(path_poses([[0, 0], [4, 0]], 'sensor'), 'success', 0, 0.)
    runtime = Supervisor(Navigator(global_planner=WrongFrame()))
    runtime.set_goal(pose2d(4, 0), 0)
    result = runtime.step(obs())
    assert result.command.mode == 'estop' and not result.safe
    assert result.diagnostics.reason == 'planning_error'


def test_legacy_override_cannot_bypass_final_native_capability_check():
    world = World2D(Bounds2D(-3, 5, -3, 3))
    nav = Navigator()
    def invalid_override(k, state, output):
        return nav.model.command(10., 0.)
    with pytest.warns(DeprecationWarning):
        run = run_navigation(world, pose2d(0, 0), pose2d(4, 0), navigator=nav,
                             command_override=invalid_override, max_steps=2)
    assert run.commands[0].mode == 'estop'
    assert not run.outputs[0].safe and not run.collided
    assert run.outputs[0].diagnostics['override_reason'] == 'command_not_representable'


@pytest.mark.parametrize('fault', ['prediction', 'clearance'])
def test_final_gate_rejects_nonfinite_model_or_collision_evidence(monkeypatch, fault):
    from types import SimpleNamespace
    nav = Navigator()
    current = obs()
    bogus = MotionResult(nav.model.command(0., 0.), MotionStatus.TRACKING,
                         SafetyAssessment.SAFE_UNDER_MODEL, None)
    if fault == 'prediction':
        states = np.zeros((1, 2, 10))
        states[..., 0] = np.nan
        batch = SimpleNamespace(states=states, times=np.array([0., .1]), sweep=np.zeros((1, 2)),
                                prediction_end=1, stopped=np.array([True]), valid_time_mask=np.ones((1, 2), bool))
        monkeypatch.setattr(nav.safety.predictor, 'predict', lambda *args, **kwargs: batch)
    else:
        world = nav.safety.index.get(current.obstacles)
        monkeypatch.setattr(world, 'clearances', lambda poses, *args: np.full(poses.shape[:-1], np.nan))
    result = nav.safety.enforce(current, bogus)
    assert result.command.mode == 'estop' and not result.safe
    assert result.safety_assessment == SafetyAssessment.MODEL_INVALID


def test_goal_frame_change_fault_returns_stop_before_any_optimizer_call():
    nav = Navigator()
    counter = CountingOptimizer(nav.optimizer)
    runtime = Supervisor(nav, optimizer=counter)
    runtime.set_goal(pose2d(4, 0, frame='different_map'), 0.)
    result = runtime.step(obs())
    assert not counter.requests and result.command.mode == 'estop'
    assert result.safety_assessment == SafetyAssessment.MODEL_INVALID
    assert result.diagnostics.reason == 'navigation_observation_error'
