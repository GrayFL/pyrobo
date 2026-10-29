"""Backend integration for the optional GUI; these tests do not require SDL."""
from threading import Event
from time import monotonic, sleep
from pathlib import Path

import numpy as np
import pytest

from pyrobo.apps.pygame_sim.config import load_config, parse_args
from pyrobo.apps.pygame_sim.session import Action, ControlInput, Session, Worker
from pyrobo import ControlMode, MotionPolicy


def config(**backend):
    return load_config(selectors={'map.preset': 'empty', **{f'backend.{k}': v for k, v in backend.items()}})


def test_config_precedence_nested_drag_and_inactive_model_tables(tmp_path):
    path = tmp_path/'example.toml'
    path.write_text('[backend]\nmodel="differential"\n[ackermann]\nwheelbase=0.7\n')
    args, data = parse_args(['--config', str(path), '--model', 'ackermann', '--planner', 'voronoi',
                             '--set', 'dynamics.longitudinal_drag.quadratic=3.0', '--speed', '.4',
                             '--set', 'teleop.speed_limit=0.5'])
    session = Session(data)
    assert session.model.wheelbase == .7 and data['teleop']['speed_limit'] == .5
    assert session.sim.dynamics.longitudinal_drag.quadratic == 3.
    assert session.nav.optimizer.config.prediction_time == 2.4
    assert type(session.nav.planner).__name__ == 'AdaptiveClearancePlanner'
    assert session.nav.planner.factory.__name__ == 'VoronoiPlanner'


@pytest.mark.parametrize('override', ['backend.local_optimizer="unknown"', 'dwa.angular_sampels=9',
                                      'app.dt=0.8', 'joystick.deadzone=1.0', 'map.start=[0,0,0,1]',
                                      'safety.expected_dt=0.2', 'app.fps=0'])
def test_invalid_configuration_is_rejected_before_opening_a_window(override):
    with pytest.raises((ValueError, TypeError)):
        load_config(overrides=[override])


def test_custom_map_and_spawn_validation():
    data = load_config(overrides=['map.preset="custom"', 'map.bounds=[-4,4,-3,3]',
                                  'map.start=[-2,0,0]', 'map.boxes=[[0,1,-1,1]]'])
    session = Session(data)
    assert session.world.boxes[0].xmin == 0 and session.nav.goal is None
    with pytest.raises(ValueError, match='start intersects'):
        load_config(overrides=['map.preset="custom"', 'map.bounds=[-4,4,-3,3]',
                               'map.start=[0.5,0,0]', 'map.boxes=[[0,1,-1,1]]'])


@pytest.mark.parametrize('model', ['differential', 'ackermann'])
@pytest.mark.parametrize('planner', ['rrt_connect', 'voronoi'])
def test_clicked_goal_runs_real_backend_and_reaches_goal(model, planner):
    session = Session(config(model=model, planner=planner))
    snapshot = session.tick(actions=[Action('goal', (0., 0., 0.))])
    assert snapshot.mode == ControlMode.AUTO and len(snapshot.path) > 1
    for _ in range(130):
        snapshot = session.tick()
        assert not snapshot.collided and snapshot.motion.safe
        if snapshot.navigation_status == 'arrived':
            break
    assert snapshot.navigation_status == 'arrived'
    assert np.linalg.norm(snapshot.state.pose.translation[:2]) <= session.nav.settings.goal_tolerance


def test_shipped_config_plans_and_drives_long_ackermann_around_walls():
    template = Path(__file__).resolve().parents[1]/'pyrobo/apps/configs/pygame.toml'
    session = Session(load_config(template, selectors={'backend.model': 'ackermann',
                      'backend.planner': 'voronoi', 'app.start_mode': 'auto'},
                      overrides=['footprint.length=1.0']))
    start = session.sim.state.pose.translation.copy()
    for _ in range(60):
        snapshot = session.tick()
        assert not snapshot.collided and snapshot.motion.safe
    assert snapshot.planner_status == 'success' and len(snapshot.path) > 2
    assert np.linalg.norm(snapshot.state.pose.translation-start) > 1.


def test_takeover_preserves_task_and_state_and_resume_retains_full_path():
    session = Session(config())
    snapshot = session.tick(actions=[Action('goal', (2., 0., 0.))])
    original = snapshot.path.copy()
    for _ in range(15):
        snapshot = session.tick()
    before = snapshot.state
    snapshot = session.tick(ControlInput(throttle=-1., engaged=True))
    assert snapshot.mode == ControlMode.MANUAL
    assert snapshot.state.timestamp == pytest.approx(before.timestamp+.1)
    assert np.linalg.norm(snapshot.state.pose.translation-before.pose.translation) < .1
    for _ in range(22):
        snapshot = session.tick(ControlInput(throttle=-1., engaged=True))
    assert snapshot.state.velocity.linear < 0
    snapshot = session.tick(actions=[Action('mode', 'auto')])
    np.testing.assert_array_equal(snapshot.path, original)
    assert snapshot.mode == ControlMode.AUTO


def test_focus_or_stale_input_pauses_and_estop_requires_explicit_release():
    session = Session(config())
    session.tick(ControlInput(throttle=1, engaged=True))
    s = session.tick(ControlInput(throttle=1, engaged=True, focused=False))
    assert s.mode == ControlMode.PAUSED and s.motion.command.mode == 'estop'
    s = session.tick(ControlInput(throttle=1, engaged=True, timestamp=monotonic()-2))
    assert s.mode == ControlMode.PAUSED
    session.tick(actions=[Action('estop')])
    s = session.tick(ControlInput(throttle=1, engaged=True), actions=[Action('mode', 'auto')])
    assert s.mode == ControlMode.ESTOP and s.motion.command.mode == 'estop'
    s = session.tick(actions=[Action('release')])
    assert s.mode == ControlMode.PAUSED


def test_collision_rejection_and_new_intent_before_execution():
    session = Session(load_config())
    goal = session.nav.goal
    snapshot = session.tick(actions=[Action('goal', (3.8, 2., 0.))])
    assert snapshot.goal is goal and 'rejected' in snapshot.notice
    snapshot = session.tick(ControlInput(throttle=1., engaged=True), before_execute=lambda: (True, True))
    assert snapshot.mode == ControlMode.ESTOP and snapshot.motion.command.mode == 'estop'
    assert snapshot.state.velocity.linear == 0


def test_worker_keeps_urgent_stop_when_optimizer_is_busy(monkeypatch):
    session = Session(config())
    entered, release = Event(), Event()
    original = session.runtime.optimizer.optimize
    def slow(observation, request):
        result = original(observation, request)
        entered.set()
        assert release.wait(2.)
        return result
    monkeypatch.setattr(session.runtime.optimizer, 'optimize', slow)
    worker = Worker(session)
    worker.publish(ControlInput(throttle=1., engaged=True))
    worker.start()
    try:
        assert entered.wait(2.)
        worker.enqueue(Action('estop'))
        worker.enqueue(Action('mode', 'auto'))
        release.set()
        deadline = monotonic()+2.
        while worker.snapshot().ticks == 0 and monotonic() < deadline:
            worker.publish(ControlInput())
            sleep(.01)
        snapshot = worker.snapshot()
        assert not snapshot.fault and snapshot.mode == ControlMode.ESTOP
        assert snapshot.state.velocity.linear == 0 and snapshot.motion.command.mode == 'estop'
    finally:
        release.set()
        worker.close()
