"""Whole-controller integration, using the portable Web cases and real plant."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from pyrobo import (Ackermann, DifferentialDrive, EBDWAConfig, ElasticBandDWA, Footprint,
                    ChassisState, MotionPolicy, MotionRequest, Navigator, RobotObservation,
                    PointCloud, Twist2D, pose2d)
from pyrobo.apps.sim_common.session import Action
from pyrobo.apps.web_sim.scene import Scene, SceneSession, SceneStore
from pyrobo.apps.web_sim.server import create_app, snapshot_data
from pyrobo.core import Path as Route
from pyrobo.core.geometry import path_poses

CASES = Path(__file__).resolve().parents[1]/'pyrobo/apps/scenes'


def scene(name, *, logical=True):
    payload = json.loads((CASES/f'eb_dwa_{name}.json').read_text())
    if logical and name != 'budget':
        payload['config'].setdefault('eb_dwa', {})['time_budget_s'] = None
    return Scene.model_validate(payload)


@pytest.mark.parametrize('path', sorted(CASES.glob('eb_dwa_*.json')), ids=lambda p: p.stem)
def test_portable_cases_freeze_and_restore_every_experiment_field(path, tmp_path):
    case = Scene.model_validate_json(path.read_bytes())
    store = SceneStore(tmp_path)
    restored = store.load(store.save(case))
    assert restored.model_dump() == case.model_dump()
    assert isinstance(SceneSession(restored).nav.optimizer, ElasticBandDWA)


@pytest.mark.parametrize('model', [DifferentialDrive(), Ackermann()])
def test_native_command_shared_predictor_and_actual_feedback(model):
    nav = Navigator(model=model, footprint=Footprint.long(offset_x=.25), controller=EBDWAConfig(time_budget_s=None))
    engine = nav.optimizer
    assert isinstance(engine, ElasticBandDWA)
    assert nav.safety.predictor.dynamics is engine.plant
    case = SceneSession(scene('straight'))
    o = RobotObservation(ChassisState(pose2d(0, 0)), case.cloud, case.world.bounds)
    request = MotionRequest(MotionPolicy.NAVIGATION, 0.,
                            path=Route(path_poses([[0, 0], [4, 0]])), goal=pose2d(4, 0))
    first = engine.optimize(o, request)
    assert first.safe and first.diagnostics.feasible_count > 0
    assert not first.diagnostics.extra['eb_execution_feedback']
    state = engine.plant.step(o.chassis, first.command, o.dt).state
    # Prediction's first cycle is the same finite-force integration as execution.
    index = np.flatnonzero(np.isclose(first.trajectory.times, o.dt))[0]
    np.testing.assert_allclose(first.trajectory.states[index], engine.plant.codec.pack(state), atol=1e-12)
    engine.record_execution(first.command, o.timestamp)
    second = engine.optimize(o, request)
    assert second.diagnostics.extra['eb_execution_feedback']
    assert 'control' in second.diagnostics.cost_terms and 'cross_track' in second.diagnostics.cost_terms
    assert nav.safety.enforce(o, second).safe
    engine.reset()
    assert engine.last_reference is None and engine._executed is None


def test_manual_delegation_preserves_index_and_does_not_repair_navigation():
    session = SceneSession(scene('straight'))
    engine = session.nav.optimizer
    observation = session.observation()
    request = MotionRequest(MotionPolicy.TELEOP_STOP, 0., desired=Twist2D(.3, .2))
    index = engine.index.get(session.cloud)
    for _ in range(2):
        result = engine.optimize(observation, request)
        assert result.safe and result.diagnostics.extra['optimizer'] == 'dwa'
        assert engine.index.get(session.cloud) is index
        assert engine.last_reference is None


def test_geometry_budget_failure_never_enters_dwa_and_stale_input_clears_reference(monkeypatch):
    session = SceneSession(scene('budget'))
    engine = session.nav.optimizer
    monkeypatch.setattr(engine._engine, 'optimize', lambda *a, **kw: pytest.fail('unverified geometry reached DWA'))
    snapshot = session.tick(actions=[Action('mode', 'auto')])
    assert snapshot.motion.safe and snapshot.motion.command.mode == 'estop'
    assert snapshot.motion.diagnostics.candidate_count == 0
    assert engine.last_reference is None
    ordinary = SceneSession(scene('straight'))
    ordinary.tick(actions=[Action('mode', 'auto')])
    o = ordinary.observation()
    o = replace(o, obstacles=PointCloud(o.obstacles.points, timestamp=o.timestamp-1.))
    request = MotionRequest(MotionPolicy.STOP, o.timestamp)
    result = ordinary.nav.optimizer.optimize(o, request)
    assert result.safety_assessment == 'input_stale'
    assert ordinary.nav.optimizer.last_reference is None


def test_chunking_preserves_spatial_winner_and_cost_terms():
    session = SceneSession(scene('corner'))
    o = session.observation()
    route = Route(path_poses(session.scene.experiment.reference_path))
    request = MotionRequest(MotionPolicy.NAVIGATION, 0., path=route, goal=session.nav.goal)
    results = [ElasticBandDWA(EBDWAConfig(time_budget_s=None, batch_size=size)).optimize(o, request)
               for size in (0, 11)]
    assert results[0].command == results[1].command
    assert results[0].diagnostics.cost_terms == results[1].diagnostics.cost_terms
    assert results[0].diagnostics.candidate_count == results[1].diagnostics.candidate_count


def test_saved_events_change_both_collision_world_and_observation_then_reset():
    case = scene('dynamic_obstacle').model_dump()
    case['experiment']['events'] = [
        {'time_s': 0., 'kind': 'add_boxes', 'boxes': [[4, 5, -.5, .5]]},
        {'time_s': .1, 'kind': 'freeze_cloud'}, {'time_s': .6, 'kind': 'resume_cloud'}]
    session = SceneSession(Scene.model_validate(case))
    original = session.cloud
    first = session.tick()
    assert session.cloud is not original and len(session.world.boxes) == 1
    assert session.sim.world is session.world
    for _ in range(7):
        snapshot = session.tick()
    assert len(session.event_log) == 3 and not session.cloud_frozen
    assert session.cloud.timestamp is None and not snapshot.collided
    assert first.experiment['events'] == [{'kind': 'add_boxes', 'scheduled_s': 0., 'applied_s': 0.}]
    session.apply(Action('reset'))
    assert not session.event_log and not session.world.boxes
    assert not session.nav.planner.used


def test_live_goal_change_does_not_restore_previous_authored_route_on_reset():
    session = SceneSession(scene('straight'))
    session.apply(Action('goal', [8., 1., 0.]))
    session.apply(Action('reset'))
    assert session.scene.experiment.reference_path is None
    np.testing.assert_array_equal(session.nav.goal.translation[:2], [8., 1.])


@pytest.mark.parametrize('edit', [
    {'events': [{'time_s': 1., 'kind': 'add_boxes', 'boxes': []}]},
    {'events': [{'time_s': 1., 'kind': 'freeze_cloud'}, {'time_s': 0., 'kind': 'resume_cloud'}]},
    {'events': [{'time_s': 0., 'kind': 'add_boxes', 'boxes': [[100, 101, 0, 1]]}]},
    {'initial_state': {'linear': float('nan')}}, {'max_steps': 100000},
    {'reference_path': [[1, 0], [9, 0]]},
])
def test_invalid_experiment_fails_before_start(edit):
    payload = scene('straight').model_dump()
    payload['experiment'].update(edit)
    with pytest.raises(ValueError):
        Scene.model_validate(payload)


@pytest.mark.parametrize('name,limit', [('straight', 160), ('corner', 240), ('differential_spin', 200),
                                     ('ackermann_reverse', 130)])
def test_real_closed_loop_basic_maneuvers(name, limit):
    session = SceneSession(scene(name))
    session.apply(Action('mode', 'auto'))
    references = 0
    for _ in range(limit):
        snapshot = session.tick()
        assert snapshot.motion.safe and not snapshot.collided and not snapshot.fault
        references += int(snapshot.reference is not None)
        if snapshot.navigation_status == 'arrived':
            break
    assert snapshot.navigation_status == 'arrived'
    assert references > 20
    assert max(abs(snapshot.state.velocity.linear), abs(snapshot.state.velocity.angular)) < 1e-3
    if name == 'ackermann_reverse':
        assert np.max(abs(snapshot.trail[:, 1])) < .05  # no terminal steering to waste distance


def test_web_builtins_and_json_diagnostics(tmp_path):
    from fastapi.testclient import TestClient
    case = scene('corner')
    session = SceneSession(case)
    payload = snapshot_data(session.tick(actions=[Action('mode', 'auto')]), 2)
    assert payload['reference']['xy'] and payload['reference']['left']
    assert payload['cost_terms'] and payload['optimizer'] == 'eb_dwa'
    json.dumps(payload, allow_nan=False)
    with TestClient(create_app(scenes_dir=tmp_path, initial_scene=case)) as client:
        examples = client.get('/api/examples').json()
        assert len([x for x in examples if x['id'].startswith('eb_dwa_')]) == 20
        result = client.get('/api/examples/eb_dwa_corner')
        assert result.status_code == 200 and result.json()['experiment']['reference_path']
        assert client.get('/api/examples/does_not_exist').status_code == 404
        assert 'value="eb_dwa"' in client.get('/').text
