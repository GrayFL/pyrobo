"""Editable solids, portable cases, real navigation and LAN transport lifecycle."""
from copy import deepcopy
import json
from pathlib import Path
from time import monotonic

import numpy as np
import pytest

pytest.importorskip('fastapi')
from fastapi.testclient import TestClient
from examples.sim_common.session import Action
from examples.web_sim.replay import replay
from examples.web_sim.scene import Polyline, Scene, SceneSession, SceneStore
from examples.web_sim.server import create_app
from examples.web_sim.world import PolygonWorld
from pyrobo import Footprint, pose2d
from pyrobo.core import Bounds2D
from pyrobo.simulation import World2D


def scene(**backend):
    return Scene(name='可回放 / 测试', config={'backend': backend,
                 'map': {'preset': 'custom', 'bounds': [-4, 4, -3, 3],
                         'start': [-2, 0, 0], 'goal': [1.5, 0, 0], 'boxes': []}},
                 polygons=[[[0, 1.2], [1, 1.2], [.8, 2.3], [0, 2]]],
                 polylines=[Polyline(points=[[-2.5, 1.], [-2.5, 2.], [-1.5, 2.]], width=.2)])


def test_scene_round_trip_freezes_parameters_and_retains_original_start(tmp_path):
    case = scene(model='ackermann', planner='voronoi')
    assert case.config['voronoi']['time_budget'] == 2.
    assert case.config['dwa']['prediction_time'] == 2.4
    store = SceneStore(tmp_path)
    identifier = store.save(case)
    assert store.load(identifier).model_dump() == case.model_dump()
    assert store.list()[0]['name'] == case.name
    assert store.save(case) != identifier
    for invalid in ('../outside', '123', 'a'*32+'.json'):
        with pytest.raises(ValueError):
            store.load(invalid)
    run = SceneSession(case)
    run.tick(actions=[Action('goal', (2., -.5, 0.))])
    run.tick(actions=[Action('reset')])
    assert run.nav.goal.translation[1] == -.5
    np.testing.assert_array_equal(case.config['map']['start'], [-2, 0, 0])
    assert run.world.polygons


@pytest.mark.parametrize('edit', [
    {'schema_version': 2}, {'polygons': [[[0, 0], [1, 1], [0, 1], [1, 0]]]},
    {'polygons': [[[-2.5, -.5], [-1.5, -.5], [-1.5, .5], [-2.5, .5]]]},
    {'polygons': [[[0, 0], [10, 0], [10, 1]]]}, {'config': {'app': {'point_spacing': .000001}}},
    {'config': {'backend': {'model': 'typo'}}}, {'config': {'dynamics': {'mass': float('nan')}}},
    {'config': {'dwa': {'linear_samples': 1000000}}},
])
def test_invalid_scene_rejected(edit):
    payload = scene().model_dump()
    payload.update(edit)
    with pytest.raises((ValueError, TypeError)):
        Scene.model_validate(payload)


def test_concave_solid_truth_handles_containment_and_notch():
    polygon = [[-2, -2], [2, -2], [2, 2], [1, 2], [1, -1], [-1, -1], [-1, 2], [-2, 2]]
    world = PolygonWorld(World2D(Bounds2D(-4, 4, -4, 4)), [polygon])
    body = Footprint(.3, .3, 0.)
    assert not world.collides(pose2d(0, 0), body)  # concave notch, inside convex hull
    assert world.collides(pose2d(1.5, 0), body)    # entirely inside solid, away from edges
    assert world.collides(pose2d(.9, 0, .7), body)
    assert world.collides(pose2d(0, 0), Footprint(6, 6, 0.))  # solid contained by robot
    cloud = world.point_cloud(.04)
    assert len(cloud.points) > 500


def test_polyline_thickness_round_caps_and_crossing_truth():
    world = PolygonWorld(World2D(Bounds2D(-4, 4, -4, 4)), [],
                         [Polyline(points=[[-1, 0], [1, 0], [1, 1]], width=.4)])
    body = Footprint(.1, .1, 0.)
    assert world.collides(pose2d(0, .24), body)
    assert not world.collides(pose2d(0, .26), body)
    assert world.collides(pose2d(-1.24, 0), body)  # round cap beyond the endpoint
    assert not world.collides(pose2d(-1.26, 0), body)
    assert world.collides(pose2d(0, 0, .8), Footprint(.5, .5, 0.))
    with pytest.raises(ValueError):
        Polyline(points=[[0, 0], [0, 0]])


@pytest.mark.parametrize('model', ['differential', 'ackermann'])
@pytest.mark.parametrize('planner', ['rrt_connect', 'voronoi'])
def test_saved_scene_replay_arrives_with_real_pipeline(model, planner, tmp_path):
    store = SceneStore(tmp_path)
    identifier = store.save(scene(model=model, planner=planner))
    result = replay(store.load(identifier), steps=160)
    assert result['arrived'] and result['all_commands_verified'] and not result['collided'], result


@pytest.mark.parametrize('kind', ['polygon', 'polyline'])
def test_drawn_obstacle_requires_detour_and_still_arrives(kind):
    payload = scene().model_dump()
    payload['polygons'] = [[[-.4, -1], [.4, -1], [.4, 1], [-.4, 1]]] if kind == 'polygon' else []
    if kind == 'polyline':
        payload['polylines'] = [{'points': [[0, -1], [0, 1]], 'width': .8}]
    session = SceneSession(Scene.model_validate(payload))
    session.apply(Action('mode', 'auto'))
    for _ in range(300):
        snapshot = session.tick()
        assert snapshot.motion.safe and not snapshot.collided
        if snapshot.navigation_status == 'arrived':
            break
    assert snapshot.navigation_status == 'arrived'
    assert np.max(np.abs(snapshot.trail[:, 1])) > 1.2


def receive(ws, predicate, limit=100):
    for _ in range(limit):
        data = ws.receive_json()
        if predicate(data):
            return data
    raise AssertionError('expected WebSocket update not received')


def reply(ws, identifier):
    return receive(ws, lambda data: data.get('type') == 'reply' and data.get('id') == identifier)


def test_http_case_storage_and_validation(tmp_path):
    with TestClient(create_app(scenes_dir=tmp_path, initial_scene=scene())) as client:
        assert client.get('/').status_code == 200
        assert client.get('/static/app.js').status_code == 200
        original = client.get('/api/bootstrap').json()['scene']
        response = client.post('/api/scenes', json=original)
        assert response.status_code == 201
        identifier = response.json()['id']
        assert client.get(f'/api/scenes/{identifier}').json() == original
        assert len(client.get('/api/scenes').json()) == 1
        assert client.get('/api/scenes/not-an-id').status_code == 422
        assert client.get('/api/scenes/'+'0'*32).status_code == 404
        invalid = deepcopy(original)
        invalid['config']['map']['start'] = [20, 0, 0]
        assert client.post('/api/scenes', json=invalid).status_code == 422
        assert client.post('/api/scenes', content='x'*2_000_001).status_code == 413
        assert len(client.get('/api/scenes').json()) == 1


def test_websocket_isolation_stale_input_estop_and_scene_apply(tmp_path):
    with TestClient(create_app(scenes_dir=tmp_path, initial_scene=scene())) as client:
        with client.websocket_connect('/ws') as first, client.websocket_connect('/ws') as second:
            assert first.receive_json()['type'] == 'hello'
            assert second.receive_json()['type'] == 'hello'
            deadline = monotonic()+1.5
            while monotonic() < deadline:
                first.send_json({'type': 'control', 'input': {'throttle': 1., 'engaged': True}})
                moving = receive(first, lambda data: data['type'] == 'state')
            assert moving['mode'] == 'manual' and moving['pose'][0] > -1.9
            isolated = receive(second, lambda data: data['type'] == 'state' and data['ticks'] >= 3)
            assert isolated['pose'][0] == -2 and isolated['mode'] == 'paused'
            stopped = receive(first, lambda data: data['type'] == 'state' and data['mode'] == 'paused')
            assert stopped['ticks'] > moving['ticks']  # expired heartbeat brakes without new input
            first.send_json({'type': 'action', 'id': 1, 'action': 'estop'})
            assert reply(first, 1)['ok']
            first.send_json({'type': 'control', 'input': {'throttle': 1., 'engaged': True}})
            first.send_json({'type': 'action', 'id': 2, 'action': 'auto'})
            assert reply(first, 2)['ok']
            latched = receive(first, lambda data: data['type'] == 'state' and data['mode'] == 'estop')
            assert latched['mode'] == 'estop'
            first.send_json({'type': 'action', 'id': 3, 'action': 'goal', 'value': [0.4, 1.5, 0]})
            assert not reply(first, 3)['ok']
            first.send_json({'type': 'control', 'id': 4, 'input': {'throttle': 100}})
            assert not reply(first, 4)['ok']
            changed = scene(model='ackermann', planner='voronoi').model_dump()
            changed['config']['map']['start'] = [-3, -.5, .2]
            first.send_json({'type': 'apply', 'id': 5, 'scene': changed})
            result = reply(first, 5)
            assert result['ok'] and result['generation'] == 2
            reset = receive(first, lambda data: data['type'] == 'state' and data['generation'] == 2)
            np.testing.assert_allclose(reset['pose'], [-3, -.5, .2])
            assert reset['mode'] == 'paused'
            second.send_json({'type': 'action', 'id': 6, 'action': 'reset'})
            assert reply(second, 6)['ok']


def test_cross_origin_websocket_is_rejected(tmp_path):
    from starlette.websockets import WebSocketDisconnect
    with TestClient(create_app(scenes_dir=tmp_path, initial_scene=scene())) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/ws', headers={'origin': 'http://elsewhere.example'}):
                pass


def test_narrow_ackermann_saved_case_reaches_goal_without_reducing_body_margin():
    path = Path(__file__).resolve().parents[1]/'examples/scenes/narrow_ackermann_001.json'
    case = Scene.model_validate_json(path.read_bytes())
    assert case.config['footprint'] == {'length': .5, 'width': .5, 'margin': .08, 'offset_x': 0.}
    assert case.config['navigation']['goal_approach_distance'] == 1.
    result = replay(case, steps=330)
    assert result['arrived'] and result['all_commands_verified'] and not result['collided'], result


@pytest.mark.parametrize('planner', ['rrt_connect', 'voronoi'])
def test_narrow_ackermann_adapts_original_extra_clearance_and_entry_length(planner):
    path = Path(__file__).resolve().parents[1]/'examples/scenes/narrow_ackermann_001.json'
    data = json.loads(path.read_text())
    config = data['config']
    config['backend'].update(planner=planner, adaptive_clearance=True)
    config[planner]['clearance'] = .4056296013484
    config['navigation']['goal_approach_distance'] = None
    case = Scene.model_validate(data)
    result = replay(case, steps=350)
    assert result['arrived'] and result['all_commands_verified'] and not result['collided'], result
    assert case.config['footprint']['margin'] == .08
