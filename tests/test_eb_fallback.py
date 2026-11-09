"""Finite topology attempts, real global fallback, and portable fault episodes."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from pyrobo import EBDWAConfig, EBPathRepairer, ChassisState, RobotObservation, pose2d
from pyrobo.core import Bounds2D, Path as Route
from pyrobo.core.geometry import path_poses
from pyrobo.simulation import Box, World2D
from pyrobo.apps.sim_common.session import Action
from pyrobo.apps.web_sim.scene import Scene, SceneSession
from pyrobo.apps.web_sim.replay import replay

CASES = Path(__file__).resolve().parents[1]/'pyrobo/apps/scenes'


def case(name, logical=True):
    data = json.loads((CASES/f'eb_dwa_{name}.json').read_text())
    if logical and name != 'budget':
        data['config'].setdefault('eb_dwa', {})['time_budget_s'] = None
    return Scene.model_validate(data)


def test_only_failed_nominal_starts_bounded_left_then_right_topologies():
    world = World2D(Bounds2D(-2, 12, -4, 4), [Box(3, 4, -.5, .5), Box(2, 5, .8, 3.)])
    observation = RobotObservation(ChassisState(pose2d(0, 0)), world.point_cloud(), world.bounds)
    route = Route(path_poses([[0, 0], [9, 0]]))
    config = EBDWAConfig(time_budget_s=None, max_evaluations=3000, topology_attempts=3)
    repairer = EBPathRepairer(config)
    result = repairer.update(observation, route)
    attempts = result.diagnostics['eb_attempts']
    assert [a['topology'] for a in attempts] == ['nominal', 'left', 'right']
    assert all(not a['validated'] for a in attempts[:-1])
    assert result.success and result.reference.xy[:, 1].min() < -.7
    assert len(attempts) <= config.topology_attempts
    assert sum(a['evaluations'] for a in attempts) <= config.max_evaluations
    assert all(a['duration_s'] >= 0 and a['iterations'] <= config.max_iterations for a in attempts)
    disabled = EBPathRepairer(replace(config, fallback_enabled=False)).update(observation, route)
    assert not disabled.success and disabled.reference is None
    assert [a['topology'] for a in disabled.diagnostics['eb_attempts']] == ['nominal']


def test_u_trap_requests_real_global_search_and_retains_new_full_route():
    session = SceneSession(case('u_trap'))
    session.nav.planner.planner.config = replace(session.nav.planner.planner.config, time_budget=None)
    first = session.tick(actions=[Action('mode', 'auto')])
    assert first.motion.safe and first.motion.command.mode == 'estop'
    assert first.motion.diagnostics.extra['request_replan']
    assert first.reference is None
    for _ in range(6):
        second = session.tick()
        if second.planning.get('replanned'):
            break
        assert second.motion.command.mode == 'estop' and second.motion.safe
    assert second.planning['replanned'] and not second.planning.get('experiment_reference')
    assert second.state.timestamp-session.config['app']['dt'] >= session.nav.retry_interval-1e-9
    assert second.planning['extensions'] > 0
    route = session.nav.path.xy.copy()
    assert route[:, 0].max() > 6.  # must exit the U to the east before going west
    for _ in range(4):
        current = session.tick()
        assert current.motion.safe and not current.collided
    np.testing.assert_array_equal(session.nav.path.xy, route)


@pytest.mark.parametrize('name', ['budget', 'no_candidate'])
def test_fault_window_enforces_verified_stop_without_claiming_arrival(name):
    records = []
    result = replay(case(name), on_step=lambda s, _: records.append(s))
    assert result['expected_met'] and not result['arrived'] and not result['collided']
    assert result['all_commands_verified']
    assert all(s.motion.command.mode == 'estop' for s in records)
    if name == 'budget':
        assert len(records) == 8 and np.max(np.linalg.norm(records[-1].trail, axis=1)) == 0
        assert all(s.reference is None and s.motion.diagnostics.candidate_count == 0 for s in records)
        assert records[0].motion.diagnostics.extra['eb_status'] == 'time_budget'
        assert records[0].motion.diagnostics.extra['eb_geometry_failure'] == 'collision'
    else:
        assert records[0].reference is not None  # geometry passed, dynamics failed
        assert result['no_candidate_stops'] == 1
        assert records[0].motion.diagnostics.candidate_count > 0
        assert records[0].motion.diagnostics.feasible_count == 0
        assert 0 < records[0].state.velocity.linear < .6  # finite-force braking, not teleporting to rest


def test_saved_stale_cloud_stops_and_recovers_without_claiming_freshness():
    records = []
    result = replay(case('stale_cloud'), on_step=lambda s, _: records.append(s))
    assert result['expected_met'] and result['arrived'] and not result['collided']
    assert not result['all_commands_verified']
    stale = [s for s in records if s.motion.safety_assessment == 'input_stale']
    assert stale and all(s.motion.command.mode == 'estop' and s.motion.diagnostics.reason == 'map_stale' for s in stale)
    assert all(s.motion.diagnostics.extra['stop_verified'] for s in stale)
    assert records[-1].motion.safe and not records[-1].experiment['cloud_frozen']
    assert len(result['events']) == 2


@pytest.mark.parametrize('name', ['long_wall', 'two_obstacles', 'u_trap', 'dynamic_obstacle'])
def test_saved_repair_and_search_cases_reach_with_real_world_checks(name, tmp_path):
    records = []
    def record(s, session):
        records.append({'pose': s.state.pose.translation[:2].tolist(), 'safe': s.motion.safe,
                        'status': s.navigation_status, 'extra': s.motion.diagnostics.extra,
                        'planning': s.planning, 'events': s.experiment})
    result = replay(case(name), on_step=record)
    (tmp_path/f'{name}.json').write_text(json.dumps({'result': result, 'steps': records}, default=float, allow_nan=False))
    assert result['expected_met'] and result['arrived'], result
    assert result['all_commands_verified'] and not result['collided']
    if name == 'dynamic_obstacle':
        assert len(result['events']) == 1
        after = records[next(i for i, r in enumerate(records) if r['events']['events']):]
        assert any(not r['extra'].get('eb_cache_hit', False) for r in after[:3])
