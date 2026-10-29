"""Real optional RSim process graph, not a mocked multiprocessing adapter."""
import asyncio
from dataclasses import replace
import importlib.util

import pytest

pytest.importorskip('rsim')
pytest.importorskip('cloudpickle')

from pyrobo import Ackermann, DWAConfig, NavigationConfig, pose2d
from pyrobo.integrations.rsim import StackConfig
from examples.navigation_benchmark.scenarios import scenarios
from examples.navigation_benchmark.rsim_platform import run


@pytest.mark.parametrize('processes', [False, True])
def test_rsim_graph_drives_native_ackermann_commands_and_closes(processes):
    if processes and importlib.util.find_spec('cyclonedds') is None:
        pytest.skip('Cyclone DDS required for ProcessPlacement')
    case = replace(scenarios()['straight'], goal=pose2d(3.5, 6, 0))
    cfg = StackConfig(model=Ackermann(), controller=DWAConfig(prediction_time=2.4, lookahead=1.6),
                      navigation=NavigationConfig(lookahead=1.6, goal_approach_distance=1.))
    report = asyncio.run(run(case, cfg, duration=15., processes=processes))
    assert report['arrived'] and not report['collided'], {k: report[k] for k in ('pids', 'last_execution', 'maximum_speed_mps')}
    ids = {report['pids']['simulation']}
    for key in ('global', 'local', 'execution'):
        assert report['pids'][key]
        ids.update(report['pids'][key])
    assert len(ids) == (4 if processes else 1)
    assert report['last_execution']['accepted'] > 2


def test_rsim_400ms_optimizer_stall_does_not_freeze_execution_or_accept_late_result():
    pytest.importorskip('cyclonedds')
    case = replace(scenarios()['straight'], goal=pose2d(3.5, 6, 0))
    report = asyncio.run(run(case, StackConfig(), duration=18., processes=True,
                             control_delay_every=8, control_delay_s=.4))
    assert report['arrived'] and not report['collided']
    delayed = [c for c in report['controls'] if c['compute_ms'] >= 400]
    assert delayed and report['last_execution']['rejected'] >= 1
    assert any(e['last_rejection'] == 'proposal_stale' for e in report['executions'])
    # There were fresh executor decisions during an actual blocked optimizer job.
    assert any(c['observed_at'] < e['timestamp'] < c['completed_at']
               for c in delayed for e in report['executions'])


@pytest.mark.parametrize('kind', ['sensor', 'map'])
def test_rsim_stale_sensor_or_map_stops_and_recovers_without_freezing_truth(kind):
    pytest.importorskip('cyclonedds')
    case = replace(scenarios()['straight'], goal=pose2d(3.5, 6, 0))
    options = {f'{kind}_pause_period': 2.5, f'{kind}_pause_s': .55}
    report = asyncio.run(run(case, StackConfig(), duration=20., processes=True, **options))
    assert report['arrived'] and not report['collided']
    reason = 'state_stale' if kind == 'sensor' else 'map_stale'
    assert report['execution_reasons'].get(reason, 0) > 0
    stale = [e for e in report['executions'] if e['reason'] == reason]
    assert all(e['mode'] == 'estop' and not e['safe'] for e in stale)
    assert report['last_execution']['timestamp'] > stale[-1]['timestamp']


def test_rsim_narrow_ackermann_arrives_despite_repeated_400ms_stalls():
    pytest.importorskip('cyclonedds')
    config = StackConfig(model=Ackermann())
    report = asyncio.run(run(scenarios()['narrow_ackermann'], config, duration=75., processes=True,
                             control_delay_every=8, control_delay_s=.4))
    assert report['arrived'] and not report['collided'], report['samples'][-1]
    assert report['minimum_margin_clearance_m'] > 0
    assert report['last_execution']['rejected'] > 0
    assert report['execution_reasons'].get('proposal_timeout_hold_steering', 0) > 0
