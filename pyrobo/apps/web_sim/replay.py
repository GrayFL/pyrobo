"""Replay a saved JSON case through the same backend, without a browser."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from pyrobo.apps.sim_common.session import Action
from .scene import Scene, SceneSession
from .storage import load_scene_file


def replay(scene, steps=None, *, on_step=None):
    experiment = scene.experiment
    steps = (experiment.max_steps if experiment is not None else 700) if steps is None else steps
    if type(steps) is not int or steps < 1:
        raise ValueError('steps must be a positive integer')
    session = SceneSession(scene)
    if session.nav.goal is None:
        raise ValueError('replay requires map.goal')
    session.apply(Action('mode', 'auto'))
    verified, minimum_gap = True, float('inf')
    assessments, reasons, statuses = Counter(), Counter(), Counter()
    known_stale = experiment is not None and any(e.kind == 'freeze_cloud' for e in experiment.events)
    contract_ok, all_stopping = True, True
    replans = replan_requests = no_candidates = 0
    geometric_rejections, stopped_tail = 0, 0.
    for _ in range(steps):
        s = session.tick()
        verified = verified and s.motion.safe
        m, extra = s.motion, s.motion.diagnostics.extra
        assessments[str(m.safety_assessment)] += 1
        reasons[m.diagnostics.reason] += 1
        statuses[s.navigation_status] += 1
        stale_stop = (known_stale and m.safety_assessment == 'input_stale' and
                      m.diagnostics.reason == 'map_stale' and m.command.mode == 'estop' and extra.get('stop_verified'))
        contract_ok = contract_ok and (m.safe or stale_stop)
        all_stopping = all_stopping and m.command.mode in ('estop', 'soft_stop')
        replan_requests += bool(extra.get('request_replan'))
        replans += bool(s.planning.get('replanned') and not s.planning.get('experiment_reference'))
        no_candidates += m.diagnostics.reason == 'no_feasible_candidate' and m.diagnostics.candidate_count > 0
        geometric_rejections += bool(extra.get('eb_geometry_failure') or extra.get('eb_status') in
                                     ('no_safe_rejoin', 'curvature_violation', 'collision', 'geometric_infeasible'))
        resting = max(abs(s.state.velocity.linear), abs(s.state.velocity.angular)) < 1e-3
        stopped_tail = stopped_tail+session.config['app']['dt'] if resting and m.command.mode in ('estop', 'soft_stop') else 0.
        minimum_gap = min(minimum_gap, s.motion.diagnostics.clearance_m)
        if on_step is not None:
            on_step(s, session)
        if s.collided or s.fault or s.navigation_status == 'arrived':
            break
    arrived = s.navigation_status == 'arrived'
    expected = experiment.expected if experiment is not None else 'arrived'
    stopped = max(abs(s.state.velocity.linear), abs(s.state.velocity.angular)) < 1e-3
    outcome = {'arrived': arrived, 'replan': arrived and replans > 0 and replan_requests > 0,
               'stopped': all_stopping and stopped and not arrived,
               'braking': no_candidates > 0 and all_stopping,
               'infeasible': (not arrived and stopped and stopped_tail >= 1.-1e-8 and replan_requests > 0
                              and geometric_rejections > 0 and s.navigation_status in ('blocked', 'no_path')),
               'emergency_stop': s.navigation_status == 'emergency_stop'}[expected]
    # A declared stale-cloud event can correctly cause safe=False (freshness is
    # absent). Keep that fact in all_commands_verified; separately check that
    # it caused a physically verified estop and recovered after fresh input.
    expected_met = bool(outcome and contract_ok and not s.collided and not s.fault)
    return {'name': scene.name, 'arrived': arrived, 'collided': s.collided,
            'all_commands_verified': bool(verified), 'steps': s.ticks, 'simulation_seconds': s.state.timestamp,
            'navigation_status': s.navigation_status, 'planner_status': s.planner_status,
            'position_m': s.state.pose.translation[:2].tolist(),
            'minimum_clearance_m': minimum_gap if np.isfinite(minimum_gap) else None, 'fault': s.fault,
            'expected': expected, 'expected_met': expected_met, 'assessments': dict(assessments),
            'reasons': dict(reasons), 'statuses': dict(statuses), 'events': list(session.event_log),
            'global_plans': replans, 'replan_requests': replan_requests, 'no_candidate_stops': no_candidates,
            'geometric_rejections': geometric_rejections, 'stopped_tail_seconds': stopped_tail}


def main():
    from pyrobo.apps.sim_common.catalog import discover
    discover()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path, nargs='+')
    parser.add_argument('--steps', type=int, help='override experiment.max_steps (ordinary scenes default to 700)')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    results = []
    for path in args.scene:
        try:
            results.append(replay(load_scene_file(path), args.steps))
        except (ValueError, TypeError, OSError) as error:
            results.append({'file': str(path), 'arrived': False, 'error': str(error)})
    output = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding='utf-8')
    return 0 if all(r.get('expected_met', False) for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
