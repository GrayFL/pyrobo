"""Replay portable Web experiments, retaining successes, failures and per-tick evidence.

Run serially for timings. The synchronous simulator waits for planning and does
not prove realtime execution. Metrics and raw scenes are saved with provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pyrobo.apps.web_sim.replay import replay
from pyrobo.apps.web_sim.scene import Scene
from pyrobo.core.geometry import planar_state, path_poses
from pyrobo.navigation.tracking import PathTracker
from .planning import provenance


CASES = Path(__file__).resolve().parents[2]/'pyrobo/apps/scenes'


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def distribution(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    return dict(zip(('p50', 'p95', 'max'), map(float, [*np.percentile(values, [50, 95]), max(values)])))


def measure(scene):
    rows, tracker = [], PathTracker()
    previous_path = None
    initial = scene.config['map']['start']

    def record(snapshot, session):
        nonlocal previous_path
        state, motion = snapshot.state, snapshot.motion
        diag, extra = motion.diagnostics, motion.diagnostics.extra
        xy = session.nav.path.poses.position[:, :2]
        if len(xy) and (previous_path is None or not np.array_equal(xy, previous_path)):
            tracker.set_path(path_poses(xy))
            previous_path = xy.copy()
        error, bend_distance = None, None
        if len(xy) > 1:
            progress = tracker.update(state.pose, 0.)
            q = np.column_stack([np.interp(progress.distance, tracker.cumulative, xy[:, axis]) for axis in range(2)])[0]
            error = float(np.linalg.norm(planar_state(state.pose)[:2]-q))
            heading = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
            turns = abs(np.diff(np.unwrap(heading))) > .25
            distances = tracker.cumulative[1:-1][turns]-progress.distance
            future = distances[distances >= 0.]
            bend_distance = float(future.min()) if len(future) else None
        ref = snapshot.reference
        reference_error = None
        if ref is not None and len(ref.xy) > 1:
            from pyrobo.control.eb_dwa.scoring import project_trajectories
            p = project_trajectories(ref, planar_state(state.pose)[None, None, :2])
            reference_error = abs(float(p.lateral[0, 0]))
        poses = planar_state(state.pose)
        gap = session.nav.optimizer.index.get(session.cloud).clearances(poses[None], session.footprint, session.world.bounds)[0]
        rows.append(clean({'time_s': state.timestamp, 'pose': poses, 'v': state.velocity.linear,
            'w': state.velocity.angular, 'steering': state.steering, 'steering_rate': state.steering_rate,
            'acceleration': state.acceleration, 'safe': motion.safe, 'assessment': str(motion.safety_assessment),
            'command': session.model.encode_command(motion.command), 'command_mode': motion.command.mode,
            'status': snapshot.navigation_status, 'reason': diag.reason, 'collided': snapshot.collided,
            'actual_margin_clearance_m': gap, 'rollout_clearance_m': diag.clearance_m,
            'global_cross_track_m': error, 'eb_cross_track_m': reference_error,
            'next_global_bend_m': bend_distance, 'local_ms': 1000*diag.elapsed_s,
            'tick_ms': snapshot.elapsed_ms, 'candidates': diag.candidate_count, 'feasible': diag.feasible_count,
            'costs': diag.cost_terms, 'extra': extra, 'planning': snapshot.planning}))

    started = perf_counter()
    outcome = replay(scene, on_step=record)
    elapsed = perf_counter()-started
    poses = np.asarray([initial]+[r['pose'] for r in rows])
    length = float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum())
    v, w = np.asarray([[r['v'], r['w']] for r in rows]).T
    moving = abs(v) >= .05
    curvature = np.divide(w, v, out=np.zeros_like(v), where=moving)
    travel = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    rate_valid = moving[1:] & moving[:-1] & (travel[1:] > .005)
    rate = np.divide(np.diff(curvature), travel[1:], out=np.zeros(len(rows)-1), where=rate_valid)
    straight, turning = moving & (abs(curvature) < .15), moving & (abs(curvature) >= .4)
    error = np.asarray([r['global_cross_track_m'] for r in rows if r['global_cross_track_m'] is not None])
    eb_error = np.asarray([r['eb_cross_track_m'] for r in rows if r['eb_cross_track_m'] is not None])
    braking = [r['next_global_bend_m'] for r in rows if r['acceleration']*r['v'] < -.025
               and r['next_global_bend_m'] is not None and abs(r['v']) >= .1]
    extras = [r['extra'] for r in rows]
    count = max(1, len(rows))
    metrics = {
        'wall_seconds_with_metrics': elapsed, 'length_m': length,
        'actual_minimum_margin_clearance_m': min(r['actual_margin_clearance_m'] for r in rows),
        'curvature_max_abs_per_m': float(max(abs(curvature))),
        'curvature_rate_max_abs_per_m2': float(max(abs(rate), default=0.)),
        'straight_mean_speed_mps': float(abs(v[straight]).mean()) if straight.any() else None,
        'corner_min_speed_mps': float(abs(v[turning]).min()) if turning.any() else None,
        'first_braking_distance_to_global_bend_m': braking[0] if braking else None,
        'global_cross_track_rms_m': float(np.sqrt(np.mean(error**2))) if len(error) else None,
        'global_cross_track_peak_m': float(error.max()) if len(error) else None,
        'eb_cross_track_rms_m': float(np.sqrt(np.mean(eb_error**2))) if len(eb_error) else None,
        'eb_cross_track_peak_m': float(eb_error.max()) if len(eb_error) else None,
        'steering_rate_max_abs_rps': max(abs(r['steering_rate']) for r in rows),
        'local_ms': distribution([r['local_ms'] for r in rows]),
        'tick_ms': distribution([r['tick_ms'] for r in rows]),
        'eb_ms': distribution([1000*x['eb_duration_s'] for x in extras if 'eb_duration_s' in x]),
        'dwa_ms': distribution([1000*x['dwa_duration_s'] for x in extras if 'dwa_duration_s' in x]),
        'speed_profile_ms': distribution([1000*x['eb_speed_profile_duration_s'] for x in extras if 'eb_speed_profile_duration_s' in x]),
        'global_ms': distribution([1000*r['planning']['elapsed'] for r in rows if 'elapsed' in r['planning']]),
        'eb_cache_hit_fraction': sum(bool(x.get('eb_cache_hit')) for x in extras)/count,
        'eb_warm_start_fraction': sum(bool(x.get('eb_warm_started')) for x in extras)/count,
        'eb_optimization_fraction': sum(bool(x.get('eb_optimized')) for x in extras)/count,
        'eb_nodes': distribution([x['eb_nodes'] for x in extras if 'eb_nodes' in x]),
        'eb_active_nodes': distribution([x['eb_active_nodes'] for x in extras if 'eb_active_nodes' in x]),
        'candidates': distribution([r['candidates'] for r in rows]),
        'feasible_fraction': sum(r['feasible'] for r in rows)/max(1, sum(r['candidates'] for r in rows)),
    }
    return clean({**outcome, 'metrics': metrics}), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='+', default=['narrow_road'])
    parser.add_argument('--optimizers', nargs='+', choices=['dwa', 'teb', 'eb_dwa'], default=['eb_dwa'])
    parser.add_argument('--planners', nargs='+', choices=['rrt_connect', 'voronoi'], default=['rrt_connect'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[7])
    parser.add_argument('--logical', action='store_true', help='disable EB and RRT wall budgets; preserve fault budget scene')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = [p.stem.removeprefix('eb_dwa_') for p in sorted(CASES.glob('eb_dwa_*.json'))] if args.scenes == ['all'] else args.scenes
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 1, 'environment': provenance(), 'logical': args.logical, 'runs': []}
    for name in names:
        for optimizer in args.optimizers:
            for planner in args.planners:
                for seed in args.seeds:
                    key = f'{name}-{optimizer}-{planner}-{seed}'
                    path = CASES/f'eb_dwa_{name}.json'
                    data = json.loads(path.read_bytes())
                    data['config']['backend'].update(local_optimizer=optimizer, planner=planner)
                    data['config'].setdefault('rrt_connect', {}).update(seed=seed)
                    if args.logical:
                        data['config']['rrt_connect']['time_budget'] = None
                        if name != 'budget':
                            data['config'].setdefault('eb_dwa', {})['time_budget_s'] = None
                    scene = Scene.model_validate(data)
                    scene_json = scene.model_dump_json(indent=2)
                    (args.output/f'{key}.scene.json').write_text(scene_json)
                    result, rows = measure(scene)
                    result.update(case=name, optimizer=optimizer, planner=planner, seed=seed,
                                  scene_sha256=hashlib.sha256(scene_json.encode()).hexdigest())
                    (args.output/f'{key}.trace.json').write_text(json.dumps(rows, allow_nan=False))
                    report['runs'].append(result)
                    (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
                    print(json.dumps({k: result[k] for k in ('case', 'optimizer', 'planner', 'seed', 'arrived',
                        'expected_met', 'steps', 'collided', 'global_plans')}, ensure_ascii=False), flush=True)
    return 0 if all(r['expected_met'] for r in report['runs']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
