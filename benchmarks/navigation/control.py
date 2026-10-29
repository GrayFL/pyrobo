"""Fast deterministic closed loops; wall-clock fault tests use rsim_platform.

This runner intentionally reports synchronous simulated time. Its measured
compute latency is real, but a slow compute does not advance its physical clock.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from pyrobo import Ackermann, DifferentialDrive, DWAConfig, Footprint, NavigationConfig, Navigator
from pyrobo.core.geometry import planar_state, wrap_angle
from pyrobo.geometry.collision import PointObstacles
from pyrobo.navigation.tracking import polyline_distances
from pyrobo.planning import AdaptiveClearancePlanner, RRTConnect, RRTConnectConfig, VoronoiConfig, VoronoiPlanner
from pyrobo.simulation import run_navigation
from .planning import provenance
from .scenarios import scenarios


def trial(case, algorithm, model_name, *, seed=0, steps=900, lookahead_scale=1., body='compact', max_speed=.9):
    model = (Ackermann if model_name == 'ackermann' else DifferentialDrive)(max_speed=max_speed)
    footprint = Footprint.compact() if body == 'compact' else Footprint.long()
    factory, cls = (RRTConnect, RRTConnectConfig) if algorithm == 'rrt_connect' else (VoronoiPlanner, VoronoiConfig)
    planner_config = cls(time_budget=1., clearance=.12, **({'seed': seed} if algorithm == 'rrt_connect' else {}))
    planner = AdaptiveClearancePlanner(factory, planner_config, footprint)
    lookahead = (1.6 if model_name == 'ackermann' else 1.3)*lookahead_scale
    controller = DWAConfig(prediction_time=2.4 if model_name == 'ackermann' else 1.8, lookahead=lookahead)
    nav = Navigator(model=model, footprint=footprint, global_planner=planner, controller=controller,
                    settings=NavigationConfig(lookahead=lookahead))
    run = run_navigation(case.world, case.start, case.goal, navigator=nav, max_steps=steps)
    # Use actual graphmap poses for heading rather than assuming Euler defaults.
    poses = np.array([planar_state(p) for p in run.poses])
    world = PointObstacles(case.world.point_cloud(case.point_spacing))
    gaps = world.clearances(poses, footprint, case.world.bounds)
    errors = [float(polyline_distances(p[None, :2], o.path.position[:, :2])[0])
              for p, o in zip(poses, run.outputs) if len(o.path.position)]
    turn = run.steering if model_name == 'ackermann' else run.velocities[:, 1]
    signs = np.sign(turn[np.abs(turn) > .03])
    final = poses[-1]
    goal = planar_state(case.goal)
    return {'scenario': case.name, 'algorithm': algorithm, 'model': model_name, 'seed': seed,
            'footprint': asdict(footprint), 'lookahead_scale': lookahead_scale, 'speed_limit_mps': max_speed,
            **run.summary(), 'safe_outputs': all(o.safe for o in run.outputs),
            'terminal_status': str(run.outputs[-1].status),
            'terminal_reason': run.outputs[-1].diagnostics.get('reason', ''),
            'position_error_m': float(np.linalg.norm(final[:2]-goal[:2])),
            'heading_error_rad': float(abs(wrap_angle(final[2]-goal[2]))),
            'maximum_speed_mps': float(np.max(run.velocities[:, 0])),
            'moving_speed_mean_mps': float(np.mean(abs(run.velocities[:, 0][abs(run.velocities[:, 0]) > .03]))) if np.any(abs(run.velocities[:, 0]) > .03) else 0.,
            'path_error_rms_m': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
            'path_error_max_m': max(errors) if errors else None,
            'minimum_margin_clearance_m': float(gaps.min()),
            'turn_reversals': int(np.count_nonzero(np.diff(signs))) if len(signs) else 0,
            'turn_total_variation': float(np.abs(np.diff(turn)).sum()),
            'stationary_steps': int(np.sum(np.max(abs(run.velocities), axis=1) < .01)),
            'path_xy': run.outputs[-1].path.position[:, :2].tolist(),
            'poses': poses.tolist(), 'times': run.timestamps.tolist(), 'velocities': run.velocities.tolist(),
            'steering': run.steering.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', default=[s for s in scenarios() if s != 'blocked'])
    parser.add_argument('--algorithms', nargs='+', default=['rrt_connect', 'voronoi'], choices=['rrt_connect', 'voronoi'])
    parser.add_argument('--models', nargs='+', default=['differential', 'ackermann'], choices=['differential', 'ackermann'])
    parser.add_argument('--body', choices=['compact', 'long'], default='compact')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--steps', type=int, default=900)
    parser.add_argument('--lookahead-scales', nargs='+', type=float, default=[1.])
    parser.add_argument('--max-speed', type=float, default=.9)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases, rows, environment = scenarios(), [], provenance()
    if args.repeats < 1 or args.steps < 1 or any(s not in cases for s in args.scenarios):
        parser.error('positive repeats/steps and known scenarios required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.scenarios:
        for algorithm in args.algorithms:
            for model in args.models:
                for scale in args.lookahead_scales:
                    for seed in range(args.repeats):
                        row = trial(cases[name], algorithm, model, seed=seed, steps=args.steps,
                                    lookahead_scale=scale, body=args.body, max_speed=args.max_speed)
                        rows.append(row)
                        summary = {k: v for k, v in row.items() if k not in ('poses', 'times', 'velocities', 'steering', 'path_xy')}
                        print(json.dumps(summary), flush=True)
                        args.output.write_text(json.dumps({'schema': 1, 'kind': 'synchronous_closed_loop',
                            'environment': environment, 'runs': rows}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
