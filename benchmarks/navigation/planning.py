"""Repeated global solves with common budgets and independent solid checks."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from pyrobo import (Ackermann, ChassisState, DifferentialDrive, Footprint, NavigationConfig,
                    Navigator, RobotObservation, pose2d)
from pyrobo.core.geometry import planar_state, wrap_angle
from pyrobo.geometry.collision import PointObstacles
from pyrobo.planning import AdaptiveClearancePlanner, RRTConnect, RRTConnectConfig, VoronoiConfig, VoronoiPlanner
from .scenarios import scenarios


def provenance():
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root/'pyrobo').rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    benchmark = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob('*.py')):
        benchmark.update(str(path.relative_to(root)).encode())
        benchmark.update(path.read_bytes())
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'versions': {k: importlib.metadata.version(k) for k in ('numpy', 'scipy')},
            'source_sha256': digest.hexdigest(), 'benchmark_sha256': benchmark.hexdigest(),
            'started_utc': datetime.now(timezone.utc).isoformat(), 'cpu_count': os.cpu_count(),
            'load_average': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None}


def solid_path_valid(case, xy, footprint):
    """Independent simulation geometry, sampled at <= 4 cm and <= 2 degrees.

    Includes rotations at XY corners. This proves only the sampled geometric
    route, not car reachability; closed-loop experiments test actual driving.
    """
    yaw = float(planar_state(case.start)[2])
    for i, (a, b) in enumerate(zip(xy[:-1], xy[1:])):
        direction = float(np.arctan2(*(b-a)[::-1]))
        turn = float(wrap_angle(direction-yaw))
        for heading in yaw+np.linspace(0, turn, max(2, int(abs(turn)/np.deg2rad(2))+2)):
            if case.world.collides(pose2d(*a, heading), footprint, padding=footprint.margin):
                return False
        for p in np.linspace(a, b, max(2, int(np.linalg.norm(b-a)/.04)+2)):
            if case.world.collides(pose2d(*p, direction), footprint, padding=footprint.margin):
                return False
        yaw = direction
    turn = float(wrap_angle(planar_state(case.goal)[2]-yaw))
    return all(not case.world.collides(pose2d(*xy[-1], heading), footprint, padding=footprint.margin)
               for heading in yaw+np.linspace(0, turn, max(2, int(abs(turn)/np.deg2rad(2))+2)))


def solve(case, algorithm, *, model='ackermann', seed=0, budget=1., clearance=.08,
          approach=1., footprint=None, bridge_sampling=True, adaptive=False):
    body = footprint or Footprint.compact()
    model = Ackermann() if model == 'ackermann' else DifferentialDrive()
    planner = (RRTConnectConfig(seed=seed, time_budget=budget, clearance=clearance, bridge_sampling=bridge_sampling)
               if algorithm == 'rrt_connect' else VoronoiConfig(time_budget=budget, clearance=clearance))
    composed = (AdaptiveClearancePlanner(RRTConnect if algorithm == 'rrt_connect' else VoronoiPlanner,
                                        planner, body) if adaptive else None)
    nav = Navigator(model=model, footprint=body, planner=planner, global_planner=composed,
                    settings=NavigationConfig(goal_approach_distance=approach))
    cloud = case.world.point_cloud(case.point_spacing)
    observation = RobotObservation(ChassisState(case.start), cloud, case.world.bounds)
    started = perf_counter()
    nav.make_motion_request(observation, goal=case.goal)
    elapsed = perf_counter()-started
    xy = nav.path.poses.position[:, :2]
    found = len(xy) > 0
    valid = solid_path_valid(case, xy, body) if found else None
    gap = None
    if found:
        directions = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
        poses = np.column_stack((xy, np.r_[directions, planar_state(case.goal)[2]]))
        gap = float(PointObstacles(cloud).clearances(poses, body, case.world.bounds).min())
    return {'scenario': case.name, 'algorithm': algorithm, 'model': type(model).__name__,
            'seed': seed, 'budget_s': budget, 'clearance_m': clearance, 'approach_m': approach,
            'bridge_sampling': bridge_sampling, 'adaptive_clearance': adaptive,
            'footprint': asdict(body), 'expected_path': case.expected_path,
            'status': str(nav._planner_status), 'found': found, 'solid_path_valid': valid,
            'elapsed_ms': elapsed*1000, 'budget_overrun_ms': max(0., elapsed-budget)*1000,
            'diagnostics': nav._planning,
            'length_m': float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if found else None,
            'minimum_margin_clearance_m': gap, 'path_xy': xy.tolist()}


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['scenario'], row['algorithm'], row['model']), []).append(row)
    output = []
    for (scenario, algorithm, model), values in groups.items():
        times = [v['elapsed_ms'] for v in values]
        good = [v for v in values if v['found'] and v['solid_path_valid']]
        output.append({'scenario': scenario, 'algorithm': algorithm, 'model': model,
                       'runs': len(values), 'expected_path': values[0]['expected_path'],
                       'successes': len(good), 'success_rate': len(good)/len(values),
                       **{f'p{q}_ms': float(np.percentile(times, q)) for q in (50, 95, 99)},
                       'max_ms': max(times), 'invalid_paths': sum(v['solid_path_valid'] is False for v in values),
                       'statuses': {s: sum(v['status'] == s for v in values) for s in sorted({v['status'] for v in values})}})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', default=list(scenarios()))
    parser.add_argument('--algorithms', nargs='+', choices=['rrt_connect', 'voronoi'], default=['rrt_connect', 'voronoi'])
    parser.add_argument('--model', choices=['differential', 'ackermann'], default='ackermann')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--budget', type=float, default=1.)
    parser.add_argument('--clearance', type=float, default=.08)
    parser.add_argument('--bridge-sampling', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--approach', type=float, default=None, help='automatic by default; explicit entry length in metres')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.budget <= 0:
        parser.error('positive repeats and budget required')
    cases, rows, environment = scenarios(), [], provenance()
    if any(name not in cases for name in args.scenarios):
        parser.error('unknown scenario')
    for name in args.scenarios:
        for algorithm in args.algorithms:
            for seed in range(args.repeats):
                rows.append(solve(cases[name], algorithm, model=args.model, seed=seed,
                                  budget=args.budget, clearance=args.clearance, approach=args.approach,
                                  bridge_sampling=args.bridge_sampling, adaptive=args.adaptive))
            print(json.dumps(summarize(rows[-args.repeats:])[0]), flush=True)
    report = {'schema': 1, 'kind': 'global_planning', 'environment': environment,
              'parameters': {**vars(args), 'output': str(args.output)}, 'summary': summarize(rows), 'runs': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
