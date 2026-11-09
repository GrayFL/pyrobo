"""Measure DWA/TEB on identical measured states and unmodified point clouds.

Latency includes local optimization, rollout and brake checks; global planning
and the independent final execution gate are excluded. Python allocation peak
is measured separately so tracemalloc does not distort latency percentiles.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
import tracemalloc

import numpy as np

from pyrobo import (Ackermann, Bounds2D, ChassisDynamics, ChassisState, DifferentialDrive,
    DWAConfig, DWAEngine, Footprint, MotionPolicy, MotionRequest, Path as Route, PointCloud,
    RobotObservation, TEBConfig, TimedElasticBand, pose2d)
from pyrobo.core.geometry import path_poses
from .planning import provenance


def benchmark(repeats=30):
    report = {'schema': 1, 'kind': 'local_optimizer_latency', 'environment': provenance(), 'runs': []}
    for model in (DifferentialDrive(), Ackermann()):
        for points in (1000, 10000):
            x = np.linspace(-4., 8., points//2)
            cloud = PointCloud(np.vstack((np.column_stack((x, np.full_like(x, -1.2))),
                                          np.column_stack((x, np.full_like(x, 1.2))))))
            plant, state, observations = ChassisDynamics(model), ChassisState(pose2d(0, 0)), []
            for _ in range(repeats+1):
                observations.append(RobotObservation(state, cloud, Bounds2D(-4., 8., -2., 2.), .1))
                state = plant.step(state, model.command(.35, 0.), .1).state
            route = Route(path_poses([[0, 0], [6, 0]], 'world'))
            for name, cls, config in [('dwa', DWAEngine, DWAConfig(prediction_time=2.4)),
                                      ('teb', TimedElasticBand, TEBConfig())]:
                engine = cls(config, Footprint.long(offset_x=.25), model=model)
                times, candidates, evaluations, statuses, gaps, stop_distances = [], [], [], [], [], []
                for i, observation in enumerate(observations):
                    request = MotionRequest(MotionPolicy.NAVIGATION, observation.timestamp, path=route, goal=pose2d(6, 0))
                    started = perf_counter()
                    result = engine.optimize(observation, request)
                    elapsed = perf_counter()-started
                    assert result.safe
                    assert engine.plant.codec.stopped(result.trajectory.states[-1:])[0]
                    if i:  # import/index/first sparse-structure warmup excluded, identical for both
                        times.append(elapsed*1000)
                        candidates.append(result.diagnostics.candidate_count)
                        evaluations.append(result.diagnostics.extra.get('teb_evaluations', 0))
                        statuses.append(result.diagnostics.extra.get('teb_status', 'sampled'))
                        gaps.append(result.diagnostics.clearance_m)
                        tail = result.trajectory.poses.position[result.trajectory.prediction_end:, :2]
                        stop_distances.append(float(np.linalg.norm(np.diff(tail, axis=0), axis=1).sum()))
                # Allocation tracing is much slower: a wall budget would otherwise
                # measure an aborted solve and a different, often smaller command set.
                if name == 'teb':
                    engine.config = replace(config, time_budget=None)
                tracemalloc.start()
                memory_result = engine.optimize(observations[-1], request)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                report['runs'].append({'model': type(model).__name__, 'optimizer': name, 'points': len(cloud.points),
                    'config': asdict(config), 'repeats': repeats, 'latency_ms': times,
                    **dict(zip(('p50_ms', 'p95_ms', 'p99_ms'), map(float, np.percentile(times, [50, 95, 99])))),
                    'python_peak_bytes': peak, 'candidates': candidates, 'residual_evaluations': evaluations,
                    'memory_scope': 'one solve and verification; time budget disabled during tracemalloc',
                    'memory_candidates': memory_result.diagnostics.candidate_count,
                    'braking_tail_distance_m': stop_distances,
                    'solver_status': statuses, 'minimum_clearance_m': min(gaps), 'all_commands_verified': True})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    report = benchmark(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    for row in report['runs']:
        print(json.dumps({k: row[k] for k in ('model', 'optimizer', 'points', 'p50_ms', 'p95_ms', 'p99_ms', 'python_peak_bytes')}))


if __name__ == '__main__':
    main()
