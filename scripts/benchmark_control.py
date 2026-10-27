"""Repeatable control/rollout baseline; run at repo root in the same idle environment.

python -m scripts.benchmark_control --output benchmarks/after.json
Latency measurements and tracemalloc measurements run separately.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np

from pyrobo import (Ackermann, Bounds2D, ChassisState, DifferentialDrive, DynamicWindow,
                    Footprint, PointCloud, PointObstacles, Twist2D, pose2d)
from pyrobo.types import path_poses


def benchmark(repeats=24):
    source = Path(__file__).resolve().parents[1] / 'pyrobo'
    digest = hashlib.sha256()
    for path in sorted(source.rglob('*.py')):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    report = {'schema': 1, 'seed': 17, 'repeats': repeats, 'python': platform.python_version(),
              'versions': {n: importlib.metadata.version(n) for n in ('numpy', 'scipy')},
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': digest.hexdigest(), 'cases': {}, 'rollout': {}}
    bounds = Bounds2D(-5, 5, -5, 5)
    points = np.column_stack((np.full(101, 2.), np.linspace(-1, 1, 101)))
    for name, model in [('differential', DifferentialDrive()), ('ackermann', Ackermann())]:
        for scene in ('empty', 'obstacle'):
            controller = DynamicWindow(model=model, footprint=Footprint.long())
            cloud = PointCloud(np.empty((0, 2)) if scene == 'empty' else points)
            obstacles = PointObstacles(cloud)
            state = ChassisState(pose2d(0, 0), Twist2D(.35))
            goal, path = pose2d(3, 1), path_poses([[0, 0], [1, 1.8], [3, 1.8], [3, 1]])
            def decide():
                controller.reset()
                return controller.step(state.pose, state.velocity, goal, path, obstacles, bounds, .1, chassis=state)
            decide()
            latency, counts, feasible, clearances = [], [], [], []
            for _ in range(repeats):
                started = time.perf_counter()
                output = decide()
                latency.append((time.perf_counter() - started) * 1000)
                counts.append(output.candidates)
                feasible.append(output.feasible_candidates)
                clearances.append(output.clearance)
            tracemalloc.start()
            decide()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            stop = controller.stop(state.pose, state.velocity, obstacles, bounds, .1, chassis=state)
            report['cases'][f'{name}/{scene}'] = {
                'p50_ms': float(np.percentile(latency, 50)), 'p95_ms': float(np.percentile(latency, 95)),
                'p99_ms': float(np.percentile(latency, 99)), 'peak_bytes': peak,
                'candidate_count': int(np.median(counts)), 'feasible_count': int(np.median(feasible)),
                'feasible_ratio': float(np.mean(np.array(feasible) / np.maximum(counts, 1))),
                'minimum_clearance_m': float(min(clearances)), 'safe': bool(output.safe),
                'selected_native': model.encode_command(output.command).tolist(),
                'selected_trajectory_samples': len(output.trajectory.position),
                'prediction_time_s': controller.config.prediction_time,
                'recording_dt_s': controller.config.integration_dt,
                'stopping_distance_m': float(np.linalg.norm(np.diff(stop.trajectory.position[:, :2], axis=0), axis=1).sum()),
                'stop_safe': bool(stop.safe)}
        for count in (1, 32, 128):
            rng = np.random.default_rng(17)
            commands = np.stack([model.encode_command(model.command(v, w))
                                 for v, w in rng.uniform([-.4, -.5], [.8, .5], (count, 2))])
            states = np.tile(controller.plant.pack(state), (count, 1))
            elapsed = []
            for _ in range(5):
                start = time.perf_counter()
                predicted = controller.plant.integrate(states, commands, .4)
                elapsed.append((time.perf_counter() - start) * 1000)
            report['rollout'][f'{name}/{count}'] = {'p50_ms': float(np.median(elapsed)),
                                                     'state_shape': list(predicted.shape)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=24)
    args = parser.parse_args()
    result = benchmark(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
