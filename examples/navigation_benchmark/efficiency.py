"""Compare full candidate evaluation with early rejection near a stopping limit.

The full-batch timing excludes sampling, selection and fallback, so it is a
lower bound on the unpruned whole-controller cost, not an old-version replay.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pyrobo import (Ackermann, Bounds2D, ChassisState, DifferentialDrive, DWAEngine,
                    PointCloud, RobotObservation, Twist2D, pose2d)
from pyrobo.core import MotionPolicy, MotionRequest, Path as Route
from pyrobo.core.geometry import path_poses
from pyrobo.control.dwa.sampling import sample_window
from .planning import provenance


def benchmark(repeats=20):
    report = {'schema': 1, 'kind': 'early_rejection_cost', 'environment': provenance(), 'runs': []}
    for model in (DifferentialDrive(), Ackermann()):
        engine = DWAEngine(model=model)
        observation = RobotObservation(ChassisState(pose2d(0, 0), Twist2D(.8)),
            PointCloud(np.column_stack((np.full(501, .5), np.linspace(-2, 2, 501)))), Bounds2D(-4, 4, -4, 4))
        request = MotionRequest(MotionPolicy.NAVIGATION, 0., goal=pose2d(3, 0),
                                path=Route(path_poses([[0, 0], [3, 0]])))
        context = engine._prepare(observation, request)
        window = engine.reachability.command_window(engine.plant.pack(observation.chassis), request,
            observation.dt, lower=context.lower, upper=context.upper, anchors=context.anchors)
        search = sample_window(window, (engine.config.linear_samples, engine.config.angular_samples))
        engine.evaluate_candidates(observation, request, search, context=context)
        engine.optimize(observation, request)
        full, optimized = [], []
        for _ in range(repeats):
            start = perf_counter()
            batch = engine.evaluate_candidates(observation, request, search, context=context)
            full.append((perf_counter()-start)*1000)
            engine.reset()
            start = perf_counter()
            result = engine.optimize(observation, request)
            optimized.append((perf_counter()-start)*1000)
            assert not batch.feasible.any() and result.safe and result.command.mode == 'estop'
        report['runs'].append({'model': type(model).__name__, 'candidates': len(search), 'repeats': repeats,
            'speed_mps': .8, 'wall_x_m': .5, 'full_evaluation_ms': full, 'optimized_ms': optimized,
            'full_p95_ms': float(np.percentile(full, 95)), 'optimized_p95_ms': float(np.percentile(optimized, 95)),
            'verified_stop': True})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('positive repeats required')
    result = benchmark(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
