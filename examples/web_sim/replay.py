"""Replay a saved JSON case through the same backend, without a browser."""
import argparse
import json
from pathlib import Path

import numpy as np

from examples.sim_common.session import Action
from .scene import Scene, SceneSession


def replay(scene, steps=700):
    session = SceneSession(scene)
    if session.nav.goal is None:
        raise ValueError('replay requires map.goal')
    session.apply(Action('mode', 'auto'))
    verified, minimum_gap = True, float('inf')
    for _ in range(steps):
        s = session.tick()
        verified = verified and s.motion.safe
        minimum_gap = min(minimum_gap, s.motion.diagnostics.clearance_m)
        if s.collided or s.fault or s.navigation_status == 'arrived':
            break
    return {'name': scene.name, 'arrived': s.navigation_status == 'arrived', 'collided': s.collided,
            'all_commands_verified': bool(verified), 'steps': s.ticks, 'simulation_seconds': s.state.timestamp,
            'navigation_status': s.navigation_status, 'planner_status': s.planner_status,
            'position_m': s.state.pose.translation[:2].tolist(),
            'minimum_clearance_m': minimum_gap if np.isfinite(minimum_gap) else None, 'fault': s.fault}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path, nargs='+')
    parser.add_argument('--steps', type=int, default=700)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    results = []
    for path in args.scene:
        try:
            results.append(replay(Scene.model_validate_json(path.read_bytes()), args.steps))
        except (ValueError, TypeError, OSError) as error:
            results.append({'file': str(path), 'arrived': False, 'error': str(error)})
    output = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding='utf-8')
    return 0 if all(r['arrived'] and not r['collided'] and r['all_commands_verified'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
