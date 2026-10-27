"""Headless assisted driving: python -m examples.assisted_teleop --model ackermann.

No destination is inferred from the joystick. STOP retains its angular intent;
AVOID may turn within the request limits. Both use the same plant and final gate.
"""
import argparse
import json
from dataclasses import replace

import numpy as np

from pyrobo import Ackermann, Bounds2D, DifferentialDrive, Navigator, Twist2D, pose2d
from pyrobo.core import MotionPolicy, RobotObservation
from pyrobo.runtime import ControlMode, Supervisor
from pyrobo.simulation import Box, Simulator2D, World2D


def run_case(name, model):
    if name not in ('free', 'stop', 'avoid', 'no_feasible'):
        raise ValueError('unknown assisted-driving case')
    boxes = [] if name == 'free' else [Box(1.7, 1.9, -.25, .25)]
    if name == 'no_feasible':
        boxes = [Box(.42, .5, -3, 3)]
    world = World2D(Bounds2D(-3, 8, -5, 5), boxes)
    nav = Navigator(model=model)
    runtime = Supervisor(nav, mode=ControlMode.MANUAL)
    sim = Simulator2D(world, pose2d(0, 0), model=model, dynamics=nav.dynamics.config)
    if name == 'no_feasible':
        sim.state = replace(sim.state, velocity=Twist2D(.8, 0))
    cloud = world.point_cloud()
    statuses, positions, results = [], [], []
    policy = MotionPolicy.TELEOP_AVOID if name == 'avoid' else MotionPolicy.TELEOP_STOP
    for _ in range(90 if name != 'no_feasible' else 1):
        state = sim.state
        runtime.submit_teleop(Twist2D(.6, 0), state.timestamp, policy=policy)
        result = runtime.step(RobotObservation(state, cloud, world.bounds))
        statuses.append(result.status.value)
        positions.append(state.pose.translation[:2].copy())
        results.append(result)
        if not result.safe:
            break  # report unavoidable/unverified braking; do not claim safe execution
        sim.step(result.command)
        if sim.collided:
            raise AssertionError('solid collision in assisted driving')
    positions = np.array(positions)
    summary = {'case': name, 'model': type(model).__name__, 'steps': len(results),
               'position_m': positions[-1].round(4).tolist(), 'collided': sim.collided,
               'statuses': sorted(set(statuses)), 'last_reason': results[-1].diagnostics.reason,
               'safety': results[-1].safety_assessment.value,
               'safety_action': results[-1].safety_action.value}
    if name == 'free':
        assert positions[-1, 0] > 4 and all(s == 'tracking' for s in statuses)
    elif name == 'stop':
        assert positions[-1, 0] < 1.4 and max(abs(positions[:, 1])) < 1e-9
        assert 'constrained' in statuses
    elif name == 'avoid':
        assert 'obstacle_avoiding' in statuses and positions[-1, 0] > 2.3
    else:
        assert results[-1].command.mode == 'estop' and not results[-1].safe
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('differential', 'ackermann', 'both'), default='both')
    parser.add_argument('--case', choices=('free', 'stop', 'avoid', 'no_feasible', 'all'), default='all')
    args = parser.parse_args()
    models = [DifferentialDrive(), Ackermann()] if args.model == 'both' else [
        Ackermann() if args.model == 'ackermann' else DifferentialDrive()]
    cases = ('free', 'stop', 'avoid', 'no_feasible') if args.case == 'all' else (args.case,)
    for model in models:
        for case in cases:
            print(json.dumps(run_case(case, model), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
