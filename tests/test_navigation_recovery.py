"""Bounded recovery after a verified stop leaves no useful forward arc."""
from dataclasses import replace

import numpy as np
import pytest

from pyrobo import (Ackermann, Bounds2D, ChassisState, DWAConfig, DWAEngine, PointCloud,
                    RobotObservation, pose2d)
from pyrobo.core import MotionPolicy, MotionRequest, Path
from pyrobo.core.geometry import path_poses


@pytest.mark.parametrize('allow_reverse', [False, True])
def test_stagnation_recovery_is_bounded_and_respects_reverse_permission(allow_reverse):
    engine = DWAEngine(DWAConfig(allow_reverse=allow_reverse), model=Ackermann())
    request = MotionRequest(MotionPolicy.NAVIGATION, 0., goal=pose2d(4, 0),
                            path=Path(path_poses([[0, 0], [4, 0]])))
    modes = []
    for i in range(300):
        stamp = i*.1
        obs = RobotObservation(ChassisState(pose2d(0, 0), timestamp=stamp),
                               PointCloud(np.empty((0, 2))), Bounds2D(-5, 5, -5, 5))
        context = engine._prepare(obs, replace(request, timestamp=stamp))
        modes.append(context.mode)
        if context.mode == 'recovering':
            assert context.lower[0] >= -engine.footprint.length/(2*engine.config.prediction_time)
            assert context.recovery_distance == engine.footprint.length/2
    assert ('recovering' in modes) is allow_reverse
    assert engine.navigation_policy.recovery_attempts == (3 if allow_reverse else 0)
    assert modes[-1] == 'following'


def test_ackermann_recovers_from_measured_narrow_corridor_stopping_pose():
    from benchmarks.navigation.control import trial
    from benchmarks.navigation.scenarios import scenarios
    case = replace(scenarios()['narrow_ackermann'],
                   start=pose2d(5.9808207646, 4.427059758, -.352111393))
    result = trial(case, 'rrt_connect', 'ackermann', steps=350)
    assert result['arrived'] and not result['collided'] and result['safe_outputs'], result
    assert result['reverse_steps'] > 0 and result['minimum_margin_clearance_m'] > 0
