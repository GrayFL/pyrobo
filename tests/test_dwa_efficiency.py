import numpy as np
import pytest

from pyrobo import (Ackermann, Bounds2D, ChassisState, DifferentialDrive, DWAConfig,
                    DWAEngine, PointCloud, RobotObservation, Twist2D, pose2d)
from pyrobo.core import MotionPolicy, MotionRequest, Path
from pyrobo.core.geometry import path_poses
from pyrobo.control.dwa.sampling import sample_window


@pytest.mark.parametrize('model', [Ackermann(), DifferentialDrive()])
@pytest.mark.parametrize('chunk', [0, 13])
@pytest.mark.parametrize('speed, wall', [(.4, .55), (.8, .53)])
def test_immediate_rejection_preserves_full_batch_selection_and_candidate_ids(model, chunk, speed, wall):
    engine = DWAEngine(DWAConfig(batch_size=chunk), model=model)
    cloud = PointCloud(np.column_stack((np.full(501, wall), np.linspace(-2, 2, 501))))
    obs = RobotObservation(ChassisState(pose2d(0, 0), Twist2D(speed, 0.)), cloud, Bounds2D(-4, 4, -4, 4))
    req = MotionRequest(MotionPolicy.NAVIGATION, 0., goal=pose2d(3, 0), path=Path(path_poses([[0, 0], [3, 0]])))
    ctx = engine._prepare(obs, req)
    window = engine.reachability.command_window(engine.plant.pack(obs.chassis), req, obs.dt,
                                                lower=ctx.lower, upper=ctx.upper, anchors=ctx.anchors)
    search = sample_window(window, (engine.config.linear_samples, engine.config.angular_samples))
    full = engine.evaluate_candidates(obs, req, search, context=ctx)
    assert not full.feasible.all()
    evaluated = []
    evaluate = engine.evaluate_candidates
    def record(observation, request, candidates, **kwargs):
        evaluated.append(len(candidates))
        return evaluate(observation, request, candidates, **kwargs)
    engine.evaluate_candidates = record
    engine.reset()
    actual = engine.optimize(obs, req)
    assert actual.safe
    assert actual.diagnostics.feasible_count == full.feasible.sum()
    if speed == .8:
        assert sum(evaluated) < len(search)
    if full.feasible.any():
        best = np.flatnonzero(full.costs == full.costs.min())
        lo, hi = engine.command_space.bounds()
        expected = best[np.argmin(np.sum(((search[best]-ctx.desired)/(hi-lo))**2, axis=1))]
        assert actual.diagnostics.extra['candidate_id'] == expected
        np.testing.assert_allclose(model.encode_command(actual.command), full.native_commands[expected])
    else:
        assert actual.command.mode == 'estop' and actual.diagnostics.reason == 'no_feasible_candidate'
