"""Numerical reference captured from commit 9aaab71, before the architecture change."""
from pathlib import Path

import numpy as np
import pytest

from pyrobo import Ackermann, ChassisDynamics, ChassisState, DifferentialDrive, Twist2D, pose2d


@pytest.mark.parametrize('model_name', ['differential', 'ackermann'])
@pytest.mark.parametrize('mode', ['track', 'coast', 'soft_stop', 'estop'])
def test_preserved_physics_against_pre_refactor_capture(model_name, mode):
    model = DifferentialDrive() if model_name == 'differential' else Ackermann()
    plant = ChassisDynamics(model)
    fixture = np.load(Path(__file__).parent / 'data/dynamics_baseline.npz')
    baseline = fixture[f'{model_name}_{mode}_states']
    initial = baseline[0]
    state = ChassisState(pose2d(*initial[:3]), Twist2D(*initial[3:5]), *initial[5:])
    for i, dt in enumerate((.1, .07, .2, .13, .08)):
        result = plant.step(state, model.command(-.25, -.3, mode=mode), dt)
        state = result.state
        np.testing.assert_allclose(plant.pack(state), baseline[i + 1], atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose([result.force, result.torque],
                                   fixture[f'{model_name}_{mode}_efforts'][i], atol=1e-11, rtol=1e-11)
