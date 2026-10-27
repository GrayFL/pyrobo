"""Capture pre-refactor dynamics; deliberately not run by tests or after refactoring."""
import numpy as np
from pathlib import Path
from pyrobo import Ackermann, ChassisDynamics, ChassisState, DifferentialDrive, Twist2D, pose2d


def main():
    destination = Path('tests/data/dynamics_baseline.npz')
    if destination.exists():
        raise FileExistsError('the pre-refactor reference is immutable; do not overwrite it with the new implementation')
    arrays = {}
    for name, model in [('differential', DifferentialDrive()), ('ackermann', Ackermann())]:
        plant = ChassisDynamics(model)
        for mode in ('track', 'coast', 'soft_stop', 'estop'):
            delta = .2 if name == 'ackermann' else 0.
            yaw = model.yaw_rate(.4, delta) if name == 'ackermann' else .2
            state = ChassisState(pose2d(1, -2, .3), Twist2D(.4, yaw), steering=delta,
                                 steering_rate=.1 if delta else 0., acceleration=.1,
                                 yaw_acceleration=.2, steering_acceleration=.15 if delta else 0.)
            values = [plant.pack(state)]
            efforts = []
            for dt in (.1, .07, .2, .13, .08):
                result = plant.step(state, model.command(-.25, -.3, mode=mode), dt)
                state = result.state
                values.append(plant.pack(state))
                efforts.append([result.force, result.torque])
            arrays[f'{name}_{mode}_states'] = np.array(values)
            arrays[f'{name}_{mode}_efforts'] = np.array(efforts)
    np.savez(destination, **arrays)


if __name__ == '__main__':
    main()
