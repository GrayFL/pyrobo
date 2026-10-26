"""Native manual commands, Twist adaptation, coasting and autonomous handoff state.

Run from the repository: python -m examples.manual_control
"""
from pyrobo import (Ackermann, ChassisDynamics, ChassisState, DifferentialDrive,
                    Twist2D, pose2d)


def main():
    for model in (DifferentialDrive(), Ackermann()):
        plant = ChassisDynamics(model)
        state = ChassisState(pose2d(0, 0, frame="map", body="axle"))
        phases = [(model.from_twist(Twist2D(.6, .3)), 40),
                  (model.command(mode="coast"), 20),
                  (model.command(-.3, .2), 30),
                  (model.command(mode="estop"), 15)]
        for command, steps in phases:
            for _ in range(steps):
                result = plant.step(state, command, .05)
                state = result.state
            print(type(model).__name__, command.mode, state.velocity,
                  "steering=", round(state.steering, 3), "force=", round(result.force, 3))
        # Feed this exact state to NavigationInput(chassis=state), or start
        # manual control from Simulator2D.state. Keep steering and acceleration.
        assert abs(state.velocity.linear) < 1e-3
        assert abs(state.velocity.angular) < 1e-3


if __name__ == "__main__":
    main()
