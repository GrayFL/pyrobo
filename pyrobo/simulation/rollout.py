"""Reusable headless experiment runner; rendering belongs to the caller."""
from dataclasses import dataclass
from dataclasses import replace
from time import perf_counter
from typing import Callable
import warnings

import numpy as np
from graphmap.pose import Pose, PoseArray

from ..navigation import NavigationInput, NavigationOutput, Navigator
from pyrobo.core.state import ChassisState
from pyrobo.core.commands import ModelCommand
from ..core import RobotObservation
from ..runtime import Supervisor
from pyrobo.core.geometry import pose_array, planar_state, positive
from .world import Simulator2D, World2D


@dataclass(frozen=True)
class SimulationRun:
    poses: PoseArray
    timestamps: np.ndarray
    velocities: np.ndarray
    outputs: tuple[NavigationOutput, ...]
    collided: bool
    commands: tuple[ModelCommand, ...]
    manual: np.ndarray
    steering: np.ndarray
    efforts: np.ndarray

    @property
    def arrived(self):
        return bool(self.outputs and self.outputs[-1].status == "arrived")

    def summary(self):
        latency = np.array([o.diagnostics["elapsed"] for o in self.outputs]) * 1000
        return {"arrived": self.arrived, "collided": self.collided,
                "steps": len(self.outputs), "simulation_seconds": float(self.timestamps[-1] - self.timestamps[0]),
                "distance_m": float(np.linalg.norm(np.diff(self.poses.position[:, :2], axis=0), axis=1).sum()),
                "latency_p50_ms": float(np.median(latency)), "latency_p95_ms": float(np.percentile(latency, 95)),
                "latency_max_ms": float(latency.max()),
                "manual_steps": int(self.manual.sum()),
                "reverse_steps": int(np.sum(self.velocities[:, 0] < -.01)),
                "replans": sum(o.diagnostics["replanned"] for o in self.outputs)}


def run_navigation(world: World2D, start: Pose, goal: Pose, *, navigator: Navigator | None = None,
                   dt: float = 0.1, max_steps: int = 1000, point_spacing: float = 0.04,
                   control_input: Callable[[int, ChassisState, Supervisor], None] | None = None,
                   command_override: Callable[[int, ChassisState, NavigationOutput], ModelCommand | None] | None = None) -> SimulationRun:
    """Run against a known static point map; stops on arrival, emergency or collision.

    Samples/outputs are aligned at the input timestamp, including the final state.
    An existing navigator is reset so experiments are independently replayable.
    `control_input` may select a mode and submit a timed teleop desire each tick.
    `command_override` is a deprecated demonstration hook, still checked by SafetyGuard.
    """
    positive(dt, "dt")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    nav = navigator or Navigator()
    nav.reset()
    if command_override is not None:
        if control_input is not None:
            raise ValueError('choose control_input or legacy command_override')
        warnings.warn('command_override is deprecated; submit desires through control_input and Supervisor',
                      DeprecationWarning, stacklevel=2)
    supervisor = Supervisor(nav)
    supervisor.set_goal(goal, 0.)
    sim = Simulator2D(world, start, footprint=nav.footprint, model=nav.model, dynamics=nav.dynamics.config)
    cloud = world.point_cloud(point_spacing)
    states, times, velocities, outputs = [], [], [], []
    commands, manual, steering, efforts = [], [], [], []
    for k in range(max_steps):
        state = sim.state
        observation = RobotObservation(state, cloud, world.bounds, dt)
        started = perf_counter()
        if command_override is not None:
            output = nav.step(NavigationInput(state.timestamp, state.pose, state.velocity, goal, cloud, world.bounds, dt, state))
        else:
            if control_input is not None:
                control_input(k, state, supervisor)
            motion = supervisor.step(observation)
            output = nav.navigation_output(observation, motion, elapsed=perf_counter()-started)
        override = command_override(k, state, output) if command_override else None
        if override is not None:
            motion = nav.safety.enforce(observation, replace(nav.last_motion, command=override))
            output = replace(output, command=motion.command, safe=motion.safe,
                             status=output.status if motion.safe else 'emergency_stop',
                             diagnostics={**output.diagnostics, 'legacy_override': True,
                                          'override_reason': motion.diagnostics.reason})
        states.append(planar_state(state.pose))
        times.append(state.timestamp)
        velocities.append([state.velocity.linear, state.velocity.angular])
        outputs.append(output)
        command = output.command
        commands.append(command)
        manual.append(override is not None or supervisor.mode == 'manual')
        steering.append(state.steering)
        last = sim.last_dynamics_result
        efforts.append([last.force, last.torque] if last is not None else [0., 0.])
        if (output.status == "arrived" and override is None) or output.status == "emergency_stop" or k == max_steps - 1:
            break
        sim.step(command, dt)
        if sim.collided:
            break
    return SimulationRun(pose_array(np.array(states), world.frame, start.ego_frame), np.array(times),
                         np.array(velocities), tuple(outputs), sim.collided, tuple(commands),
                         np.array(manual), np.array(steering), np.array(efforts))

