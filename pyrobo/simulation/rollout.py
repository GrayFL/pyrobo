"""Reusable headless experiment runner; rendering belongs to the caller."""
from dataclasses import dataclass

import numpy as np
from graphmap.pose import Pose, PoseArray

from ..navigation import NavigationInput, NavigationOutput, Navigator
from ..types import pose_array, planar_state, positive
from .world import Simulator2D, World2D


@dataclass(frozen=True)
class SimulationRun:
    poses: PoseArray
    timestamps: np.ndarray
    velocities: np.ndarray
    outputs: tuple[NavigationOutput, ...]
    collided: bool

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
                "replans": sum(o.diagnostics["replanned"] for o in self.outputs)}


def run_navigation(world: World2D, start: Pose, goal: Pose, *, navigator: Navigator | None = None,
                   dt: float = 0.1, max_steps: int = 1000, point_spacing: float = 0.04) -> SimulationRun:
    """Run against a known static point map; stops on arrival, emergency or collision.

    Samples/outputs are aligned at the input timestamp, including the final state.
    An existing navigator is reset so experiments are independently replayable.
    """
    positive(dt, "dt")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    nav = navigator or Navigator()
    nav.reset()
    sim = Simulator2D(world, start, footprint=nav.footprint, dynamics=nav.controller.config)
    cloud = world.point_cloud(point_spacing)
    states, times, velocities, outputs = [], [], [], []
    for k in range(max_steps):
        state = sim.state
        output = nav.step(NavigationInput(state.timestamp, state.pose, state.velocity, goal, cloud, world.bounds, dt))
        states.append(planar_state(state.pose))
        times.append(state.timestamp)
        velocities.append([state.velocity.linear, state.velocity.angular])
        outputs.append(output)
        if output.status in ("arrived", "emergency_stop") or k == max_steps - 1:
            break
        sim.step(output.command, dt)
        if sim.collided:
            break
    return SimulationRun(pose_array(np.array(states), world.frame, start.ego_frame), np.array(times),
                         np.array(velocities), tuple(outputs), sim.collided)

