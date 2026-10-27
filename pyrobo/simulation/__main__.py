"""Headless examples: global planner + model-space DWA + shared force plant."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import (Ackermann, Bounds2D, DifferentialDrive, Footprint, Navigator,
                RRTConnectConfig, Twist2D, VoronoiConfig, pose2d)
from ..runtime import ControlMode
from . import World2D, demo_world, run_navigation


def plot_run(run, world, footprint, destination):
    """Optional rendering; matplotlib is never imported by the core package."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    for box in world.boxes:
        ax.add_patch(Rectangle((box.xmin, box.ymin), box.xmax-box.xmin, box.ymax-box.ymin, color='.25'))
    path = next((o.path.position for o in run.outputs if len(o.path.position)), np.empty((0, 3)))
    ax.plot(path[:, 0], path[:, 1], '--', color='.6', label='Global path')
    xy = run.poses.position[:, :2]
    ax.plot(xy[:, 0], xy[:, 1], color='tab:blue', label='Executed trajectory')
    for i in np.linspace(0, len(xy)-1, min(18, len(xy)), dtype=int):
        ax.add_patch(Polygon(footprint.corners(run.poses[i]), fill=False, alpha=.5, color='tab:blue'))
    ax.set(xlim=(world.bounds.xmin, world.bounds.xmax), ylim=(world.bounds.ymin, world.bounds.ymax),
           xlabel='x (m)', ylabel='y (m)', title='Rectangular body and point-map navigation')
    ax.set_aspect('equal')
    ax.legend()
    axes[0, 1].plot(run.timestamps, run.velocities[:, 0], label='Speed (m/s)')
    axes[0, 1].plot(run.timestamps, run.velocities[:, 1], label='Yaw rate (rad/s)')
    axes[0, 1].plot(run.timestamps, run.steering, label='Steering (rad)')
    progress = [o.diagnostics['progress_m'] for o in run.outputs]
    axes[1, 0].plot(run.timestamps, progress, label='Path progress (m)')
    axes[1, 0].fill_between(run.timestamps, 0, 1, where=run.manual,
                          transform=axes[1, 0].get_xaxis_transform(), alpha=.15, label='Manual reverse')
    axes[1, 1].plot(run.timestamps, run.efforts[:, 0], label='Drive / brake force (N)')
    axes[1, 1].plot(run.timestamps, run.efforts[:, 1], label='Yaw / steering torque (Nm)')
    for ax in (axes[0, 1], *axes[1]):
        ax.set_xlabel('Time (s)')
        ax.legend()
        ax.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner', choices=('rrt_connect', 'voronoi'), default='rrt_connect')
    parser.add_argument('--model', choices=('differential', 'ackermann'), default='differential')
    parser.add_argument('--footprint', choices=('compact', 'long'), default='compact')
    parser.add_argument('--scenario', choices=('detour', 'handoff'), default='detour')
    parser.add_argument('--plot', type=Path, help='Save an optional matplotlib plot')
    args = parser.parse_args()
    model = Ackermann() if args.model == 'ackermann' else DifferentialDrive()
    footprint = Footprint.long() if args.footprint == 'long' else Footprint.compact()
    clearance = .12 + .4 * model.minimum_turn_radius if isinstance(model, Ackermann) else .12
    config = (VoronoiConfig(clearance=clearance, time_budget=2.) if args.planner == 'voronoi'
              else RRTConnectConfig(clearance=clearance, time_budget=.5))
    world, start, goal = demo_world()
    control_input = None
    if args.scenario == 'handoff':
        world = World2D(Bounds2D(-2, 9, -3, 3))
        start, goal = pose2d(0, 0), pose2d(7, 0)
        def control_input(step, state, supervisor):
            supervisor.set_mode(ControlMode.MANUAL if 35 <= step < 65 else ControlMode.AUTO)
            if 35 <= step < 65:
                supervisor.submit_teleop(Twist2D(-.35, 0.), state.timestamp)
    run = run_navigation(world, start, goal, navigator=Navigator(model=model, footprint=footprint, planner=config),
                         control_input=control_input)
    print(json.dumps({'planner': args.planner, 'model': args.model, 'footprint': args.footprint,
                      'scenario': args.scenario, **run.summary()}, indent=2))
    if args.plot:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_run(run, world, footprint, args.plot)
    if not run.arrived or run.collided:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
