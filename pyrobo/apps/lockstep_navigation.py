"""Complete controlled navigation through one backend-independent public graph."""
import argparse
import asyncio
import importlib
import json
import math
from pathlib import Path
import sys
from time import monotonic, monotonic_ns, process_time

from rsim import Runtime
from rsim.devices import ChassisPortsDevice
from pyrobo import DWAConfig, Footprint, pose2d
from pyrobo.planning import RRTConnectConfig
from pyrobo.rsim import LockstepNavigation, MemoryChassisFactory, StackConfig
from .navigation_provider import NavigationMission, room_map


def _peak_rss_bytes():
    if sys.platform not in ('linux', 'darwin'):
        return None
    try:
        import resource
    except ImportError:
        return None
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == 'darwin' else value*1024


def lockstep_config(*, clock='simulation', control_period=.1):
    """Explicit reproducible demo settings; general StackConfig defaults stay unchanged."""
    return StackConfig(clock=clock, scheduling='lockstep', control_period=control_period,
        execution_period=control_period, adaptive_clearance=False,
        planner=RRTConnectConfig(time_budget=None, max_extensions=1200, shortcut_attempts=30, seed=7, clearance=.08),
        controller=DWAConfig(prediction_time=.6, integration_dt=.1, linear_samples=3, angular_samples=3),
        footprint=Footprint(.6, .6), command_speed_limit=.45, command_yaw_limit=.75)


async def run_lockstep(source, *, physics_timestep, requested_substeps, map_data,
                       goal=(.8, 0., 0.), config=None, max_steps=250, episodes=1, timeout=5.):
    """Own an unstarted compatible provider and run real accepted/advanced steps.

    Provider selection belongs to the caller. This function accesses only public
    chassis ports and the controlled management contract, never a live engine.
    Every trace row represents a successful complete PhysicsAdvanced ACK.
    """
    if type(max_steps) is not int or max_steps < 1 or type(episodes) is not int or episodes < 1:
        raise ValueError('max_steps and episodes must be positive integers')
    device = ChassisPortsDevice.from_component(source)
    config = config or lockstep_config(clock=device.state.clock,
                                       control_period=physics_timestep*requested_substeps)
    mission = NavigationMission(map_data, goal, clock=config.clock)
    runner = LockstepNavigation(device, source, mission.maps, mission.missions, config,
        physics_timestep=physics_timestep, requested_substeps=requested_substeps, timeout=timeout)
    started, cpu_started = monotonic(), process_time()
    reports = []
    async with Runtime(runner):
        ready = monotonic()
        for repetition in range(episodes):
            if repetition:
                await runner.reset()
            record_start, wall_start = len(runner.records), monotonic_ns()
            initial = await device.state.get()
            samples, arrived, collided = [], False, False
            minimum_clearance = float('inf')
            for _ in range(max_steps):
                record = await runner.step()
                state, pose = await device.state.get(), await device.pose.get()
                x, y = map(float, pose.data.position[:2])
                forward = pose.data.rotation.as_matrix()[:, 0]
                yaw = math.atan2(float(forward[1]), float(forward[0]))
                corners = config.footprint.corners(pose2d(x, y, yaw, frame=map_data['frame']))
                xmin, xmax, ymin, ymax = map_data['bounds']
                clearance = min(float((corners[:, 0]-xmin).min()), float((xmax-corners[:, 0]).min()),
                                float((corners[:, 1]-ymin).min()), float((ymax-corners[:, 1]).min()))
                minimum_clearance = min(minimum_clearance, clearance)
                collided |= bool(state.data.get('collided', False)) or clearance <= 0
                speed, turn = float(state.data['linear_x']), float(state.data['angular_z'])
                samples.append({'stamp_ns': state.stamp_ns, 'context': record['advanced']['context'],
                                'pose': [x, y, yaw], 'speed': speed, 'turn': turn,
                                'status': record['status'], 'clearance_m': clearance})
                yaw_error = abs((yaw-goal[2]+math.pi) % (2*math.pi)-math.pi)
                arrived = (record['status'] == 'arrived' and math.hypot(x-goal[0], y-goal[1]) <= config.navigation.goal_tolerance
                           and yaw_error <= config.navigation.yaw_tolerance and abs(speed) < .02 and abs(turn) < .02)
                if arrived or collided:
                    break
            wall_end = monotonic_ns()
            physical_delta = (samples[-1]['stamp_ns']-initial.stamp_ns)/1e9
            wall = (wall_end-wall_start)/1e9
            traces = runner.records[record_start:]
            reports.append({'arrived': arrived, 'collided': collided, 'samples': samples, 'trace': traces,
                'source_initial_stamp_ns': initial.stamp_ns, 'source_delta_s': physical_delta,
                'host_started_ns': wall_start, 'host_finished_ns': wall_end, 'control_window_s': wall,
                'control_window_rtf': physical_delta/wall,
                'completed_control_steps': len(traces), 'completed_control_steps_per_host_s': len(traces)/wall,
                'minimum_boundary_clearance_m': minimum_clearance,
                'final_owner': await source.describe()})
        before_close = monotonic()
    elapsed = monotonic()-started
    source_delta = sum(report['source_delta_s'] for report in reports)
    return {'arrived': all(report['arrived'] for report in reports),
            'collided': any(report['collided'] for report in reports), 'episodes': reports,
            'physics_timestep': physics_timestep, 'requested_substeps': requested_substeps,
            'configuration_fingerprint': config.fingerprint(), 'elapsed_host_s': elapsed,
            'runtime_open_s': ready-started, 'runtime_close_s': elapsed-(before_close-started),
            'physical_delta_s': source_delta, 'end_to_end_rtf': source_delta/elapsed,
            'timing_scope': 'Runtime open, control transactions/resets, and close; excludes factory/build/import',
            'process_cpu_s': process_time()-cpu_started,
            'process_lifetime_peak_rss_bytes': _peak_rss_bytes()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factory', default='pyrobo.rsim:MemoryChassisFactory',
                        help='module:callable producing a provider Factory; no backend branch in the consumer')
    parser.add_argument('--factory-config', type=Path, help='JSON keyword arguments for the selected Factory')
    parser.add_argument('--physics-timestep', type=float, default=.02)
    parser.add_argument('--substeps', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=250)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    module, name = args.factory.split(':', 1)
    factory = getattr(importlib.import_module(module), name)
    options = json.loads(args.factory_config.read_text()) if args.factory_config else {
        'scheduling': 'controlled', 'motion_enabled': True}
    source = factory(**options).build()
    report = asyncio.run(run_lockstep(source, physics_timestep=args.physics_timestep,
        requested_substeps=args.substeps, map_data=room_map(spacing=.3), max_steps=args.max_steps,
        episodes=args.episodes))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'episodes'}, allow_nan=False))
    return 0 if report['arrived'] and not report['collided'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
