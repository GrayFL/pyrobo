"""Navigate using an already running common chassis provider.

The consumer imports neither MujocoSim nor MuJoCo. The default example uses a
prebuilt empty 6 m room and explicit planar projection of localization; it is
not an online mapping or SLAM demonstration.
"""
import argparse
import asyncio
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from time import monotonic, monotonic_ns

import numpy as np
from rsim import Component, Runtime
from rsim.devices import ChassisPortsDevice
from rsim.transport.descriptor import TransportConfig

from pyrobo import Footprint, pose2d
from pyrobo.rsim import NavigationInputs, NavigationCommandOutput, NavigationStack, StackConfig


def room_map(bounds=(-3., 3., -3., 3.), *, frame='world', spacing=.05):
    xmin, xmax, ymin, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError('map bounds must be xmin/xmax/ymin/ymax')
    x = np.linspace(xmin, xmax, math.ceil((xmax-xmin)/spacing)+1)
    y = np.linspace(ymin, ymax, math.ceil((ymax-ymin)/spacing)+1)
    points = np.concatenate((np.column_stack((x, np.full_like(x, ymin))),
                             np.column_stack((x, np.full_like(x, ymax))),
                             np.column_stack((np.full_like(y, xmin), y)),
                             np.column_stack((np.full_like(y, xmax), y))))
    return {'points': points, 'frame': frame, 'revision': 1, 'static': True, 'bounds': list(bounds)}


class NavigationMission(Component):
    def __init__(self, map_data, goal, *, clock):
        super().__init__()
        self.map_data, self.goal, self.clock = map_data, tuple(goal), clock
        self.maps = self.signal('maps', clock=clock)
        self.missions = self.signal('missions', clock=clock)

    async def open(self):
        await self.maps.publish(self.map_data, stamp_ns=0, clock=self.clock)
        await self.missions.publish({'epoch': 'navigation-example', 'goal': self.goal, 'mode': 'auto'},
                                    stamp_ns=0, clock=self.clock)


async def navigate(device, *, map_data, goal=(.8, 0., 0.), config=None,
                   execution='thread', duration=15., planar_projection=True, scheduling=None, tick_source=None):
    """The same consumer graph for every compatible provider; no backend branch."""
    if execution not in ('local', 'thread', 'process'):
        raise ValueError('execution must be local, thread or process')
    config = config or StackConfig(clock=device.pose.clock, footprint=Footprint(.6, .6),
                                  command_speed_limit=.45, command_yaw_limit=.75,
                                  scheduling=scheduling or 'realtime')
    if scheduling is not None and config.scheduling != scheduling:
        raise ValueError('explicit scheduling must match the supplied StackConfig')
    if config.scheduling == 'lockstep':
        raise ValueError('controlled navigation uses NavigationStack.step_once and explicit physics advancement')
    mission = NavigationMission(map_data, goal, clock=config.clock)
    inputs = NavigationInputs(device, mission.maps, mission.missions, config,
                              planar_projection=planar_projection, tick_source=tick_source)
    navigation = NavigationStack(inputs.states, inputs.maps, inputs.missions, config,
        local_options={'offload': execution == 'thread'}, global_options={'offload': execution == 'thread'},
        tick_source=tick_source)
    output = NavigationCommandOutput(navigation.commands, device, config)
    placement = navigation.placement() if execution == 'process' else None
    started, arrived, collided = monotonic(), False, False
    minimum_boundary_clearance = float('inf')
    samples = []
    computation_samples = {'local': [], 'execution': []}
    computation_sequences = {'local': 0, 'execution': 0}
    pids = {'local': set(), 'execution': set(), 'global': set()}
    previous = 0
    async with Runtime(*output.roots, device.pose, device.state, navigation.diagnostics,
                       navigation.execution.diagnostics, navigation.global_planning.routes,
                       placement=placement):
        while monotonic()-started < duration:
            await asyncio.sleep(.03)
            for component in (inputs, navigation.local, navigation.global_planning, navigation.execution, output):
                if component._failure is not None:
                    raise RuntimeError(f'navigation component failed: {type(component).__name__}') from component._failure
            rows = device.state.frames
            if not rows or rows[-1].sequence == previous or not device.pose.frames:
                continue
            state, pose = rows[-1], device.pose.frames[-1]
            previous = state.sequence
            x, y = map(float, pose.data.position[:2])
            forward = pose.data.rotation.as_matrix()[:, 0]
            yaw = math.atan2(float(forward[1]), float(forward[0]))
            xmin, xmax, ymin, ymax = map_data['bounds']
            # Independent conservative footprint-vs-room check, alongside any
            # provider collision telemetry. Normal wheel/floor contacts are not collisions.
            corners = config.footprint.corners(pose2d(x, y, yaw, frame=map_data['frame']))
            clearance = min(float((corners[:, 0]-xmin).min()), float((xmax-corners[:, 0]).min()),
                            float((corners[:, 1]-ymin).min()), float((ymax-corners[:, 1]).min()))
            minimum_boundary_clearance = min(minimum_boundary_clearance, clearance)
            collided |= bool(state.data.get('collided', False)) or clearance <= 0
            speed, turn = float(state.data['linear_x']), float(state.data['angular_z'])
            samples.append({'stamp_ns': state.stamp_ns, 'episode': state.data['episode'],
                            'step_index': state.data['step_index'],
                            'session': state.metadata.get('session'),
                            'source_rate': state.metadata.get('source_rate', 1.),
                            'host_observed_ns': monotonic_ns(),
                            'pose': [x, y, yaw], 'speed': speed, 'turn': turn,
                            **{key: state.data[key] for key in ('command_accepted', 'command_expirations',
                               'command_host_deadline_ns', 'command_source_deadline_ns', 'applied_command')
                               if key in state.data}})
            for key, signal in (('local', navigation.diagnostics), ('execution', navigation.execution.diagnostics),
                                ('global', navigation.global_planning.routes)):
                for frame in signal.frames:
                    pids[key].add(frame.data['pid'])
                    if key in computation_samples and frame.sequence > computation_sequences[key]:
                        computation_samples[key].append({'stamp_ns': frame.stamp_ns,
                            'context': {name: frame.metadata.get(name) for name in ('session', 'episode', 'step_index')},
                            **frame.data})
                        computation_sequences[key] = frame.sequence
            status = navigation.diagnostics.frames[-1].data['status'] if navigation.diagnostics.frames else None
            error = math.hypot(x-goal[0], y-goal[1])
            yaw_error = abs((yaw-goal[2]+math.pi) % (2*math.pi)-math.pi)
            if (status == 'arrived' and error <= config.navigation.goal_tolerance
                    and yaw_error <= config.navigation.yaw_tolerance and abs(speed) < .02 and abs(turn) < .02):
                arrived = True
                break
            if collided:
                break
        final_state = (await device.state.get(timeout=5)).data
        pose_kind = final_state['pose_kind']
        diagnostics = navigation.diagnostics.frames[-1].data if navigation.diagnostics.frames else {}
        execution_diagnostics = (navigation.execution.diagnostics.frames[-1].data
                                 if navigation.execution.diagnostics.frames else {})
        pipeline = {name: {'failure': None if component._failure is None else repr(component._failure),
                           'last_stamp_ns': signal.frames[-1].stamp_ns if signal.frames else None}
                    for name, component, signal in (
                        ('inputs', inputs, inputs.states), ('local', navigation.local, navigation.diagnostics),
                        ('execution', navigation.execution, navigation.execution.diagnostics),
                        ('command_output', output, output.commands))}
    elapsed = monotonic()-started
    source_delta = (samples[-1]['stamp_ns']-samples[0]['stamp_ns'])/1e9 if len(samples)>1 else 0.
    observed_wall = (samples[-1]['host_observed_ns']-samples[0]['host_observed_ns'])/1e9 if len(samples)>1 else 0.
    return {'arrived': arrived, 'collided': collided, 'elapsed_host_s': elapsed,
            'localization': pose_kind, 'planar_projection': planar_projection, 'map': 'fixed_prebuilt',
            'clock': config.clock, 'execution': execution, 'scheduling': config.scheduling,
            'environment_policy': config.environment_policy,
            'configuration_fingerprint': config.fingerprint(), 'goal': list(goal),
            'command_limits': {'speed_mps': config.command_speed_limit, 'yaw_radps': config.command_yaw_limit},
            'minimum_boundary_clearance_m': minimum_boundary_clearance if samples else None,
            'samples': samples, 'pids': {key: sorted(value) for key, value in pids.items()},
            'computation_samples': computation_samples,
            'source_delta_s': source_delta, 'observed_wall_s': observed_wall,
            'rtf': source_delta/observed_wall if observed_wall else 0.,
            'rtf_window': 'first_to_last_observed_sample',
            'source_delta_per_total_host_s': source_delta/elapsed,
            'navigation_compute_steps': diagnostics.get('source_samples', 0),
            'command_drops': output.drive.dropped if output.drive is not None else 0,
            'navigation_compute_steps_per_host_s': diagnostics.get('source_samples', 0)/elapsed,
            'last_status': diagnostics.get('status'), 'last_reason': diagnostics.get('reason'),
            'last_execution': execution_diagnostics,
            'pipeline': pipeline,
            'environment': {'executable': sys.executable, 'sys_path': sys.path, 'pid': os.getpid(),
                            'mujoco_available': importlib.util.find_spec('mujoco') is not None,
                            'mujocosim_available': importlib.util.find_spec('mujocosim') is not None}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--key', required=True)
    parser.add_argument('--clock', default='simulation')
    parser.add_argument('--domain', type=int, default=0)
    parser.add_argument('--goal', type=float, nargs=3, default=(.8, 0., 0.))
    parser.add_argument('--bounds', type=float, nargs=4, default=(-3., 3., -3., 3.))
    parser.add_argument('--map', type=Path, help='JSON points/frame/revision/static/bounds map packet')
    parser.add_argument('--execution', choices=('local', 'thread', 'process'), default='thread')
    parser.add_argument('--scheduling', choices=('realtime', 'environment'), default='realtime')
    parser.add_argument('--duration', type=float, default=15.)
    parser.add_argument('--strict-planar-pose', action='store_true', help='reject z/roll/pitch instead of projecting')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error('duration must be positive')
    data = json.loads(args.map.read_text()) if args.map else room_map(args.bounds)
    data['points'] = np.asarray(data['points'], dtype=float)
    device = ChassisPortsDevice(key=args.key, clock=args.clock, transport=TransportConfig(domain_id=args.domain))
    report = asyncio.run(navigate(device, map_data=data, goal=args.goal, execution=args.execution,
                                 duration=args.duration, planar_projection=not args.strict_planar_pose,
                                 scheduling=args.scheduling))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('samples', 'computation_samples')},
                     ensure_ascii=False, allow_nan=False))
    return 0 if report['arrived'] and not report['collided'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
