"""Wall-clock simulation with independent RSim planning/control/execution.

The plant keeps advancing while planning is blocked. No hardware is opened.
"""
import argparse
import asyncio
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from time import monotonic, monotonic_ns, perf_counter
from uuid import uuid4

import numpy as np
from rsim import Component, CommandSink, Connect, Runtime

from pyrobo import Ackermann, DifferentialDrive, DWAConfig, NavigationConfig
from pyrobo.core.geometry import planar_state, wrap_angle
from pyrobo.geometry.collision import PointObstacles
from pyrobo.rsim import StackConfig, NavigationStack
from pyrobo.rsim.packets import CLOCK, latest
from pyrobo.planning import RRTConnectConfig, VoronoiConfig
from pyrobo.simulation import Simulator2D
from .planning import provenance
from .scenarios import scenarios, from_scene


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None  # unavailable clearance is not a valid finite measurement
    return value


class Simulation(Component):
    """Independent actuator with deadlines and distinct truth / sensor Signals."""
    def __init__(self, case, config, *, sensor_pause_period=0., sensor_pause_s=0.,
                 map_pause_period=0., map_pause_s=0.):
        super().__init__()
        self.case, self.config = case, config
        self.pause_period, self.pause_s = sensor_pause_period, sensor_pause_s
        self.map_pause_period, self.map_pause_s = map_pause_period, map_pause_s
        self.states = self.signal('states', history=4, clock=CLOCK)
        self.truth = self.signal('truth', history=8, clock=CLOCK)
        self.maps = self.signal('maps', history=2, clock=CLOCK)
        self.missions = self.signal('missions', history=2, clock=CLOCK)
        self.command = CommandSink(self, 'command', self.receive,
                                   fallback={'native': [0., 0.], 'mode': 'estop'}, safe=self.stop, hz=100)

    async def open(self):
        now = monotonic()
        cfg = self.config
        self.sim = Simulator2D(self.case.world, self.case.start, model=cfg.model, footprint=cfg.footprint,
                               dynamics=cfg.dynamics, timestamp=now)
        self.origin = now
        self.applied = cfg.model.decode_command([0., 0.], 'estop')
        self.deadline = self.issued_at = now
        self.received = self.expired = 0
        self.max_physics_gap = 0.
        cloud = self.case.world.point_cloud(self.case.point_spacing)
        b = self.case.world.bounds
        self.map_packet = {'points': cloud.points, 'frame': cloud.frame, 'revision': 1,
                           'static': not bool(self.map_pause_period),
                           'bounds': [b.xmin, b.xmax, b.ymin, b.ymax]}
        self.last_map = now
        await self.maps.publish(self.map_packet, stamp_ns=int(now*1e9), clock=CLOCK)
        await self.missions.publish({'epoch': uuid4().hex, 'goal': planar_state(self.case.goal), 'mode': 'auto'},
                                    stamp_ns=int(now*1e9), clock=CLOCK)
        await self.publish()
        self.task('physics', self.tick, hz=1/cfg.execution_period)

    def advance(self, now):
        start = self.sim.state.timestamp
        if now <= start or self.sim.collided:
            return
        self.max_physics_gap = max(self.max_physics_gap, now-start)
        # Honor expiry inside a missed interval, even when this event loop was
        # unavailable. A newly arrived command never acts retroactively.
        if start < self.deadline:
            end = min(now, self.deadline)
            self.sim.step(self.applied, end-start, command_age=max(0., start-self.issued_at))
            start = end
        if start < now and not self.sim.collided:
            if self.applied.mode == 'track':
                self.expired += 1
            self.applied = self.config.model.decode_command([0., 0.], 'estop')
            self.sim.step(self.applied, now-start)

    async def receive(self, envelope):
        now = monotonic()
        self.advance(now)
        self.applied = self.config.model.decode_command(envelope.value['native'], envelope.value['mode'])
        self.issued_at = now
        self.deadline = envelope.deadline_ns/1e9
        self.received += 1

    async def stop(self, _fallback):
        now = monotonic()
        self.advance(now)
        self.applied = self.config.model.decode_command([0., 0.], 'estop')
        self.deadline = now

    async def publish(self):
        now = self.sim.state.timestamp
        packet = {'state': self.sim.plant.codec.pack(self.sim.state), 'frame': self.case.world.frame,
                  'time_s': now-self.origin, 'collided': self.sim.collided,
                  'actuator_expirations': self.expired, 'commands_received': self.received,
                  'max_physics_gap_ms': self.max_physics_gap*1000, 'pid': os.getpid()}
        await self.truth.publish(packet, stamp_ns=int(now*1e9), clock=CLOCK)
        phase = (now-self.origin) % self.pause_period if self.pause_period else self.pause_s
        paused = self.pause_period > 0 and now-self.origin >= self.pause_period and phase < self.pause_s
        if not paused:
            await self.states.publish(packet, stamp_ns=int(now*1e9), clock=CLOCK)
        phase = (now-self.origin) % self.map_pause_period if self.map_pause_period else self.map_pause_s
        map_paused = self.map_pause_period > 0 and now-self.origin >= self.map_pause_period and phase < self.map_pause_s
        if self.map_pause_period and not map_paused and now-self.last_map >= .1:
            await self.maps.publish(self.map_packet, stamp_ns=int(now*1e9), clock=CLOCK)
            self.last_map = now

    async def tick(self):
        self.advance(monotonic())
        await self.publish()


async def run(case, config, *, duration=75., processes=True, control_delay_every=0, control_delay_s=0.,
              planning_delay_s=0., sensor_pause_period=0., sensor_pause_s=0.,
              map_pause_period=0., map_pause_s=0.):
    environment = provenance()
    sim = Simulation(case, config, sensor_pause_period=sensor_pause_period, sensor_pause_s=sensor_pause_s,
                     map_pause_period=map_pause_period, map_pause_s=map_pause_s)
    stack = NavigationStack(sim.states, sim.maps, sim.missions, config,
                            local_options={'delay_every': control_delay_every, 'delay_s': control_delay_s,
                                           'offload': not processes},
                            global_options={'delay_s': planning_delay_s, 'offload': not processes})
    local, global_plan, execute = stack.local, stack.global_planning, stack.execution
    drive = Connect(execute.commands, sim.command, ttl=.1, hz=100)
    placement = stack.placement() if processes else None
    samples, controls, executions, plans = [], [], [], []
    cursors = {'control': 0, 'execution': 0, 'plan': 0}
    arrived = False
    started = perf_counter()
    async with Runtime(sim.truth, local.diagnostics, execute.diagnostics, global_plan.routes, drive,
                       placement=placement):
        current = await sim.truth.get(timeout=30.)
        deadline = monotonic()+duration
        while monotonic() < deadline:
            current = await sim.truth.get(after=current.sequence, timeout=5.)
            data = current.data
            state = data['state']
            samples.append({'t': data['time_s'], 'state': state.tolist()})
            for name, signal, output in [('control', local.diagnostics, controls),
                                          ('execution', execute.diagnostics, executions),
                                          ('plan', global_plan.routes, plans)]:
                item = latest(signal)
                if item is not None and item.sequence != cursors[name]:
                    cursors[name] = item.sequence
                    output.append({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in item.data.items()})
            if data['collided']:
                break
            goal = planar_state(case.goal)
            arrived = (controls and controls[-1]['status'] == 'arrived'
                       and np.linalg.norm(state[:2]-goal[:2]) <= config.navigation.goal_tolerance
                       and abs(wrap_angle(state[2]-goal[2])) <= config.navigation.yaw_tolerance
                       and max(abs(state[3:5])) < .001)
            if arrived:
                break
        terminal = dict(current.data)
    states = np.array([sample['state'] for sample in samples]).reshape(-1, len(terminal['state']))
    def latency(rows, key):
        values = [r[key] for r in rows]
        return {f'p{q}_ms': float(np.percentile(values, q)) if values else None for q in (50, 95, 99)} | {
            'max_ms': max(values) if values else None, 'over_100ms': sum(v > 100 for v in values),
            'over_400ms': sum(v > 400 for v in values), 'samples': len(values)}
    gaps = PointObstacles(case.world.point_cloud(case.point_spacing)).clearances(states[:, :3], config.footprint,
                                                                case.world.bounds) if len(states) else []
    turn = states[:, 5] if isinstance(config.model, Ackermann) else states[:, 4]
    signs = np.sign(turn[abs(turn) > .03])
    moving = abs(states[:, 3]) > .03
    report = {'schema': 1, 'kind': 'rsim_wall_clock_navigation', 'environment': environment,
              'scenario': case.name, 'processes': processes, 'arrived': bool(arrived),
              'scene': case.source_scene,
              'collided': terminal['collided'], 'elapsed_wall_s': perf_counter()-started,
              'simulation_s': terminal['time_s'], 'model': type(config.model).__name__,
              'time_origin': current.stamp_ns/1e9-terminal['time_s'],
              'planner': type(config.planner).__name__, 'config': asdict(config),
              'faults': {'control_delay_every': control_delay_every, 'control_delay_s': control_delay_s,
                         'planning_delay_s': planning_delay_s, 'sensor_pause_period': sensor_pause_period,
                         'sensor_pause_s': sensor_pause_s, 'map_pause_period': map_pause_period,
                         'map_pause_s': map_pause_s},
              'control_latency': latency(controls, 'compute_ms'),
              'execution_latency': latency(executions, 'compute_ms'),
              'planning_latency': latency([{'ms': p['elapsed_s']*1000} for p in plans], 'ms'),
              'max_physics_gap_ms': terminal['max_physics_gap_ms'],
              'actuator_expirations': terminal['actuator_expirations'],
              'pids': {'simulation': terminal['pid'], 'global': sorted({p['pid'] for p in plans}),
                       'local': sorted({c['pid'] for c in controls}), 'execution': sorted({e['pid'] for e in executions})},
              'maximum_speed_mps': float(np.max(states[:, 3])) if len(states) else 0.,
              'minimum_margin_clearance_m': float(np.min(gaps)) if len(gaps) else None,
              'moving_speed_mean_mps': float(np.mean(abs(states[moving, 3]))) if moving.any() else 0.,
              'turn_reversals': int(np.count_nonzero(np.diff(signs))) if len(signs) else 0,
              'turn_total_variation': float(np.abs(np.diff(turn)).sum()),
              'execution_reasons': {reason: sum(e['reason'] == reason for e in executions)
                                    for reason in sorted({e['reason'] for e in executions})},
              'distance_m': float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()) if len(states) else 0.,
              'last_execution': executions[-1] if executions else None,
              'samples': samples, 'controls': controls, 'executions': executions, 'plans': plans}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=list(scenarios()), default='straight')
    parser.add_argument('--scene', type=Path, help='saved Web scene, including model and navigation configuration')
    parser.add_argument('--model', choices=['differential', 'ackermann'])
    parser.add_argument('--planner', choices=['rrt_connect', 'voronoi'])
    parser.add_argument('--duration', type=float, default=75.)
    parser.add_argument('--max-speed', type=float)
    parser.add_argument('--clearance', type=float)
    parser.add_argument('--approach', type=float, default=None)
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--control-delay-every', type=int, default=0)
    parser.add_argument('--control-delay', type=float, default=0.)
    parser.add_argument('--planning-delay', type=float, default=0.)
    parser.add_argument('--sensor-pause-period', type=float, default=0.)
    parser.add_argument('--sensor-pause', type=float, default=0.)
    parser.add_argument('--map-pause-period', type=float, default=0.)
    parser.add_argument('--map-pause', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    car = args.model != 'differential'
    if args.duration <= 0 or min(args.control_delay, args.planning_delay, args.sensor_pause,
                                 args.sensor_pause_period, args.control_delay_every,
                                 args.map_pause, args.map_pause_period) < 0:
        parser.error('positive duration and nonnegative fault parameters required')
    if args.sensor_pause_period and args.sensor_pause >= args.sensor_pause_period:
        parser.error('sensor pause must be shorter than its period')
    if args.map_pause_period and args.map_pause >= args.map_pause_period:
        parser.error('map pause must be shorter than its period')
    if args.scene:
        from pyrobo.apps.web_sim.scene import Scene
        from pyrobo.apps.sim_common.config import build_parameters
        scene = Scene.model_validate_json(args.scene.read_bytes()).model_dump()
        data = scene['config']
        if args.model:
            data['backend']['model'] = args.model
        if args.planner:
            data['backend']['planner'] = args.planner
        model, footprint, dynamics, controller, planner, safety = build_parameters(data)
        config = StackConfig(model=model, footprint=footprint, dynamics=dynamics, controller=controller,
                             planner=planner, safety=safety, navigation=NavigationConfig(**data['navigation']),
                             control_period=data['app']['dt'],
                             adaptive_clearance=data['backend']['adaptive_clearance'])
        case = from_scene(scene)
    else:
        config = StackConfig(model=(Ackermann if car else DifferentialDrive)(),
                         controller=DWAConfig(prediction_time=2.4 if car else 1.8, lookahead=1.6 if car else 1.3),
                         planner=(VoronoiConfig if args.planner == 'voronoi' else RRTConnectConfig)(
                             time_budget=1., clearance=.08),
                         navigation=NavigationConfig(lookahead=1.6 if car else 1.3))
        case = scenarios()[args.scenario]
    if args.max_speed is not None:
        config = replace(config, model=replace(config.model, max_speed=args.max_speed))
    if args.clearance is not None:
        config = replace(config, planner=replace(config.planner, clearance=args.clearance))
    if args.approach is not None:
        config = replace(config, navigation=replace(config.navigation, goal_approach_distance=args.approach))
    report = asyncio.run(run(case, config, duration=args.duration, processes=not args.local,
                             control_delay_every=args.control_delay_every, control_delay_s=args.control_delay,
                             planning_delay_s=args.planning_delay, sensor_pause_period=args.sensor_pause_period,
                             sensor_pause_s=args.sensor_pause, map_pause_period=args.map_pause_period,
                             map_pause_s=args.map_pause))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = json_safe(report)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('samples', 'controls', 'executions', 'plans', 'config', 'scene')}, indent=2))
    return 0 if report['arrived'] and not report['collided'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
