"""TOML configuration and explicit backend construction, without a pygame import."""
import argparse
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import tomllib

import numpy as np

from pyrobo import (Ackermann, DifferentialDrive, DWAConfig, DynamicsConfig, Drag, Footprint,
                    NavigationConfig, RRTConnectConfig, SafetyConfig, VoronoiConfig)
from pyrobo.core import Bounds2D, pose2d
from pyrobo.simulation import Box, World2D, demo_world

DEFAULTS = {
    'app': {'width': 1280, 'height': 820, 'fps': 60, 'dt': .1, 'point_spacing': .04,
            'trail_length': 2400, 'input_timeout': .35, 'start_mode': 'paused'},
    'backend': {'planner': 'rrt_connect', 'model': 'differential', 'local_optimizer': 'dwa'},
    'map': {'preset': 'detour'},
    'footprint': {}, 'differential': {}, 'ackermann': {}, 'dynamics': {},
    'dwa': {}, 'rrt_connect': {}, 'voronoi': {}, 'navigation': {}, 'safety': {},
    'teleop': {'policy': 'teleop_stop', 'speed_limit': .6, 'turn_limit': 0., 'ttl': .3,
               'max_turn_deviation': .8, 'max_lateral_deviation': .8, 'max_heading_deviation': 1.2},
    'joystick': {'enabled': True, 'device_index': 0, 'throttle_axis': 1, 'turn_axis': 0,
                 'invert_throttle': True, 'invert_turn': True, 'deadzone': .12, 'deadman_button': 0},
}


def merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def known_keys(values, allowed, label):
    if not isinstance(values, dict):
        raise ValueError(f'{label} must be a TOML table')
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f'unknown {label} setting(s): {", ".join(sorted(unknown))}')


def parameters(cls, values, label):
    known_keys(values, (f.name for f in fields(cls)), label)
    try:
        return cls(**values)
    except (ValueError, TypeError) as error:
        raise ValueError(f'{label}: {error}') from error


def finite_number(value, name, *, minimum=0., allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < minimum or (value == minimum and not allow_zero):
        raise ValueError(f'{name} must be {">=" if allow_zero else ">"} {minimum}')


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def make_world(config):
    data = config['map']
    preset = data['preset']
    if preset == 'detour':
        world, start, goal = demo_world()
    elif preset == 'empty':
        world = World2D(Bounds2D(-5, 5, -4, 4))
        start, goal = pose2d(-3, 0), pose2d(3, 0)
    elif preset == 'custom':
        if 'bounds' not in data or 'start' not in data:
            raise ValueError('map.preset="custom" requires bounds and start')
        world = World2D(Bounds2D(*data['bounds']))
        start, goal = pose2d(*data['start']), None
    else:
        raise ValueError('map.preset must be detour, empty or custom')
    bounds = Bounds2D(*data['bounds']) if 'bounds' in data else world.bounds
    boxes = [Box(*values) for values in data['boxes']] if 'boxes' in data else world.boxes
    world = World2D(bounds, boxes)
    if 'start' in data:
        start = pose2d(*data['start'])
    if 'goal' in data:
        goal = pose2d(*data['goal'])
    return world, start, goal


def build_parameters(config):
    """Only this example's composition root selects concrete implementations."""
    backend = config['backend']
    model = parameters(DifferentialDrive if backend['model'] == 'differential' else Ackermann,
                       config[backend['model']], backend['model'])
    footprint = parameters(Footprint, config['footprint'], 'footprint')
    dynamics = dict(config['dynamics'])
    for name in ('longitudinal_drag', 'yaw_drag', 'steering_drag'):
        if name in dynamics:
            dynamics[name] = parameters(Drag, dynamics[name], f'dynamics.{name}')
    dynamics = parameters(DynamicsConfig, dynamics, 'dynamics')
    defaults = {} if backend['model'] == 'differential' else {'prediction_time': 2.4, 'lookahead': 1.6}
    dwa = parameters(DWAConfig, {**defaults, **config['dwa']}, 'dwa')
    clearance = .12 + (.4 * model.minimum_turn_radius if isinstance(model, Ackermann) else 0.)
    planner_cls = RRTConnectConfig if backend['planner'] == 'rrt_connect' else VoronoiConfig
    budget = .5 if backend['planner'] == 'rrt_connect' else 2.
    planner = parameters(planner_cls, {'clearance': clearance, 'time_budget': budget, **config[backend['planner']]},
                         backend['planner'])
    safety = parameters(SafetyConfig, config['safety'], 'safety')
    return model, footprint, dynamics, dwa, planner, safety


def validate(config):
    known_keys(config, DEFAULTS, 'root')
    for section in ('app', 'backend', 'teleop', 'joystick'):
        known_keys(config[section], DEFAULTS[section], section)
    known_keys(config['map'], ('preset', 'bounds', 'boxes', 'start', 'goal'), 'map')
    backend, app, teleop, joystick = (config[key] for key in ('backend', 'app', 'teleop', 'joystick'))
    for key, choices in (('model', ('differential', 'ackermann')), ('planner', ('rrt_connect', 'voronoi')),
                         ('local_optimizer', ('dwa',))):
        if backend[key] not in choices:
            raise ValueError(f'backend.{key} must be one of {choices}')
    for key, minimum in (('width', 1000), ('height', 720), ('fps', 1), ('trail_length', 2)):
        integer(app[key], f'app.{key}', minimum)
    for key in ('dt', 'point_spacing', 'input_timeout'):
        finite_number(app[key], f'app.{key}')
    if app['start_mode'] not in ('paused', 'manual', 'auto'):
        raise ValueError('app.start_mode must be paused, manual or auto')
    if teleop['policy'] not in ('teleop_stop', 'teleop_avoid'):
        raise ValueError('teleop.policy must be teleop_stop or teleop_avoid')
    for key in ('speed_limit', 'ttl'):
        finite_number(teleop[key], f'teleop.{key}')
    for key in ('turn_limit', 'max_turn_deviation', 'max_lateral_deviation', 'max_heading_deviation'):
        finite_number(teleop[key], f'teleop.{key}', allow_zero=True)
    for key in ('enabled', 'invert_throttle', 'invert_turn'):
        if not isinstance(joystick[key], bool):
            raise ValueError(f'joystick.{key} must be boolean')
    for key in ('device_index', 'throttle_axis', 'turn_axis'):
        integer(joystick[key], f'joystick.{key}')
    integer(joystick['deadman_button'], 'joystick.deadman_button', -1)
    finite_number(joystick['deadzone'], 'joystick.deadzone', allow_zero=True)
    if joystick['deadzone'] >= 1:
        raise ValueError('joystick.deadzone must be < 1')
    # Check inactive backend tables too, so switching a CLI selector is predictable.
    for cls, key in ((DifferentialDrive, 'differential'), (Ackermann, 'ackermann'),
                     (RRTConnectConfig, 'rrt_connect'), (VoronoiConfig, 'voronoi'),
                     (NavigationConfig, 'navigation')):
        parameters(cls, config[key], key)
    model, footprint, dynamics, dwa, planner, safety = build_parameters(config)
    dt = app['dt']
    if dt > min(dwa.prediction_time, dynamics.timeout, safety.max_control_dt, teleop['ttl']) + 1e-9:
        raise ValueError('app.dt exceeds prediction horizon, watchdog, safety period or teleop TTL')
    if safety.expected_dt is not None and not np.isclose(dt, safety.expected_dt, rtol=0, atol=1e-9):
        raise ValueError('app.dt must equal safety.expected_dt')
    world, start, goal = make_world(config)
    if world.collides(start, footprint, padding=footprint.margin):
        raise ValueError('map.start intersects an obstacle or boundary, including footprint margin')
    if goal is not None and world.collides(goal, footprint, padding=footprint.margin):
        raise ValueError('map.goal intersects an obstacle or boundary, including footprint margin')
    return config


def load_config(path=None, overrides=(), selectors=None):
    data = {}
    if path:
        with Path(path).open('rb') as stream:
            data = tomllib.load(stream)
    config = merge(DEFAULTS, data)
    for key, value in (selectors or {}).items():
        if value is not None:
            section, name = key.split('.')
            config[section][name] = value
    for assignment in overrides:
        if '=' not in assignment:
            raise ValueError('--set expects section.key=TOML_VALUE')
        key, raw = assignment.split('=', 1)
        parts = key.strip().split('.')
        if len(parts) < 2 or parts[0] not in DEFAULTS or any(not p.isidentifier() for p in parts):
            raise ValueError(f'invalid setting path: {key}')
        value = tomllib.loads(f'value = {raw}')['value']
        target = config
        for part in parts[:-1]:
            if not isinstance(target.get(part, {}), dict):
                raise ValueError(f'{part} is not a table')
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return validate(config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python -m examples.pygame_sim',
                                     description='Pygame robot workbench: WASD / gamepad / virtual stick / click-to-go.')
    parser.add_argument('--config', type=Path, help='TOML file; unspecified settings retain model-aware defaults')
    parser.add_argument('--planner', choices=('rrt_connect', 'voronoi'))
    parser.add_argument('--local-optimizer', choices=('dwa',))
    parser.add_argument('--model', choices=('differential', 'ackermann'))
    parser.add_argument('--map', dest='map_preset', choices=('detour', 'empty', 'custom'))
    parser.add_argument('--policy', choices=('teleop_stop', 'teleop_avoid'))
    parser.add_argument('--dt', type=float)
    parser.add_argument('--speed', type=float)
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=TOML_VALUE', help='repeatable; highest priority')
    parser.add_argument('--autostart', action='store_true', help='navigate to the configured goal immediately')
    parser.add_argument('--headless', action='store_true', help='use SDL dummy display and deterministic simulation ticks')
    parser.add_argument('--steps', type=int, help='exit after this many simulation ticks (headless default: 100)')
    parser.add_argument('--screenshot', type=Path, help='save the final window as PNG')
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    selectors = {'backend.planner': args.planner, 'backend.local_optimizer': args.local_optimizer,
                 'backend.model': args.model, 'map.preset': args.map_preset, 'teleop.policy': args.policy,
                 'app.dt': args.dt, 'teleop.speed_limit': args.speed,
                 'app.start_mode': 'auto' if args.autostart else None}
    try:
        config = load_config(args.config, args.set, selectors)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    return args, config
