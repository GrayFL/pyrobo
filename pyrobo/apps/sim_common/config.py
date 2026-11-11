"""TOML configuration and explicit backend construction, without a pygame import."""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
import tomllib

import numpy as np

from pyrobo import (Ackermann, DynamicsConfig, Drag, Footprint, NavigationConfig, SafetyConfig)
from pyrobo.core import Bounds2D, pose2d
from pyrobo.registry import LOCAL_OPTIMIZERS, GLOBAL_PLANNERS, KINEMATIC_MODELS
from pyrobo.simulation import Box, World2D, demo_world

DEFAULTS = {
    'app': {'width': 1280, 'height': 820, 'fps': 60, 'dt': .1, 'point_spacing': .04,
            'trail_length': 2400, 'input_timeout': .35, 'start_mode': 'paused'},
    'backend': {'planner': 'rrt_connect', 'model': 'differential', 'local_optimizer': 'dwa',
                'adaptive_clearance': True},
    'map': {'preset': 'detour'},
    'footprint': {}, 'dynamics': {},
    **{name: {} for name in KINEMATIC_MODELS},
    **{name: {} for name in LOCAL_OPTIMIZERS},
    **{name: {} for name in GLOBAL_PLANNERS}, 'navigation': {}, 'safety': {},
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


def controller_parameters(config, name=None):
    name = name or config['backend']['local_optimizer']
    lookahead = 'track_lookahead_m' if LOCAL_OPTIMIZERS[name].family == 'eb_dwa' else 'lookahead'
    cls = LOCAL_OPTIMIZERS[name].config
    defaults = ({'prediction_time': 2.4, lookahead: 1.6}
                if KINEMATIC_MODELS[config['backend']['model']].family == 'ackermann' else {})
    defaults = {k: v for k, v in defaults.items() if k in {f.name for f in fields(cls)}}
    return parameters(cls, {**defaults, **config.get(name, {})}, name)


def build_parameters(config):
    """Only this example's composition root selects concrete implementations."""
    backend = config['backend']
    model = parameters(KINEMATIC_MODELS[backend['model']].config,
                       config[backend['model']], backend['model'])
    footprint = parameters(Footprint, config['footprint'], 'footprint')
    dynamics = dict(config['dynamics'])
    for name in ('longitudinal_drag', 'yaw_drag', 'steering_drag'):
        if name in dynamics:
            dynamics[name] = parameters(Drag, dynamics[name], f'dynamics.{name}')
    dynamics = parameters(DynamicsConfig, dynamics, 'dynamics')
    controller = controller_parameters(config)
    clearance = .12 + (.4 * model.minimum_turn_radius if isinstance(model, Ackermann) else 0.)
    spec = GLOBAL_PLANNERS[backend['planner']]
    defaults = {'clearance': clearance, **dict(spec.defaults)}
    defaults = {k: v for k, v in defaults.items() if k in {f.name for f in fields(spec.config)}}
    planner = parameters(spec.config, {**defaults, **config[backend['planner']]},
                         backend['planner'])
    safety = parameters(SafetyConfig, config['safety'], 'safety')
    return model, footprint, dynamics, controller, planner, safety


def default_config():
    result = deepcopy(DEFAULTS)
    for registry in (LOCAL_OPTIMIZERS, GLOBAL_PLANNERS, KINEMATIC_MODELS):
        for name in registry:
            result.setdefault(name, {})
    return result


def validate(config):
    config = merge(default_config(), config)
    known_keys(config, default_config(), 'root')
    for section in ('app', 'backend', 'teleop', 'joystick'):
        known_keys(config[section], DEFAULTS[section], section)
    known_keys(config['map'], ('preset', 'bounds', 'boxes', 'start', 'goal'), 'map')
    backend, app, teleop, joystick = (config[key] for key in ('backend', 'app', 'teleop', 'joystick'))
    if not isinstance(backend['adaptive_clearance'], bool):
        raise ValueError('backend.adaptive_clearance must be boolean')
    for key, choices in (('model', tuple(KINEMATIC_MODELS)), ('planner', tuple(GLOBAL_PLANNERS)),
                         ('local_optimizer', tuple(LOCAL_OPTIMIZERS))):
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
    for registry in (KINEMATIC_MODELS, GLOBAL_PLANNERS):
        for key, spec in registry.items():
            parameters(spec.config, config[key], key)
    parameters(NavigationConfig, config['navigation'], 'navigation')
    for name in LOCAL_OPTIMIZERS:
        controller_parameters(config, name)
    model, footprint, dynamics, controller, planner, safety = build_parameters(config)
    dt = app['dt']
    if dt > min(getattr(controller, 'prediction_time', safety.max_control_dt), controller_parameters(config, 'dwa').prediction_time,
                dynamics.timeout, safety.max_control_dt, teleop['ttl']) + 1e-9:
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
    config = merge(default_config(), data)
    for key, value in (selectors or {}).items():
        if value is not None:
            section, name = key.split('.')
            config[section][name] = value
    for assignment in overrides:
        if '=' not in assignment:
            raise ValueError('--set expects section.key=TOML_VALUE')
        key, raw = assignment.split('=', 1)
        parts = key.strip().split('.')
        if len(parts) < 2 or parts[0] not in default_config() or any(not p.isidentifier() for p in parts):
            raise ValueError(f'invalid setting path: {key}')
        value = tomllib.loads(f'value = {raw}')['value']
        target = config
        for part in parts[:-1]:
            if not isinstance(target.get(part, {}), dict):
                raise ValueError(f'{part} is not a table')
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return validate(config)
