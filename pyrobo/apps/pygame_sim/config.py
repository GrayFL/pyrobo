"""Pygame CLI; configuration is shared with the web workbench."""
import argparse
from pathlib import Path
from pyrobo.apps.sim_common.config import (DEFAULTS, build_parameters, load_config, make_world,
                                       merge, parameters, validate)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog='python -m pyrobo.apps.pygame_sim',
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
