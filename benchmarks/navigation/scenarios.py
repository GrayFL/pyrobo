"""Fixed continuous-geometry problems; these are not imported occupancy grids."""
from dataclasses import dataclass
import json
from pathlib import Path
from importlib.resources import files
from types import SimpleNamespace

import numpy as np

from pyrobo import Bounds2D, pose2d
from pyrobo.simulation.world import Box, World2D
from pyrobo.apps.web_sim.world import PolygonWorld


@dataclass(frozen=True)
class Scenario:
    name: str
    world: object
    start: object
    goal: object
    expected_path: bool = True
    description: str = ''
    point_spacing: float = .04
    source_scene: dict | None = None


def saved_scene(path):
    data = json.loads(Path(path).read_text())
    return from_scene(data, name='narrow_ackermann')


def from_scene(data, *, name=None):
    """Construct geometry from an already validated, normalized Web scene."""
    m = data['config']['map']
    world = World2D(Bounds2D(*m['bounds']), [Box(*box) for box in m['boxes']])
    world = PolygonWorld(world, data.get('polygons', []),
                         [SimpleNamespace(**line) for line in data.get('polylines', [])])
    return Scenario(name or data['name'], world, pose2d(*m['start']), pose2d(*m['goal']),
                    description='Saved geometry and poses; no obstacle edits.',
                    point_spacing=data['config'].get('app', {}).get('point_spacing', .04), source_scene=data)


def scenarios():
    bounds = Bounds2D(0, 16, 0, 12)
    def make(name, start, goal, boxes=(), expected=True, description=''):
        return Scenario(name, World2D(bounds, [Box(*b) for b in boxes]), pose2d(*start),
                        pose2d(*goal), expected, description)
    return {
        s.name: s for s in [
            make('straight', (1.5, 6, 0), (14, 6, 0), description='Cruise speed and straight-line oscillation.'),
            make('offset_gate', (1.5, 3, 0), (14, 9, 0),
                 [(7, 7.4, 0, 5.3), (7, 7.4, 6.7, 12)], description='1.4 m off-axis doorway.'),
            make('staggered', (1.5, 2, 0), (14, 10, np.pi/2),
                 [(4, 4.7, 0, 7), (9, 9.7, 5, 12)], description='Alternating walls; route corners need blending.'),
            make('u_trap', (7, 5, np.pi/2), (7, 2, 0),
                 [(4, 4.4, 3, 8), (4, 10, 3, 3.4), (9.6, 10, 3, 8)],
                 description='Goal behind U wall; escape initially increases goal distance.'),
            make('slalom', (1.5, 6, 0), (14, 6, 0),
                 [(4, 4.7, 0, 6.6), (7, 7.7, 5.4, 12), (10, 10.7, 0, 6.6)],
                 description='Three alternating gaps and several consecutive turns.'),
            make('warehouse', (1.5, 1.5, 0), (14, 10.5, 0),
                 [(x, x+1, y, y+2) for x in (4, 7, 10) for y in (2, 7)],
                 description='Repeated aisles with multiple homotopy classes.'),
            make('blocked', (2, 6, 0), (14, 6, 0), [(7, 8, 0, 12)], expected=False,
                 description='Deliberately disconnected; failure must not be reported as success.'),
            saved_scene(files('pyrobo.apps').joinpath('scenes/narrow_ackermann_001.json')),
        ]
    }
