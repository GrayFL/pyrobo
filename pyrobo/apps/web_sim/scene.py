"""Versioned, portable scenarios and atomic server-side case storage."""
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyrobo.apps.sim_common.config import (build_parameters, controller_parameters, make_world, merge,
                                        parameters, validate, default_config)
from pyrobo.apps.sim_common.session import Session
from pyrobo import Ackermann, NavigationConfig
from pyrobo.core.geometry import planar_state
from pyrobo.core import PointCloud, Twist2D
from pyrobo.registry import LOCAL_OPTIMIZERS, GLOBAL_PLANNERS, KINEMATIC_MODELS
from pyrobo.simulation import Box, World2D
from .world import PolygonWorld
from .experiment import Experiment, InitialRoutePlanner

MAX_SCENE_BYTES = 2_000_000


def finite_json(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError('scene numbers must be finite')


class Polyline(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    points: list[tuple[float, float]] = Field(min_length=2, max_length=1024)
    width: float = Field(default=.3, ge=.02, le=3.)

    @model_validator(mode='after')
    def valid_segments(self):
        if np.any(np.linalg.norm(np.diff(self.points, axis=0), axis=1) < 1e-6):
            raise ValueError('polyline has duplicate adjacent vertices')
        return self


class Geometry(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    boxes: list[tuple[float, float, float, float]] = Field(default_factory=list, max_length=256)
    polygons: list[list[tuple[float, float]]] = Field(default_factory=list, max_length=256)
    polylines: list[Polyline] = Field(default_factory=list, max_length=256)

    def world(self, bounds):
        return PolygonWorld(World2D(bounds, [Box(*b) for b in self.boxes]), self.polygons, self.polylines)

    def check(self, bounds, spacing=.01):
        from pyrobo.core import Bounds2D
        bounds = Bounds2D(*bounds)
        if max(bounds.xmax-bounds.xmin, bounds.ymax-bounds.ymin) > 200:
            raise ValueError('map dimensions must be <= 200 m')
        if sum(map(len, self.polygons))+sum(len(p.points) for p in self.polylines) > 4096:
            raise ValueError('geometry exceeds 4096 vertices')
        world = self.world(bounds)
        if any(not np.all(bounds.contains(box.corners)) for box in world.boxes):
            raise ValueError('box lies outside map bounds')
        if any(not np.all(bounds.contains(p)) for p in world.polygons):
            raise ValueError('polygon lies outside map bounds')
        if any(not np.all(bounds.contains(p-r)) or not np.all(bounds.contains(p+r)) for p, r in world.polylines):
            raise ValueError('polyline lies outside map bounds')
        segments = world.segments()
        if np.ceil(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1)/spacing).sum() > 100000:
            raise ValueError('geometry exceeds 100000 surface samples')
        return world


class Terrain(Geometry):
    schema_version: Literal[1] = 1
    kind: Literal['terrain'] = 'terrain'
    name: str = Field(default='地形', min_length=1, max_length=80)
    bounds: tuple[float, float, float, float]

    @model_validator(mode='after')
    def valid_geometry(self):
        self.check(self.bounds, .04)
        return self


class LidarSettings(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    max_range: float = Field(default=4., gt=0, le=50)
    beams: int = Field(default=360, ge=16, le=1440)


class Scene(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    schema_version: Literal[1] = 1
    name: str = Field(default='未命名场景', min_length=1, max_length=80)
    description: str = Field(default='', max_length=2000)
    config: dict = Field(default_factory=dict)
    polygons: list[list[tuple[float, float]]] = Field(default_factory=list, max_length=256)
    polylines: list[Polyline] = Field(default_factory=list, max_length=256)
    experiment: Experiment | None = None
    unknown_obstacles: Geometry = Field(default_factory=Geometry)
    lidar: LidarSettings = Field(default_factory=LidarSettings)

    @model_validator(mode='before')
    @classmethod
    def expand_document(cls, value):
        if not isinstance(value, dict) or value.get('schema_version', 1) != 2:
            return value
        data = deepcopy(value)
        terrain = Terrain.model_validate(data.pop('terrain', None))
        data['schema_version'] = 1
        config = data.setdefault('config', {})
        # A v2 document has one source of truth for terrain geometry.
        if set(config.get('map', {})) - {'start', 'goal'} or 'polygons' in data or 'polylines' in data:
            raise ValueError('v2 terrain geometry must only appear in terrain')
        config['map'] = {**config.get('map', {}), 'preset': 'custom',
                         'bounds': list(terrain.bounds), 'boxes': terrain.boxes}
        data['polygons'], data['polylines'] = terrain.polygons, terrain.polylines
        return data

    def terrain(self, name=None):
        m = self.config['map']
        return Terrain(name=name or self.name, bounds=m['bounds'], boxes=m['boxes'],
                       polygons=self.polygons, polylines=self.polylines)

    def document(self, terrain_name=None):
        data = self.model_dump(mode='json')
        data['schema_version'] = 2
        data['terrain'] = self.terrain(terrain_name).model_dump(mode='json')
        data.pop('polygons')
        data.pop('polylines')
        data['config']['map'] = {k: v for k, v in data['config']['map'].items() if k in ('start', 'goal')}
        return data

    @model_validator(mode='after')
    def normalize(self):
        finite_json(self.config)
        config = merge(default_config(), self.config)
        # Validate resource bounds before generating any point samples.
        spacing = config['app']['point_spacing']
        if not isinstance(spacing, (float, int)) or not .01 <= spacing <= .25:
            raise ValueError('point_spacing must be between 0.01 and 0.25 m')
        if sum(len(p) for p in self.polygons)+sum(len(p.points) for p in self.polylines) > 4096:
            raise ValueError('scene exceeds 4096 polygon/polyline vertices')
        if len(config['map'].get('boxes', [])) > 256:
            raise ValueError('scene exceeds 256 boxes')
        config = validate(config)
        base, start, goal = make_world(config)
        world = PolygonWorld(base, self.polygons, self.polylines)
        b = world.bounds
        self.unknown_obstacles.check([b.xmin, b.xmax, b.ymin, b.ymax], spacing)
        truth = PolygonWorld(World2D(b, (*world.boxes, *(Box(*x) for x in self.unknown_obstacles.boxes))),
                             [*self.polygons, *self.unknown_obstacles.polygons],
                             [*self.polylines, *self.unknown_obstacles.polylines])
        if max(b.xmax-b.xmin, b.ymax-b.ymin) > 200:
            raise ValueError('map dimensions must be <= 200 m')
        for polygon in world.polygons:
            if not np.all(b.contains(polygon)):
                raise ValueError('polygon lies outside map bounds')
        for points, radius in world.polylines:
            if not np.all(b.contains(points-radius)) or not np.all(b.contains(points+radius)):
                raise ValueError('polyline wall extends outside map bounds')
        segments = truth.segments()
        if len(segments)*self.lidar.beams > 2_000_000:
            raise ValueError('lidar workload exceeds 2000000 ray/segment pairs; reduce beams or geometry')
        if np.ceil(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1)/spacing).sum() > 100_000:
            raise ValueError('map exceeds 100000 surface samples; increase point_spacing')
        model, footprint, dynamics, controller, planner, safety = build_parameters(config)
        controllers = {key: controller_parameters(config, key) for key in LOCAL_OPTIMIZERS}
        eb = controllers['eb_dwa']
        if config['app']['dt'] < .02 or config['app']['trail_length'] > 10000:
            raise ValueError('web simulation needs dt >= 0.02 s and trail_length <= 10000')
        if safety.braking_timeout/min(dynamics.substep, config['app']['dt']) > 10000:
            raise ValueError('prediction workload exceeds the interactive limit; reduce samples/horizon or increase integration steps')
        # Validate inactive variants too: a later UI switch must remain bounded.
        for name, settings in controllers.items():
            family = LOCAL_OPTIMIZERS[name].family
            if family == 'dwa':
                samples = settings.linear_samples*settings.angular_samples
                candidates = (samples+64 if name == 'dwa' else
                              (settings.linear_samples+1)*(settings.angular_samples+1))
                work = candidates*np.ceil((settings.prediction_time+settings.braking_timeout)/
                                            min(dynamics.substep, settings.integration_dt))
                if samples > 5000 or work > 2_000_000:
                    raise ValueError(f'{name} prediction workload exceeds the interactive limit')
            elif family == 'teb':
                work = 16*np.ceil((settings.prediction_time+settings.braking_timeout)/
                                   min(dynamics.substep, settings.integration_dt))
                if (settings.max_samples > 64 or settings.max_iterations > 100 or work > 2_000_000 or
                        (settings.time_budget is not None and settings.time_budget > 1.)):
                    raise ValueError(f'{name} TEB workload exceeds the interactive limit (64 nodes, 100 iterations, 1 s budget)')
        eb_work = (eb.linear_samples*eb.angular_samples+64)*np.ceil(
            (eb.prediction_time+eb.braking_timeout)/min(dynamics.substep, eb.integration_dt))
        if (eb.max_nodes > 128 or eb.max_iterations > 100 or eb.max_evaluations > 10000 or
                eb.max_refinement_cycles > 32 or eb.max_validation_intervals > 32768 or
                eb.speed_profile_samples > 33 or eb.speed_response_time_s > 3 or
                eb.linear_samples*eb.angular_samples > 5000 or eb_work > 2_000_000 or
                (eb.time_budget_s is not None and eb.time_budget_s > 1.)):
            raise ValueError('EB-DWA workload exceeds the interactive limit')
        if (getattr(planner, 'time_budget', None) is not None and planner.time_budget > 10) or getattr(planner, 'path_spacing', .1) < .01:
            raise ValueError('interactive planners require time_budget <= 10 s and path_spacing >= 0.01 m')
        if config['rrt_connect'].get('max_extensions', 1800) > 100000:
            raise ValueError('max_extensions exceeds 100000')
        if config['voronoi'].get('boundary_step', .5) < .01:
            raise ValueError('boundary_step must be >= 0.01 m')
        for label, pose in [('start', start), ('goal', goal)]:
            if pose is not None and truth.collides(pose, footprint, padding=footprint.margin):
                raise ValueError(f'map.{label} intersects solid geometry, including footprint margin')
        if self.experiment is not None:
            route = self.experiment.reference_path
            if route is not None and (goal is None or not np.all(b.contains(route)) or
                    not np.allclose(route[0], start.translation[:2], rtol=0, atol=1e-6) or
                    not np.allclose(route[-1], goal.translation[:2], rtol=0, atol=1e-6)):
                raise ValueError('experiment reference must join start to goal inside bounds')
            state = self.experiment.initial_state
            if isinstance(model, Ackermann) and not np.isclose(state.angular, model.yaw_rate(state.linear, state.steering)):
                raise ValueError('initial Ackermann yaw rate must agree with speed and steering')
            event_boxes = [Box(*box) for event in self.experiment.events for box in event.boxes]
            if any(not np.all(b.contains(box.corners)) for box in event_boxes):
                raise ValueError('event box lies outside bounds')
            if sum(2*((box.xmax-box.xmin)+(box.ymax-box.ymin))/spacing+4 for box in event_boxes) + len(world.point_cloud(spacing).points) > 100000:
                raise ValueError('experiment map exceeds 100000 surface samples')
        # Freeze effective defaults into the saved case, including inactive backends.
        for key in GLOBAL_PLANNERS:
            alternate = deepcopy(config)
            alternate['backend']['planner'] = key
            config[key] = asdict(build_parameters(alternate)[4])
        for key, spec in KINEMATIC_MODELS.items():
            config[key] = asdict(parameters(spec.config, config[key], key))
        config[config['backend']['model']] = asdict(model)
        config[config['backend']['planner']] = asdict(planner)
        for key, value in [('footprint', footprint), ('dynamics', dynamics), ('safety', safety), *controllers.items()]:
            config[key] = asdict(value)
        config['navigation'] = asdict(parameters(NavigationConfig, {
            **{key: getattr(controller, key) for key in ('lookahead', 'goal_tolerance', 'yaw_tolerance') if hasattr(controller, key)},
            **config['navigation']}, 'navigation'))
        config['map'] = {'preset': 'custom', 'bounds': [b.xmin, b.xmax, b.ymin, b.ymax],
                         'start': planar_state(start).tolist(),
                         'boxes': [[box.xmin, box.xmax, box.ymin, box.ymax] for box in world.boxes]}
        if goal is not None:
            config['map']['goal'] = planar_state(goal).tolist()
        config['app']['start_mode'] = 'paused'
        self.config = config
        return self


class SceneSession(Session):
    def __init__(self, scene):
        # Session.reset calls __init__(config), so retain the scene's polygons.
        if isinstance(scene, dict):
            data = self.scene.model_dump()
            data['config'] = scene
            scene = Scene.model_validate(data)
        self.scene = scene
        super().__init__(deepcopy(scene.config))
        self.event_index, self.event_log = 0, []
        self.added_boxes = []
        self.cloud_frozen = False
        self.scan_points = []
        self.known_cloud = self.known_world.point_cloud(self.config['app']['point_spacing'])
        self.refresh_cloud()
        if scene.experiment is not None:
            experiment = scene.experiment
            if experiment.reference_path is not None:
                self.nav.planner = InitialRoutePlanner(self.nav.planner, experiment.reference_path, self.nav.goal)
            initial = experiment.initial_state.model_dump()
            linear, angular = initial.pop('linear'), initial.pop('angular')
            self.sim.state = replace(self.sim.state, velocity=Twist2D(linear, angular), **initial)

    def make_world(self, config):
        world, start, goal = make_world(config)
        self.known_world = PolygonWorld(world, self.scene.polygons, self.scene.polylines)
        unknown = self.scene.unknown_obstacles
        truth = PolygonWorld(World2D(world.bounds, (*world.boxes, *(Box(*b) for b in unknown.boxes))),
                             [*self.scene.polygons, *unknown.polygons], [*self.scene.polylines, *unknown.polylines])
        return truth, start, goal

    def refresh_cloud(self):
        # Known terrain plus this scan only: no hidden geometry or implicit SLAM.
        known = self.known_cloud
        unknown = self.scene.unknown_obstacles
        if unknown.boxes or unknown.polygons or unknown.polylines:
            s = self.sim.state
            scan = self.world.lidar(s.pose, timestamp=s.timestamp, **self.scene.lidar.model_dump()).transformed(s.pose)
            self.scan_points = scan.points[:, :2].tolist()
            self.cloud = PointCloud(np.concatenate((known.points, scan.points)), self.world.frame, s.timestamp)
        else:
            self.scan_points = []
            self.cloud = known

    def apply(self, action):
        super().apply(action)
        if action.kind == 'goal' and self.nav.goal is not None:
            experiment = self.scene.experiment
            if experiment is not None and experiment.reference_path is not None and not np.allclose(
                    experiment.reference_path[-1], self.nav.goal.translation[:2], rtol=0, atol=1e-6):
                # A live goal edit starts an ordinary planning task. Reset must
                # not resurrect an authored route to the former goal.
                self.scene = self.scene.model_copy(update={'experiment': experiment.model_copy(update={'reference_path': None})})
            self.config['map']['goal'] = planar_state(self.nav.goal).tolist()
        elif action.kind == 'policy':
            self.config['teleop']['policy'] = str(self.policy)

    def observation(self):
        experiment = self.scene.experiment
        events = () if experiment is None else experiment.events
        now = self.sim.state.timestamp
        while self.event_index < len(events) and events[self.event_index].time_s <= now+1e-9:
            event = events[self.event_index]
            if event.kind == 'add_boxes':
                base = World2D(self.world.bounds, (*self.known_world.boxes, *(Box(*box) for box in event.boxes)), frame=self.world.frame)
                self.known_world = PolygonWorld(base, self.scene.polygons, self.scene.polylines)
                self.known_cloud = self.known_world.point_cloud(self.config['app']['point_spacing'])
                unknown = self.scene.unknown_obstacles
                self.world = PolygonWorld(World2D(base.bounds, (*base.boxes, *(Box(*b) for b in unknown.boxes))),
                                          [*self.scene.polygons, *unknown.polygons], [*self.scene.polylines, *unknown.polylines])
                self.sim.world = self.world
                self.added_boxes.extend(event.boxes)
                if not self.cloud_frozen:
                    self.refresh_cloud()
            elif event.kind == 'freeze_cloud':
                stamp = self.cloud.timestamp if self.cloud.timestamp is not None else now
                self.cloud = PointCloud(self.cloud.points, self.cloud.frame, stamp)
                self.cloud_frozen = True
            else:
                self.cloud_frozen = False
                self.refresh_cloud()
            self.event_log.append({'kind': event.kind, 'scheduled_s': event.time_s, 'applied_s': now})
            self.event_index += 1
            self.notice = f'Experiment: {event.kind} at {now:.2f} s'
        if not self.cloud_frozen:
            self.refresh_cloud()
        return super().observation()

    def experiment_snapshot(self):
        return {'events': list(self.event_log), 'added_boxes': list(self.added_boxes), 'cloud_frozen': self.cloud_frozen, 'scan_points': list(self.scan_points)}


# Compatibility import; storage is separate from simulation schemas.
from .storage import SceneStore
