"""Versioned, portable scenarios and atomic server-side case storage."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyrobo.apps.sim_common.config import (DEFAULTS, build_parameters, controller_parameters, make_world, merge,
                                        parameters, validate)
from pyrobo.apps.sim_common.session import Session
from pyrobo import Ackermann, DifferentialDrive, NavigationConfig
from pyrobo.core.geometry import planar_state
from pyrobo.core import PointCloud, Twist2D
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


class Scene(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    schema_version: Literal[1] = 1
    name: str = Field(default='未命名场景', min_length=1, max_length=80)
    description: str = Field(default='', max_length=2000)
    config: dict = Field(default_factory=dict)
    polygons: list[list[tuple[float, float]]] = Field(default_factory=list, max_length=256)
    polylines: list[Polyline] = Field(default_factory=list, max_length=256)
    experiment: Experiment | None = None

    @model_validator(mode='after')
    def normalize(self):
        finite_json(self.config)
        config = merge(DEFAULTS, self.config)
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
        if max(b.xmax-b.xmin, b.ymax-b.ymin) > 200:
            raise ValueError('map dimensions must be <= 200 m')
        for polygon in world.polygons:
            if not np.all(b.contains(polygon)):
                raise ValueError('polygon lies outside map bounds')
        for points, radius in world.polylines:
            if not np.all(b.contains(points-radius)) or not np.all(b.contains(points+radius)):
                raise ValueError('polyline wall extends outside map bounds')
        segments = world.segments()
        if np.ceil(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1)/spacing).sum() > 100_000:
            raise ValueError('map exceeds 100000 surface samples; increase point_spacing')
        model, footprint, dynamics, controller, planner, safety = build_parameters(config)
        dwa, teb, eb = (controller_parameters(config, key) for key in ('dwa', 'teb', 'eb_dwa'))
        if config['app']['dt'] < .02 or config['app']['trail_length'] > 10000:
            raise ValueError('web simulation needs dt >= 0.02 s and trail_length <= 10000')
        work = (dwa.linear_samples*dwa.angular_samples+64)*np.ceil(
            (dwa.prediction_time+dwa.braking_timeout)/min(dynamics.substep, dwa.integration_dt))
        if (dwa.linear_samples*dwa.angular_samples > 5000 or work > 2_000_000 or
                safety.braking_timeout/min(dynamics.substep, config['app']['dt']) > 10000):
            raise ValueError('prediction workload exceeds the interactive limit; reduce samples/horizon or increase integration steps')
        teb_work = 16*np.ceil((teb.prediction_time+teb.braking_timeout)/min(dynamics.substep, teb.integration_dt))
        if (teb.max_samples > 64 or teb.max_iterations > 100 or teb_work > 2_000_000 or
                (teb.time_budget is not None and teb.time_budget > 1.)):
            raise ValueError('TEB workload exceeds the interactive limit (64 nodes, 100 iterations, 1 s budget)')
        eb_work = (eb.linear_samples*eb.angular_samples+64)*np.ceil(
            (eb.prediction_time+eb.braking_timeout)/min(dynamics.substep, eb.integration_dt))
        if (eb.max_nodes > 128 or eb.max_iterations > 100 or eb.max_evaluations > 10000 or
                eb.max_refinement_cycles > 32 or eb.max_validation_intervals > 32768 or
                eb.linear_samples*eb.angular_samples > 5000 or eb_work > 2_000_000 or
                (eb.time_budget_s is not None and eb.time_budget_s > 1.)):
            raise ValueError('EB-DWA workload exceeds the interactive limit')
        if planner.time_budget > 10 or planner.path_spacing < .01:
            raise ValueError('interactive planners require time_budget <= 10 s and path_spacing >= 0.01 m')
        if config['rrt_connect'].get('max_extensions', 1800) > 100000:
            raise ValueError('max_extensions exceeds 100000')
        if config['voronoi'].get('boundary_step', .5) < .01:
            raise ValueError('boundary_step must be >= 0.01 m')
        for label, pose in [('start', start), ('goal', goal)]:
            if pose is not None and world.collides(pose, footprint, padding=footprint.margin):
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
        for key in ('rrt_connect', 'voronoi'):
            alternate = deepcopy(config)
            alternate['backend']['planner'] = key
            config[key] = asdict(build_parameters(alternate)[4])
        for cls, key in [(DifferentialDrive, 'differential'), (Ackermann, 'ackermann')]:
            config[key] = asdict(parameters(cls, config[key], key))
        config[config['backend']['model']] = asdict(model)
        config[config['backend']['planner']] = asdict(planner)
        for key, value in [('footprint', footprint), ('dynamics', dynamics), ('dwa', dwa), ('teb', teb), ('eb_dwa', eb), ('safety', safety)]:
            config[key] = asdict(value)
        config['navigation'] = asdict(parameters(NavigationConfig, {
            'lookahead': controller.lookahead, 'goal_tolerance': controller.goal_tolerance, 'yaw_tolerance': controller.yaw_tolerance,
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
        if scene.experiment is not None:
            experiment = scene.experiment
            if experiment.reference_path is not None:
                self.nav.planner = InitialRoutePlanner(self.nav.planner, experiment.reference_path, self.nav.goal)
            initial = experiment.initial_state.model_dump()
            linear, angular = initial.pop('linear'), initial.pop('angular')
            self.sim.state = replace(self.sim.state, velocity=Twist2D(linear, angular), **initial)

    def make_world(self, config):
        world, start, goal = make_world(config)
        return PolygonWorld(world, self.scene.polygons, self.scene.polylines), start, goal

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
                base = World2D(self.world.bounds, (*self.world.boxes, *(Box(*box) for box in event.boxes)), frame=self.world.frame)
                self.world = PolygonWorld(base, self.scene.polygons, self.scene.polylines)
                self.sim.world = self.world
                self.added_boxes.extend(event.boxes)
                if not self.cloud_frozen:
                    self.cloud = self.world.point_cloud(self.config['app']['point_spacing'])
            elif event.kind == 'freeze_cloud':
                self.cloud = PointCloud(self.cloud.points, self.cloud.frame, now)
                self.cloud_frozen = True
            else:
                self.cloud_frozen = False
                self.cloud = self.world.point_cloud(self.config['app']['point_spacing'])
            self.event_log.append({'kind': event.kind, 'scheduled_s': event.time_s, 'applied_s': now})
            self.event_index += 1
            self.notice = f'Experiment: {event.kind} at {now:.2f} s'
        return super().observation()

    def experiment_snapshot(self):
        return {'events': list(self.event_log), 'added_boxes': list(self.added_boxes), 'cloud_frozen': self.cloud_frozen}


class SceneStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, identifier):
        if not re.fullmatch(r'[a-f0-9]{32}', identifier):
            raise ValueError('invalid scene id')
        return self.directory/f'{identifier}.json'

    def save(self, scene):
        identifier = uuid4().hex
        payload = scene.model_dump_json(indent=2).encode('utf-8')
        if len(payload) > MAX_SCENE_BYTES:
            raise ValueError('scene file exceeds 2 MB')
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                os.replace(temporary, self.path(identifier))
            finally:
                temporary.unlink(missing_ok=True)
        return identifier

    def load(self, identifier):
        path = self.path(identifier)
        if path.stat().st_size > MAX_SCENE_BYTES:
            raise ValueError('scene file exceeds 2 MB')
        return Scene.model_validate_json(path.read_bytes())

    def list(self):
        result = []
        for path in self.directory.glob('*.json'):
            if not re.fullmatch(r'[a-f0-9]{32}', path.stem) or path.stat().st_size > MAX_SCENE_BYTES:
                continue
            try:
                data = json.loads(path.read_text())
                result.append({'id': path.stem, 'name': data['name'], 'description': data.get('description', ''),
                               'modified': path.stat().st_mtime})
            except (OSError, ValueError, KeyError):
                continue
        return sorted(result, key=lambda item: item['modified'], reverse=True)
