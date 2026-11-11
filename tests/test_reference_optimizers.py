"""Editable classic objectives, real chassis execution and Web registration."""
from dataclasses import asdict
import importlib
import json

import numpy as np
import pytest

from pyrobo import (Ackermann, Bounds2D, ChassisState, DifferentialDrive, Footprint, MotionPolicy, MotionRequest,
                    Navigator, Path, PointCloud, RobotObservation, SafetyAssessment, Twist2D, pose2d)
from pyrobo.control.registry import LOCAL_OPTIMIZERS
from pyrobo.core.geometry import path_poses

NAMES = ('ref_dwa', 'ref_teb', 'towing_dwa', 'towing_teb')


def observation(points=(), stamp=0.):
    return RobotObservation(ChassisState(pose2d(0, 0), timestamp=stamp),
        PointCloud(np.asarray(points).reshape(-1, 2)), Bounds2D(-5, 5, -4, 4), .1)


def request(stamp=0.):
    return MotionRequest(MotionPolicy.NAVIGATION, stamp,
        path=Path(path_poses([[0, 0], [2, .2], [3, .2]])), goal=pose2d(3, .2))


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('model', [DifferentialDrive(), Ackermann()])
@pytest.mark.parametrize('footprint', [Footprint(), Footprint(length=.79, width=.54, offset_x=-.295)])
def test_native_prediction_and_no_legacy_navigation_policy(name, model, footprint, monkeypatch):
    from pyrobo.control.dwa.policies import NavigationPolicy
    from pyrobo.control.dwa import rollout
    def forbidden(*args, **kwargs):
        raise AssertionError('reference autonomy invoked the old heuristic policy')
    monkeypatch.setattr(NavigationPolicy, 'prepare', forbidden)
    monkeypatch.setattr(rollout, 'evaluate_safety', forbidden)
    spec = LOCAL_OPTIMIZERS[name]
    nav = Navigator(controller=spec.config(), model=model, footprint=footprint)
    engine = nav.optimizer
    assert type(engine) is spec.factory
    assert engine.footprint == nav.footprint == footprint
    assert engine.model == nav.model == model
    o = observation()
    result = engine.optimize(o, request())
    assert result.safe and result.diagnostics.extra['optimizer'] == name
    actual = engine.plant.step(o.chassis, result.command, o.dt).state
    first = np.flatnonzero(np.isclose(result.trajectory.times, o.dt))[0]
    np.testing.assert_allclose(result.trajectory.states[first], engine.plant.codec.pack(actual), atol=1e-12)
    assert engine.plant.codec.stopped(result.trajectory.states[-1:])[0]
    if spec.family == 'teb':
        assert result.diagnostics.candidate_count == 1
        assert engine.last_band.cost < engine.last_band.initial_cost
        assert result.trajectory.times[result.trajectory.prediction_end] == pytest.approx(o.dt)
    else:
        assert set(result.diagnostics.cost_terms) == {'heading', 'clearance', 'velocity'}


@pytest.mark.parametrize('name', NAMES)
def test_stale_observation_stop_manual_and_reset(name):
    engine = LOCAL_OPTIMIZERS[name].factory()
    stale = engine.optimize(observation(stamp=1.), request())
    assert not stale.safe and stale.safety_assessment == SafetyAssessment.INPUT_STALE
    blocked = engine.optimize(observation([[0, 0]]), request())
    assert not blocked.safe and blocked.command.mode == 'estop'
    o = observation()
    manual = engine.optimize(o, MotionRequest(MotionPolicy.TELEOP_STOP, 0., desired=Twist2D(-.3, .1)))
    assert manual.safe and manual.diagnostics.extra['optimizer'] == 'dwa'
    assert manual.diagnostics.extra['navigation_optimizer'] == name
    command = engine.command_space.from_native_batch(engine.model.encode_command(manual.command)[None])[0]
    assert -.3 <= command[0] <= 0 and command[1] == pytest.approx(.1)
    engine.reset()
    assert engine.optimize(o, request()).safe


@pytest.mark.parametrize('family', ['dwa', 'teb'])
@pytest.mark.parametrize('model', [DifferentialDrive(), Ackermann()])
def test_towing_clone_has_same_math_but_independent_types(family, model):
    ref, towing = (LOCAL_OPTIMIZERS[prefix+family] for prefix in ('ref_', 'towing_'))
    assert ref.config is not towing.config and ref.factory is not towing.factory
    assert asdict(ref.config()) == asdict(towing.config())
    # At equal geometry, cloned optimizers produce the same numerical result.
    a = ref.factory(model=model).optimize(observation(), request())
    b = towing.factory(model=model).optimize(observation(), request())
    assert a.command == b.command
    np.testing.assert_array_equal(a.trajectory.states, b.trajectory.states)
    for spec in (ref, towing):
        assert spec.factory().footprint == Navigator(controller=spec.config()).footprint == Footprint()


@pytest.mark.parametrize('name', NAMES)
def test_explicit_rear_geometry_checks_obstacles_independent_of_algorithm(name):
    spec = LOCAL_OPTIMIZERS[name]
    footprint = Footprint(length=.79, width=.54, offset_x=-.295)
    engine = spec.factory(footprint=footprint)
    assert Navigator(local_optimizer=engine).footprint == footprint
    corners = engine.footprint.corners(pose2d(0, 0))
    assert corners[:, 0].max() == pytest.approx(.1)
    assert corners[:, 0].min() == pytest.approx(-.69)
    assert np.ptp(corners[:, 1]) == pytest.approx(.54)
    stop = MotionRequest(MotionPolicy.STOP, 0.)
    # Obstacle in the rear extension must be checked, despite the axle being far ahead.
    assert not engine.optimize(observation([[-.6, 0]]), stop).safe
    assert spec.factory().optimize(observation([[-.6, 0]]), stop).safe


@pytest.mark.parametrize('name', ['ref_dwa', 'towing_dwa'])
def test_classic_window_and_three_term_objective(name):
    module = importlib.import_module(f'pyrobo.control.{name}.optimizer')
    search = module.velocity_window(np.array([.4, .2]), np.array([0., -1.5]),
        np.array([.9, 1.5]), np.array([.9, 2.5]), .1, (5, 5))
    np.testing.assert_allclose(search.min(axis=0), [.31, -.05])
    np.testing.assert_allclose(search.max(axis=0), [.49, .45])
    assert np.any(search[:, 1] == 0.)
    cfg = LOCAL_OPTIMIZERS[name].config()
    terminal = np.array([[0., 0, 0], [0, 0, np.pi]])
    commands = np.array([[.9, 0], [0, 0]])
    costs, _ = module.objective(terminal, commands, np.array([1., 0.]), np.array([3., 0]), .9, cfg)
    np.testing.assert_allclose(costs, [0., cfg.heading_weight+cfg.clearance_weight+cfg.speed_weight])


@pytest.mark.parametrize('name', ['ref_teb', 'towing_teb'])
def test_teb_time_objective_has_no_route_tracking_term(name):
    spec = LOCAL_OPTIMIZERS[name]
    cfg = spec.config(obstacle_weight=0., kinematics_weight=0., velocity_weight=0.,
        acceleration_weight=0., turning_weight=0., max_iterations=1)
    assert not hasattr(cfg, 'path_weight')
    engine = spec.factory(cfg)
    engine.optimize(observation(), request())
    band = engine.last_band
    assert band.cost == pytest.approx(cfg.time_weight*np.sum(band.intervals**2))


@pytest.mark.parametrize('name', ['ref_teb', 'towing_teb'])
def test_teb_disabled_reverse_applies_to_command_despite_soft_solver_residual(name):
    spec = LOCAL_OPTIMIZERS[name]
    engine = spec.factory(spec.config(allow_reverse=False))
    goal = MotionRequest(MotionPolicy.NAVIGATION, 0.,
        path=Path(path_poses([[0, 0], [-3, 0]])), goal=pose2d(-3, 0))
    result = engine.optimize(observation(), goal)
    assert result.safe
    command = engine.command_space.from_native_batch(engine.model.encode_command(result.command)[None])[0]
    assert command[0] >= 0.


@pytest.mark.parametrize('name', ['ref_teb', 'towing_teb'])
def test_teb_turning_band_does_not_require_holding_first_command_for_entire_horizon(name, monkeypatch):
    from pyrobo.geometry.sweep import swept_clearances
    from pyrobo.models.dynamics.prediction import BatchPredictor
    from pyrobo.safety import SafetyGuard
    controller = importlib.import_module(f'pyrobo.control.{name}.controller')
    math = importlib.import_module(f'pyrobo.control.{name}.optimizer')
    # A straight first segment, then a turn before the obstacle at x=.9.
    band = math.Band(np.array([[0., 0., 0.], [.15, 0., 0.], [.3, .2, np.pi/2], [.3, 1., np.pi/2]]),
                     np.array([.3, .5, .8]), 1., 2., 1, 1, 'converged', False)
    monkeypatch.setattr(controller, 'solve_band', lambda *args, **kwargs: band)
    engine = LOCAL_OPTIMIZERS[name].factory(footprint=Footprint(length=.2, width=.2, margin=.02))
    o = observation([[.9, 0.]])
    req = MotionRequest(MotionPolicy.NAVIGATION, 0., path=Path(path_poses(band.poses[:, :2])),
                        goal=pose2d(.3, 1., np.pi/2))
    result = engine.optimize(o, req)
    assert result.safe and result.command.mode == 'track'
    assert result.trajectory.times[result.trajectory.prediction_end] == pytest.approx(o.dt)
    assert SafetyGuard(engine.plant, engine.command_space, engine.footprint).enforce(o, result, req).safe
    # Holding that same first command over 2.4s tests a different, colliding path.
    predictor = BatchPredictor(engine.plant, engine.footprint, prediction_time=engine.config.prediction_time)
    batch = predictor.predict(engine.plant.codec.pack(o.chassis), engine.model.encode_command(result.command)[None], o.dt)
    assert swept_clearances(engine.plant.codec.poses(batch.states), batch.sweep, engine.index.get(o.obstacles),
        engine.footprint, o.bounds, batch.valid_time_mask)[0] <= 0.


@pytest.mark.parametrize('name', ['ref_teb', 'towing_teb'])
def test_teb_band_horizon_does_not_impose_a_dwa_control_period_limit(name):
    spec = LOCAL_OPTIMIZERS[name]
    engine = spec.factory(spec.config(prediction_time=.05))
    result = engine.optimize(observation(), request())
    assert result.safe
    assert result.trajectory.times[result.trajectory.prediction_end] == pytest.approx(.1)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('planner', ['rrt_connect', 'voronoi'])
@pytest.mark.parametrize('model', ['differential', 'ackermann'])
def test_web_scene_persistence_and_closed_loop_arrival(name, planner, model, tmp_path):
    from pyrobo.apps.web_sim.scene import Scene, SceneSession, SceneStore
    from pyrobo.apps.sim_common.session import Action
    from pyrobo.apps.web_sim.server import snapshot_data
    footprint = Footprint(length=.79, width=.54, offset_x=-.295)
    case = Scene(config={'backend': {'local_optimizer': name, 'planner': planner, 'model': model},
                         'footprint': asdict(footprint),
                         'map': {'preset': 'empty'}})
    store = SceneStore(tmp_path)
    restored = store.load(store.save(case))
    assert restored.model_dump() == case.model_dump()
    assert case.config['footprint'] == asdict(footprint)
    assert case.config['backend']['model'] == model
    session = SceneSession(restored)
    session.apply(Action('mode', 'auto'))
    path = None
    for _ in range(200):
        snapshot = session.tick()
        assert snapshot.motion.safe and not snapshot.collided and not snapshot.fault
        if path is None and len(snapshot.path):
            path = snapshot.path.copy()
        if snapshot.navigation_status == 'arrived':
            break
    assert snapshot.navigation_status == 'arrived'
    assert max(abs(snapshot.state.velocity.linear), abs(snapshot.state.velocity.angular)) < 1e-3
    np.testing.assert_array_equal(snapshot.path, path)
    data = snapshot_data(snapshot, 1)
    assert data['optimizer'] == name and data['forecast']
    json.dumps(data, allow_nan=False)


@pytest.mark.parametrize('name', NAMES)
def test_web_validates_variant_workload_even_when_inactive(name):
    from pyrobo.apps.web_sim.scene import Scene
    values = {'max_samples': 1000} if name.endswith('teb') else {'linear_samples': 10000}
    with pytest.raises(ValueError, match='interactive limit'):
        Scene(config={'map': {'preset': 'empty'}, name: values})


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('model', ['differential', 'ackermann'])
def test_web_uses_shared_body_defaults_and_preserves_explicit_footprint(name, model):
    from pyrobo.apps.web_sim.scene import Scene
    case = Scene(config={'backend': {'local_optimizer': name, 'model': model}, 'map': {'preset': 'empty'},
                         'footprint': {'width': .6, 'margin': .1}})
    assert case.config['footprint'] == {'width': .6, 'margin': .1, 'length': .5, 'offset_x': 0.}
    assert case.config['backend']['model'] == model
