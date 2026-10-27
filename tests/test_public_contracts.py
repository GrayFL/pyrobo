import importlib
from typing import get_type_hints

import numpy as np
import pytest
from graphmap.pose import PoseArray

from pyrobo import (Ackermann, ChassisDynamics, ChassisState, DWAConfig, DynamicWindow, MotionPolicy,
                    MotionRequest, Navigator, Path, PlanResult, PlanStatus, RobotObservation,
                    Trajectory, Twist2D, pose2d)
from pyrobo.core.geometry import path_poses
from pyrobo.core.obstacles import Bounds2D, PointCloud
from pyrobo.geometry.cache import SnapshotIndex


@pytest.mark.parametrize('old,new,names', [
    ('pyrobo.types', 'pyrobo.core', ('Footprint', 'Bounds2D', 'Twist2D', 'pose2d')),
    ('pyrobo.dynamics', 'pyrobo.models.dynamics', ('ChassisDynamics', 'DynamicsConfig', 'DynamicsResult', 'Drag')),
    ('pyrobo.kinematics', 'pyrobo.models.kinematics', ('DifferentialDrive', 'Ackermann')),
    ('pyrobo.kinematics', 'pyrobo.core', ('WheelCommand', 'AckermannCommand')),
    ('pyrobo.pointcloud', 'pyrobo.core', ('PointCloud',)),
    ('pyrobo.control.dwa.path', 'pyrobo.navigation.tracking', ('PathTracker', 'PathProgress', 'path_lookahead')),
])
def test_compatibility_exports_have_one_defining_implementation(old, new, names):
    old, new = importlib.import_module(old), importlib.import_module(new)
    assert all(getattr(old, name) is getattr(new, name) for name in names)


def test_typed_path_trajectory_frames_and_time_contracts():
    poses = path_poses([[0, 0], [1, 0]], 'map')
    route = Path(poses)
    result = PlanResult(route, 'success', 0, .1)
    assert result.status is PlanStatus.SUCCESS and isinstance(result.path, PoseArray)
    assert result.route.frame == 'map'
    trajectory = Trajectory(poses, [0., .1], start_time=4.)
    np.testing.assert_allclose(trajectory.timestamps, [4, 4.1])
    assert not trajectory.times.flags.writeable
    with pytest.raises(ValueError, match='increase'):
        Trajectory(poses, [0., 0.])
    with pytest.raises(ValueError, match='planar'):
        Path(PoseArray(position=[[0, 0, 1]], rotation=[[0, 0, 0]], degrees=False, wrd_frame='map'))
    with pytest.raises(ValueError, match='frames'):
        MotionRequest(MotionPolicy.NAVIGATION, 0., path=route, goal=pose2d(1, 0))
    with pytest.raises(ValueError, match='desired'):
        MotionRequest(MotionPolicy.TELEOP_STOP, 0.)
    assert get_type_hints(RobotObservation)['chassis'] is ChassisState


def test_snapshot_index_reuses_only_identical_immutable_snapshots():
    index = SnapshotIndex()
    first = PointCloud([[1, 0]], timestamp=0.)
    world = index.get(first)
    assert index.get(first) is world
    assert not first.points.flags.writeable
    # Same timestamp is not permission to reuse a different cloud's geometry.
    assert index.get(PointCloud([[2, 0]], timestamp=0.)) is not world


def test_injected_plant_infers_model_and_legacy_speed_defaults_are_preserved():
    plant = ChassisDynamics(Ackermann())
    engine = DynamicWindow(dynamics=plant)
    assert engine.model is plant.model and engine.plant is plant
    assert not engine.command_space.can_spin
    nav = Navigator(controller=DWAConfig(max_speed=.4, max_yaw_rate=.6))
    assert nav.model.max_speed == .4 and nav.model.max_yaw_rate == .6


def test_legacy_dwa_without_explicit_state_uses_timed_cloud_as_observation_clock():
    engine = DynamicWindow()
    obstacles = SnapshotIndex().get(PointCloud(np.empty((0, 2)), timestamp=42.))
    result = engine.step(pose2d(0, 0), Twist2D(), pose2d(3, 0), path_poses([[0, 0], [3, 0]]),
                         obstacles, Bounds2D(-4, 4, -4, 4), .1)
    assert result.safe and result.command.mode == 'track'
