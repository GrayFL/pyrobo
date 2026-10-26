import numpy as np

from pyrobo import PathTracker, pose2d
from pyrobo.types import path_poses


def test_lookahead_passes_corner_before_robot_reaches_waypoint():
    tracker = PathTracker()
    tracker.set_path(path_poses([[0, 0], [2, 0], [2, 3]]))
    progress = tracker.update(pose2d(1.5, 0), 1.)
    np.testing.assert_allclose(progress.target.translation[:2], [2, .5])
    assert progress.segment == 0 and progress.target_segment == 1


def test_reverse_restores_old_segments_without_destroying_waypoints():
    tracker = PathTracker()
    path = path_poses([[0, 0], [2, 0], [2, 3]])
    tracker.set_path(path)
    before = tracker.xy.copy()
    for x, y in ((0, 0), (1.8, 0), (2, .7), (2, 2), (2, .5), (1.5, 0)):
        progress = tracker.update(pose2d(x, y), .4)
    assert progress.segment == 0 and progress.distance < 2
    np.testing.assert_allclose(progress.target.translation[:2], [1.9, 0])
    np.testing.assert_array_equal(tracker.xy, before)


def test_crossing_does_not_jump_to_later_branch():
    tracker = PathTracker()
    tracker.set_path(path_poses([[-2, 0], [2, 0], [2, 2], [0, 2], [0, -2]]))
    for x in np.linspace(-2, .1, 30):
        progress = tracker.update(pose2d(x, .03), .5)
    assert progress.distance < 2.2 and progress.segment == 0
    for x in np.linspace(.1, -1., 20):
        progress = tracker.update(pose2d(x, 0), .5)
    assert progress.distance < 1.1


def test_duplicates_single_point_and_path_replacement():
    tracker = PathTracker()
    tracker.set_path(path_poses([[1, 0], [1, 0], [2, 0]]))
    assert tracker.update(pose2d(1, 0), .5).target.translation[0] == 1.5
    tracker.set_path(path_poses([[4, 3]]))
    result = tracker.update(pose2d(0, 0), 2.)
    np.testing.assert_array_equal(result.target.translation[:2], [4, 3])
    assert result.distance == result.remaining_distance == 0
