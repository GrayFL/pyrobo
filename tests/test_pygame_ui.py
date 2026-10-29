"""SDL dummy-driver tests exercise the actual event handlers and renderer."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import numpy as np
import pytest
pg = pytest.importorskip('pygame')

from pyrobo.apps.pygame_sim.app import App
from pyrobo.apps.pygame_sim.config import load_config
from pyrobo.apps.pygame_sim.session import Action
from pyrobo import ControlMode, MotionPolicy


@pytest.fixture
def app():
    config = load_config(selectors={'map.preset': 'empty'})
    instance = App(config, headless=True)
    yield instance
    instance.close()


def key(key, down=True):
    return pg.event.Event(pg.KEYDOWN if down else pg.KEYUP, key=key)


def click(pos):
    return [pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=pos),
            pg.event.Event(pg.MOUSEBUTTONUP, button=1, pos=pos)]


def test_wasd_click_navigation_and_goal_drag_go_through_real_pipeline(app, tmp_path):
    start = app.current.state.pose.translation.copy()
    pg.event.post(key(pg.K_w))
    app.frame()
    for _ in range(12):
        snapshot = app.frame(events=[])
    assert snapshot.mode == ControlMode.MANUAL and snapshot.state.pose.translation[0] > start[0]+.1
    app.frame([key(pg.K_w, False)])
    pos = app.view.viewport.pixel([1., 0.])
    snapshot = app.frame(click(pos))
    assert snapshot.mode == ControlMode.AUTO and len(snapshot.path) > 1
    assert app.session.runtime.last_request.policy == MotionPolicy.NAVIGATION
    np.testing.assert_allclose(snapshot.goal.translation[:2], [1., 0.], atol=.02)
    end = (pos[0], pos[1]-45)
    snapshot = app.frame([pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=pos),
                          pg.event.Event(pg.MOUSEMOTION, pos=end),
                          pg.event.Event(pg.MOUSEBUTTONUP, button=1, pos=end)])
    assert snapshot.goal.rot_mat[1, 0] == pytest.approx(1.)
    destination = tmp_path/'ui.png'
    pg.image.save(app.screen, str(destination))
    assert destination.stat().st_size > 10000


def test_estop_blocks_held_keys_and_auto_until_release_and_neutral(app):
    app.frame([key(pg.K_w)])
    snapshot = app.frame([key(pg.K_e), key(pg.K_g)])
    assert snapshot.mode == ControlMode.ESTOP
    snapshot = app.frame([key(pg.K_r)])
    assert snapshot.mode == ControlMode.PAUSED
    snapshot = app.frame([])
    assert snapshot.mode == ControlMode.PAUSED  # W is still physically held
    app.frame([key(pg.K_w, False)])
    snapshot = app.frame([key(pg.K_w)])
    assert snapshot.mode == ControlMode.MANUAL


def test_virtual_stick_focus_loss_and_reset(app):
    c, r = app.view.stick_center, app.view.stick_radius
    pos = (c[0], c[1]-r*.8)
    event = pg.event.Event(pg.MOUSEBUTTONDOWN, button=1, pos=pos)
    snapshot = app.frame([event])
    assert snapshot.mode == ControlMode.MANUAL
    for _ in range(5):
        snapshot = app.frame([])
    assert snapshot.state.velocity.linear > .05
    snapshot = app.frame([pg.event.Event(pg.WINDOWFOCUSLOST)])
    assert snapshot.mode == ControlMode.PAUSED and not app.inputs.virtual_active
    snapshot = app.frame([pg.event.Event(pg.WINDOWFOCUSGAINED), key(pg.K_HOME)])
    assert snapshot.mode == ControlMode.PAUSED
    assert snapshot.state.timestamp == pytest.approx(.1)
    assert snapshot.state.velocity.linear == 0.


@pytest.mark.parametrize('model', ['differential', 'ackermann'])
def test_a_key_is_yaw_intent_for_diff_but_steering_intent_for_car(model):
    app = App(load_config(selectors={'map.preset': 'empty', 'backend.model': model}), headless=True)
    try:
        app.frame([key(pg.K_a)])
        for _ in range(8):
            snapshot = app.frame([])
        assert snapshot.state.velocity.linear == 0.
        if model == 'differential':
            assert snapshot.state.velocity.angular > .1
        else:
            assert snapshot.state.steering > .05 and snapshot.state.velocity.angular == 0.
    finally:
        app.close()


def test_gamepad_deadman_and_instance_id_removal_pause(app):
    class Gamepad:
        held = False
        closed = False
        def get_name(self): return 'Test controller'
        def get_numaxes(self): return 2
        def get_numbuttons(self): return 1
        def get_button(self, button): return self.held
        def get_axis(self, axis): return -1. if axis == 1 else 0.
        def quit(self): self.closed = True
    device = Gamepad()
    app.inputs.devices, app.inputs.selected_id = {71: device}, 71
    assert app.frame([]).mode == ControlMode.PAUSED
    device.held = True
    for _ in range(6):
        snapshot = app.frame([])
    assert snapshot.mode == ControlMode.MANUAL and snapshot.state.velocity.linear > .05
    snapshot = app.frame([pg.event.Event(pg.JOYDEVICEREMOVED, instance_id=71)])
    assert device.closed and snapshot.mode == ControlMode.PAUSED
    assert snapshot.motion.command.mode == 'estop'


def test_letterbox_click_does_not_set_goal_and_resize_preserves_coordinates(app):
    assert app.view.viewport.world((10, 10)) is None
    app.frame([pg.event.Event(pg.VIDEORESIZE, w=1000, h=720)])
    for point in ([-4, -3], [0, 0], [3, 2]):
        mapped = app.view.viewport.world(app.view.viewport.pixel(point))
        np.testing.assert_allclose(mapped, point, atol=1/app.view.viewport.scale)
    before = app.session.nav.goal
    app.frame(click((10, 10)))
    assert app.session.nav.goal is before
    assert app.view.stick_center[1]-app.view.stick_radius > 540
