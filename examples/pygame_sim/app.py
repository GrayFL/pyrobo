"""SDL stays on this thread. Session/Worker own planning and physical integration."""
import json
from pathlib import Path

import numpy as np
import pygame as pg

from pyrobo import ControlMode, MotionPolicy
from pyrobo.core.geometry import planar_state
from .inputs import Inputs
from .session import Action, Session, Worker
from .view import View


class App:
    def __init__(self, config, *, headless=False):
        pg.display.init()
        self.config, self.headless = config, headless
        self.screen = pg.display.set_mode((config['app']['width'], config['app']['height']), pg.RESIZABLE)
        pg.display.set_caption('Pyrobo | Motion Lab')
        self.session = Session(config)
        self.inputs = Inputs(config['joystick'])
        self.view = View(self.screen, self.session)
        self.worker = None if headless else Worker(self.session)
        self.current = self.session.snapshot()
        self.pending = []
        self.running = True
        self.drag_start, self.drag_end = None, None
        self.ui_notice = ''
        if self.worker:
            self.worker.start()

    def emit(self, action):
        if action.kind != 'policy':
            self.inputs.suppress()
        if self.worker:
            if not self.worker.enqueue(action):
                self.ui_notice = 'Input queue busy. Try again.'
                return
        else:
            self.pending.append(action)
        self.ui_notice = ''

    def button(self, name):
        actions = {
            'Manual': Action('mode', ControlMode.MANUAL), 'Auto': Action('mode', ControlMode.AUTO),
            'Pause': Action('mode', ControlMode.PAUSED), 'E-stop': Action('estop'),
            'Release': Action('release'), 'Reset': Action('reset'),
            'Stop assist': Action('policy', MotionPolicy.TELEOP_STOP),
            'Avoid assist': Action('policy', MotionPolicy.TELEOP_AVOID),
        }
        self.emit(actions[name])

    def handle(self, event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.VIDEORESIZE:
            size = max(1000, event.w), max(720, event.h)
            if self.screen.get_size() != size:
                self.screen = pg.display.set_mode(size, pg.RESIZABLE)
            self.view.resize(self.screen)
            self.drag_start = self.drag_end = None
            self.inputs.virtual_active = False
        elif event.type in (pg.WINDOWFOCUSLOST, pg.WINDOWFOCUSGAINED):
            self.inputs.focus(event.type == pg.WINDOWFOCUSGAINED)
            if not self.inputs.focused:
                self.drag_start = self.drag_end = None
                self.emit(Action('mode', ControlMode.PAUSED))
        elif event.type == pg.JOYDEVICEADDED:
            self.inputs.add_device(event.device_index)
        elif event.type == pg.JOYDEVICEREMOVED:
            if self.inputs.remove_device(event.instance_id):
                self.emit(Action('mode', ControlMode.PAUSED))
        elif event.type in (pg.KEYDOWN, pg.KEYUP):
            self.inputs.key(event.key, event.type == pg.KEYDOWN)
            if event.type == pg.KEYDOWN and not getattr(event, 'repeat', False):
                buttons = {pg.K_g: 'Auto', pg.K_m: 'Manual', pg.K_SPACE: 'Pause', pg.K_e: 'E-stop',
                           pg.K_r: 'Release', pg.K_HOME: 'Reset', pg.K_1: 'Stop assist', pg.K_2: 'Avoid assist'}
                if event.key == pg.K_ESCAPE:
                    self.running = False
                elif event.key == pg.K_p:
                    self.view.show_points = not self.view.show_points
                elif event.key in buttons:
                    self.button(buttons[event.key])
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 3:
                self.emit(Action('mode', ControlMode.PAUSED))
                self.drag_start = self.drag_end = None
            elif event.button == 1 and self.inputs.focused:
                for name, rect in self.view.buttons.items():
                    if rect.collidepoint(event.pos):
                        self.button(name)
                        return
                if np.linalg.norm(np.asarray(event.pos)-self.view.stick_center) <= self.view.stick_radius:
                    self.inputs.virtual_active = True
                    self.inputs.move_virtual(event.pos, self.view.stick_center, self.view.stick_radius)
                elif self.view.viewport.world(event.pos) is not None:
                    self.drag_start = self.drag_end = event.pos
        elif event.type == pg.MOUSEMOTION:
            if self.inputs.virtual_active:
                self.inputs.move_virtual(event.pos, self.view.stick_center, self.view.stick_radius)
            elif self.drag_start:
                self.drag_end = event.pos
        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            self.inputs.virtual_active = False
            self.inputs.virtual = (0., 0.)
            if self.drag_start is not None:
                viewport = self.view.viewport
                xy = viewport.world(self.drag_start)
                delta = np.asarray(event.pos)-self.drag_start
                if np.linalg.norm(delta) >= 7:
                    direction = np.array([delta[0], -delta[1]])
                else:
                    direction = xy-self.current.state.pose.translation[:2]
                yaw = (float(np.arctan2(direction[1], direction[0])) if np.linalg.norm(direction) > 1e-9
                       else float(planar_state(self.current.state.pose)[2]))
                self.emit(Action('goal', (*xy, yaw)))
                self.drag_start = self.drag_end = None

    def frame(self, events=None):
        for event in pg.event.get() if events is None else events:
            self.handle(event)
        if not self.running:
            return self.current
        controls = self.inputs.sample()
        if self.inputs.device_lost:
            self.emit(Action('mode', ControlMode.PAUSED))
            self.inputs.device_lost = False
            controls = self.inputs.sample()
        if self.worker:
            self.worker.publish(controls)
            self.current = self.worker.snapshot()
        else:
            self.current = self.session.tick(controls, tuple(self.pending))
            self.pending.clear()
        drag = (self.drag_start, self.drag_end) if self.drag_start is not None else None
        self.view.draw(self.current, self.inputs, drag=drag, ui_notice=self.ui_notice)
        return self.current

    def run(self, *, steps=None, screenshot=None):
        clock = pg.time.Clock()
        while self.running:
            snapshot = self.frame()
            if steps is not None and snapshot.ticks >= steps:
                break
            if (self.headless or steps is not None) and (snapshot.collided or snapshot.fault):
                break
            if not self.headless:
                clock.tick(self.config['app']['fps'])
        if screenshot:
            path = Path(screenshot)
            path.parent.mkdir(parents=True, exist_ok=True)
            pg.image.save(self.screen, str(path))
        s = self.current
        result = {'model': self.config['backend']['model'], 'planner': self.config['backend']['planner'],
                  'ticks': s.ticks, 'simulation_seconds': s.state.timestamp, 'mode': str(s.mode),
                  'navigation_status': s.navigation_status, 'planner_status': s.planner_status,
                  'collided': s.collided,
                  'position_m': s.state.pose.translation[:2].round(4).tolist(), 'fault': s.fault}
        print(json.dumps(result, ensure_ascii=False))
        return 1 if s.collided or s.fault else 0

    def close(self):
        if self.worker:
            self.worker.close()
        self.inputs.close()
        pg.quit()
