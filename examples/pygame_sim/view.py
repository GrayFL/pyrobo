"""Pygame-only drawing and aspect-preserving world/screen conversion."""
from dataclasses import dataclass

import numpy as np
import pygame as pg

from pyrobo.core.geometry import planar_state

BG = '#111923'
PANEL = '#192430'
CARD = '#223140'
TEXT = '#e9f0f4'
MUTED = '#9aafbf'
GRID = '#22313e'
BLUE = '#55b8ff'
TEAL = '#59d4b0'
GOLD = '#f0c66c'
RED = '#ff7979'


@dataclass
class Viewport:
    bounds: object
    area: pg.Rect

    @property
    def scale(self):
        b = self.bounds
        return min(self.area.width/(b.xmax-b.xmin), self.area.height/(b.ymax-b.ymin))

    @property
    def rect(self):
        b, s = self.bounds, self.scale
        return pg.Rect(round(self.area.centerx - (b.xmax-b.xmin)*s/2),
                       round(self.area.centery - (b.ymax-b.ymin)*s/2),
                       round((b.xmax-b.xmin)*s), round((b.ymax-b.ymin)*s))

    def pixel(self, xy):
        b, s, rect = self.bounds, self.scale, self.rect
        return (round(rect.left+(xy[0]-b.xmin)*s), round(rect.bottom-(xy[1]-b.ymin)*s))

    def world(self, pixel):
        if not self.rect.collidepoint(pixel):
            return None
        b, s, rect = self.bounds, self.scale, self.rect
        return np.array([b.xmin+(pixel[0]-rect.left)/s, b.ymin+(rect.bottom-pixel[1])/s])


class View:
    def __init__(self, screen, session):
        self.screen, self.world, self.footprint, self.model = screen, session.world, session.footprint, session.model
        self.config = session.config
        self.points = session.cloud.points
        self.show_points = False
        pg.font.init()
        self.font = pg.font.Font(None, 23)
        self.small = pg.font.Font(None, 20)
        self.title = pg.font.Font(None, 31)
        self.large = pg.font.Font(None, 34)
        self.buttons = {}
        self.resize(screen)

    def resize(self, screen):
        self.screen = screen
        w, h = screen.get_size()
        self.sidebar = pg.Rect(w-310, 64, 310, h-64)
        self.viewport = Viewport(self.world.bounds, pg.Rect(28, 105, w-368, h-225))
        x = self.sidebar.x+18
        self.buttons = {name: pg.Rect(x+col*136, y, 128, 34) for name, col, y in
                        [('Manual', 0, 285), ('Auto', 1, 285), ('Pause', 0, 327), ('E-stop', 1, 327),
                         ('Release', 0, 369), ('Reset', 1, 369), ('Stop assist', 0, 431), ('Avoid assist', 1, 431)]}
        self.stick_center = (self.sidebar.centerx, h-110 if h < 780 else h-137)
        self.stick_radius = 52 if h < 780 else 65

    def text(self, value, pos, color=TEXT, font=None, max_width=None):
        font = font or self.font
        value = str(value)
        if max_width is not None:
            while value and font.size(value)[0] > max_width:
                value = value[:-2] + '…' if len(value) > 2 else ''
        self.screen.blit(font.render(value, True, color), pos)

    def wrapped(self, value, x, y, width, color=MUTED):
        line = ''
        for word in str(value).split():
            candidate = (line+' '+word).strip()
            if self.small.size(candidate)[0] > width and line:
                self.text(line, (x, y), color, self.small)
                line, y = word, y+18
            else:
                line = candidate
        if line:
            self.text(line, (x, y), color, self.small)
        return y+18

    def line(self, points, color, width=2):
        if len(points) > 1:
            pg.draw.lines(self.screen, color, False, [self.viewport.pixel(p) for p in points], width)

    def arrow(self, xy, yaw, color, length=.7, width=2):
        end = np.asarray(xy)+length*np.array([np.cos(yaw), np.sin(yaw)])
        start_px, end_px = self.viewport.pixel(xy), self.viewport.pixel(end)
        pg.draw.line(self.screen, color, start_px, end_px, width)
        for offset in (-.5, .5):
            tip = end - .14*np.array([np.cos(yaw+offset), np.sin(yaw+offset)])
            pg.draw.line(self.screen, color, end_px, self.viewport.pixel(tip), width)

    def draw(self, snapshot, inputs, *, drag=None, ui_notice=''):
        screen, vp = self.screen, self.viewport
        w, h = screen.get_size()
        screen.fill(BG)
        pg.draw.rect(screen, PANEL, (0, 0, w, 64))
        pg.draw.rect(screen, PANEL, self.sidebar)
        self.text('PYROBO  /  MOTION LAB', (24, 19), TEXT, self.title)
        cfg = self.config['backend']
        self.text(f"{cfg['planner'].upper()}   /   DWA   /   {cfg['model'].upper()}", (400, 25), MUTED, self.small)
        b = self.world.bounds
        screen.set_clip(vp.rect)
        pg.draw.rect(screen, '#16212c', vp.rect)
        for x in np.arange(np.ceil(b.xmin), b.xmax+1, 1):
            self.line([[x, b.ymin], [x, b.ymax]], GRID, 1)
        for y in np.arange(np.ceil(b.ymin), b.ymax+1, 1):
            self.line([[b.xmin, y], [b.xmax, y]], GRID, 1)
        for box in self.world.boxes:
            points = [vp.pixel(p) for p in box.corners]
            pg.draw.polygon(screen, '#344654', points)
            pg.draw.polygon(screen, '#738796', points, 2)
        if self.show_points:
            for point in self.points:
                pg.draw.circle(screen, MUTED, vp.pixel(point[:2]), 1)
        self.line(snapshot.trail, BLUE, 2)
        self.line(snapshot.path, GOLD, 3)
        if snapshot.motion and snapshot.motion.trajectory:
            self.line(snapshot.motion.trajectory.poses.position[:, :2], TEAL, 2)
        if snapshot.progress:
            pg.draw.circle(screen, TEAL, vp.pixel(snapshot.progress.target.translation[:2]), 5, 2)
        if snapshot.goal is not None:
            goal = planar_state(snapshot.goal)
            pg.draw.circle(screen, GOLD, vp.pixel(goal[:2]), 9, 2)
            self.arrow(goal[:2], goal[2], GOLD)
        state = snapshot.state
        body_color = RED if snapshot.collided or snapshot.mode == 'estop' else BLUE
        pg.draw.polygon(screen, '#234a63', [vp.pixel(p) for p in self.footprint.corners(state.pose)])
        pg.draw.polygon(screen, body_color, [vp.pixel(p) for p in self.footprint.corners(state.pose)], 3)
        pose = planar_state(state.pose)
        self.arrow(pose[:2], pose[2], TEXT, length=max(self.footprint.length*.65, .35), width=2)
        pg.draw.circle(screen, TEXT, vp.pixel(pose[:2]), 3)
        # Wheel illustrations follow measured steering; the outlined rectangle is
        # the authoritative footprint, independent of decorative axle placement.
        for front in (-1, 1):
            for side in (-1, 1):
                local = np.array([self.footprint.offset_x+front*self.footprint.length*.32,
                                  side*self.footprint.width*.54, 0.])
                center = state.pose.apply(local)[:2]
                angle = pose[2] + (state.steering if front == 1 and cfg['model'] == 'ackermann' else 0.)
                d = .09*np.array([np.cos(angle), np.sin(angle)])
                self.line([center-d, center+d], TEXT, 4)
        if drag:
            start, end = drag
            goal_xy = vp.world(start)
            if goal_xy is not None:
                pg.draw.circle(screen, GOLD, start, 9, 2)
                pg.draw.line(screen, GOLD, start, end, 2)
        screen.set_clip(None)
        pg.draw.rect(screen, '#516675', vp.rect, 1)
        self.text('Click to navigate  /  Drag for arrival heading  /  P: point cloud', (28, 78), MUTED, self.small)
        for label, color, x in [('GLOBAL ROUTE', GOLD, 30), ('FORECAST', TEAL, 192), ('DRIVEN TRACE', BLUE, 326)]:
            pg.draw.line(screen, color, (x, h-68), (x+20, h-68), 3)
            self.text(label, (x+28, h-74), MUTED, self.small)
        self.text('WASD drive    G auto    Space pause    E stop    R release    Home reset    Esc quit',
                  (28, h-37), MUTED, self.small, max_width=w-350)
        x = self.sidebar.x+18
        mode_color = RED if snapshot.mode == 'estop' else TEAL if snapshot.mode == 'auto' else GOLD
        self.text(str(snapshot.mode).upper(), (x, 82), mode_color, self.large)
        self.text(f"{snapshot.state.timestamp:6.1f} s   |   {snapshot.elapsed_ms:5.1f} ms / control", (x, 119), MUTED, self.small)
        self.text(f'Speed       {state.velocity.linear:+.3f} m/s', (x, 151))
        self.text(f'Yaw rate    {state.velocity.angular:+.3f} rad/s', (x, 177))
        self.text(f'Steering    {np.rad2deg(state.steering):+.1f} deg', (x, 203))
        self.text(f'Body        {self.footprint.length:g} x {self.footprint.width:g} m', (x, 229))
        self.text('CONTROL AUTHORITY', (x, 260), MUTED, self.small)
        for name, rect in self.buttons.items():
            active = (name.lower() == str(snapshot.mode) or
                      name == ('Stop assist' if snapshot.policy == 'teleop_stop' else 'Avoid assist'))
            fill = '#51313a' if name == 'E-stop' else '#345966' if active else CARD
            pg.draw.rect(screen, fill, rect, border_radius=6)
            self.text(name, (rect.x+12, rect.y+9), TEXT, self.small)
        self.text('MANUAL ASSISTANCE   [1 / 2]', (x, 409), MUTED, self.small)
        status = (snapshot.navigation_status if snapshot.mode == 'auto' else
                  snapshot.motion.status if snapshot.motion else 'ready')
        safe = snapshot.motion.safety_assessment if snapshot.motion else 'awaiting input'
        self.text(f'{status}  /  {snapshot.planner_status}', (x, 479), TEXT, self.small, max_width=275)
        self.text(safe, (x, 501), TEAL if snapshot.motion and snapshot.motion.safe else MUTED, self.small, max_width=275)
        if snapshot.motion:
            d = snapshot.motion.diagnostics
            self.text(f'Candidates {d.feasible_count}/{d.candidate_count}   Gap {d.clearance_m:.2f} m',
                      (x, 523), MUTED, self.small)
        notice = snapshot.fault or ui_notice or snapshot.notice
        self.wrapped(notice, 30, h-111, w-370, RED if snapshot.fault or snapshot.collided else MUTED)
        center, radius = self.stick_center, self.stick_radius
        pg.draw.circle(screen, CARD, center, radius)
        pg.draw.circle(screen, '#526a7b', center, radius, 2)
        pg.draw.line(screen, '#526a7b', (center[0]-radius+12, center[1]), (center[0]+radius-12, center[1]), 1)
        pg.draw.line(screen, '#526a7b', (center[0], center[1]-radius+12), (center[0], center[1]+radius-12), 1)
        throttle, turn = inputs.virtual if inputs.virtual_active else (0., 0.)
        knob = (round(center[0]-turn*(radius-12)), round(center[1]-throttle*(radius-12)))
        pg.draw.circle(screen, TEAL if inputs.virtual_active else BLUE, knob, 15)
        self.text('DRAG TO DRIVE', (center[0]-53, center[1]+radius+12), MUTED, self.small)
        self.text(inputs.device_status, (x, h-30), MUTED, self.small, max_width=275)
        pg.display.flip()
