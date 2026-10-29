"""SDL input normalization; source selection never writes chassis state."""
from time import monotonic

import numpy as np
import pygame as pg

from .session import ControlInput

DRIVE_KEYS = {pg.K_w, pg.K_s, pg.K_a, pg.K_d, pg.K_UP, pg.K_DOWN, pg.K_LEFT, pg.K_RIGHT}


def deadzone(value, threshold):
    value = float(np.clip(value, -1., 1.))
    return 0. if abs(value) <= threshold else np.sign(value)*(abs(value)-threshold)/(1-threshold)


class Inputs:
    def __init__(self, config):
        self.config = config
        self.keys = set()
        self.virtual_active = False
        self.virtual = (0., 0.)
        self.focused = True
        self.needs_neutral = False
        self.devices, self.selected_id = {}, None
        self.device_status = 'No gamepad connected' if config['enabled'] else 'Gamepad disabled'
        self.device_lost = False
        pg.joystick.init()
        if config['enabled']:
            for index in range(pg.joystick.get_count()):
                self.add_device(index)

    def add_device(self, index):
        if not self.config['enabled']:
            return
        try:
            device = pg.joystick.Joystick(index)
            self.devices[device.get_instance_id()] = device
            if self.selected_id is None and len(self.devices) > self.config['device_index']:
                self.selected_id = list(self.devices)[self.config['device_index']]
        except pg.error:
            self.device_status = 'Gamepad unavailable'

    def remove_device(self, instance_id):
        device = self.devices.pop(instance_id, None)
        selected = instance_id == self.selected_id
        if device is not None:
            device.quit()
        if selected:
            self.selected_id = None
            self.device_status = 'Gamepad disconnected'
            self.device_lost = True
            self.suppress()
        return selected

    def suppress(self):
        """Mode/safety changes require held controls to return to neutral first."""
        self.needs_neutral = True
        self.virtual_active = False
        self.virtual = (0., 0.)

    def focus(self, active):
        self.focused = active
        self.keys.clear()
        self.suppress()

    def key(self, key, down):
        if key in DRIVE_KEYS:
            (self.keys.add if down else self.keys.discard)(key)

    def move_virtual(self, pos, center, radius):
        x, y = (np.asarray(pos) - center) / radius
        vector = np.array([-y, -x])
        vector /= max(1., np.linalg.norm(vector))
        self.virtual = tuple(vector)

    def gamepad(self):
        device = self.devices.get(self.selected_id)
        if device is None:
            return 0., 0., False
        cfg = self.config
        try:
            self.device_status = device.get_name()
            if max(cfg['throttle_axis'], cfg['turn_axis']) >= device.get_numaxes():
                self.device_status = 'Configured gamepad axis unavailable'
                return 0., 0., False
            button = cfg['deadman_button']
            if button >= device.get_numbuttons():
                self.device_status = 'Configured deadman button unavailable'
                return 0., 0., False
            throttle = deadzone(device.get_axis(cfg['throttle_axis']), cfg['deadzone'])
            turn = deadzone(device.get_axis(cfg['turn_axis']), cfg['deadzone'])
            throttle *= -1 if cfg['invert_throttle'] else 1
            turn *= -1 if cfg['invert_turn'] else 1
            enabled = bool(device.get_button(button)) if button >= 0 else bool(throttle or turn)
            return (throttle, turn, True) if enabled else (0., 0., False)
        except pg.error:
            self.selected_id = None
            self.device_lost = True
            self.device_status = 'Gamepad read failed'
            return 0., 0., False

    def sample(self):
        gamepad = self.gamepad()
        k = self.keys
        throttle = int(bool(k & {pg.K_w, pg.K_UP})) - int(bool(k & {pg.K_s, pg.K_DOWN}))
        turn = int(bool(k & {pg.K_a, pg.K_LEFT})) - int(bool(k & {pg.K_d, pg.K_RIGHT}))
        if k:
            raw = (throttle, turn, True, 'keyboard')
        elif self.virtual_active:
            raw = (*self.virtual, True, 'virtual')
        else:
            raw = (*gamepad, 'gamepad' if gamepad[2] else 'keyboard')
        if self.needs_neutral and not raw[2]:
            self.needs_neutral = False
        if not self.focused or self.needs_neutral:
            raw = (0., 0., False, raw[3])
        return ControlInput(*raw, focused=self.focused, timestamp=monotonic())

    def close(self):
        for device in self.devices.values():
            device.quit()
        self.devices.clear()
