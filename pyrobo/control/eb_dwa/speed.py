"""Advisory spatial speed limits from curvature and shared plant rollouts.

These sampled response envelopes shape tracking preferences. They are not a
reachable-set proof; every candidate still passes the original physical gates.
"""
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from ...core import Twist2D
from ...models.dynamics.prediction import BatchPredictor


@dataclass(frozen=True)
class SpeedProfile:
    values: np.ndarray
    curvature_limit: np.ndarray
    steering_limit: np.ndarray
    diagnostics: dict


class SpeedPlanner:
    def __init__(self, plant, space, footprint, config):
        self.plant, self.space, self.config = plant, space, config
        self.predictor = BatchPredictor(plant, footprint, prediction_time=config.speed_response_time_s,
            recording_dt=config.integration_dt, braking_timeout=config.braking_timeout)
        self.brake_predictor = BatchPredictor(plant, footprint, prediction_time=config.braking_timeout,
            recording_dt=config.integration_dt, braking_timeout=config.braking_timeout)
        self.reset()

    def reset(self):
        self._tables = {}

    def _response_table(self, observation, gear):
        model, plant, cfg = self.space.model, self.plant, self.config
        key = (gear, observation.dt, plant.config, model)
        if key in self._tables:
            return (*self._tables[key], True)
        top = model.max_speed if gear > 0 else model.max_reverse_speed
        speed = np.linspace(0., top, cfg.speed_profile_samples)
        # Sample straight and both turning extremes. In a car, steering-rate
        # extremes also change the turning friction budget during braking.
        turns = [0., -model.max_yaw_rate, model.max_yaw_rate] if self.space.can_spin else [0., -model.max_steering, model.max_steering]
        rates = [0.] if self.space.can_spin else [0., -model.max_steering_rate, model.max_steering_rate]
        states, commands = [], []
        for magnitude in speed:
            for turn in turns:
                for rate in rates:
                    v = gear*magnitude
                    w = turn if self.space.can_spin else model.yaw_rate(v, turn)
                    state = replace(observation.chassis, velocity=Twist2D(v, w),
                        steering=0. if self.space.can_spin else turn, steering_rate=rate,
                        acceleration=0., yaw_acceleration=0., steering_acceleration=0.)
                    states.append(plant.codec.pack(state))
                    commands.append(model.encode_command(model.command(gear*top, turn)))
        states, commands = np.asarray(states), np.asarray(commands)
        # Track a zero-speed reference before the predictor's emergency tail.
        # Ordinary tracking respects servo/jerk limits and generally slows much
        # earlier than an estop. Using only emergency distance here encourages
        # arriving at a bend too fast for an ordinary DWA command to slow down.
        brake = self.brake_predictor.predict(states, np.zeros_like(commands), observation.dt)
        velocities = abs(plant.codec.velocities(brake.states)[..., 0])
        distance = np.sum((velocities[:, :-1]+velocities[:, 1:])*.5*np.diff(brake.times)*brake.valid_time_mask[:, 1:], axis=1)
        distance[~brake.stopped] = np.inf
        distance = np.max(distance.reshape(len(speed), -1), axis=1)
        distance = np.maximum.accumulate(distance)
        drive = self.predictor.predict(states, commands, observation.dt)
        duration = drive.times[drive.prediction_end]
        final = abs(plant.codec.velocities(drive.states[:, drive.prediction_end])[:, 0]).reshape(len(speed), -1)
        acceleration = np.maximum(0., np.min(final-speed[:, None], axis=1)/duration)
        self._tables[key] = (speed, distance, acceleration)
        # Model/config are fixed for this controller; retain only both gears for
        # the current control interval, never an unbounded dt history.
        self._tables = {k: v for k, v in self._tables.items() if k[1:] == key[1:]}
        return speed, distance, acceleration, False

    def update(self, reference, observation, *, terminal=False):
        started = perf_counter()
        cfg, model = self.config, self.space.model
        lower, upper = self.space.bounds()
        maximum = upper[0] if reference.gear > 0 else -lower[0]
        curvature = abs(reference.curvature)
        lateral = min(self.plant.config.lateral_acceleration,
                      self.plant.config.friction*self.plant.config.gravity)*cfg.speed_lateral_fraction
        curve_limit = np.minimum(np.sqrt(lateral/np.maximum(curvature, 1e-12)),
                                 model.max_yaw_rate/np.maximum(curvature, 1e-12))
        steering_limit = np.full(len(curvature), maximum)
        if len(curvature) > 1:
            if self.space.can_spin:
                rate = abs(np.gradient(reference.curvature, reference.s))
                steering_limit = np.sqrt(self.plant.config.yaw_acceleration/np.maximum(rate, 1e-12))
            else:
                delta = np.arctan(model.wheelbase*reference.curvature)
                rate = abs(np.gradient(delta, reference.s))
                steering_limit = model.max_steering_rate/np.maximum(rate, 1e-12)
                steering_limit[abs(delta) > model.max_steering+1e-9] = 0.
        caps = np.minimum.reduce((np.full(len(curvature), maximum), curve_limit, steering_limit))
        if terminal:
            # The DWA hold command cannot switch to a slower reference halfway
            # through its horizon. Taper its terminal tracking preference over
            # that horizon, otherwise a curved candidate can keep a high speed
            # by staying just short of the endpoint instead of approaching it.
            caps = np.minimum(caps, (reference.s[-1]-reference.s)/cfg.prediction_time)
        speed, stop_distance, acceleration, cached = self._response_table(observation, reference.gear)
        finite = np.isfinite(stop_distance)
        if not finite.any():
            values = np.zeros_like(caps)
        else:
            table_v, table_d = speed[finite], stop_distance[finite]
            values = np.minimum(caps, table_v[-1])
            if terminal:
                values[-1] = 0.
            # A backward distance pass through actual sampled braking responses,
            # followed by an acceleration pass through sampled drive responses.
            # Different future curvature/actuator states can depart from this
            # approximation: the complete DWA/SafetyGuard checks remain binding.
            for i in range(len(values)-2, -1, -1):
                allowance = reference.s[i+1]-reference.s[i]+np.interp(values[i+1], table_v, table_d)
                values[i] = min(values[i], np.interp(allowance, table_d, table_v))
            values[0] = min(values[0], abs(observation.chassis.velocity.linear))
            for i in range(1, len(values)):
                a = np.interp(values[i-1], speed, acceleration)
                limit = np.sqrt(values[i-1]**2+2*a*(reference.s[i]-reference.s[i-1]))
                values[i] = min(values[i], limit)
        signed = reference.gear*values
        for array in (signed, curve_limit, steering_limit):
            array.setflags(write=False)
        limited = np.flatnonzero(caps < maximum-.05)
        diagnostics = {'eb_speed_profile_duration_s': perf_counter()-started,
            'eb_speed_table_cached': cached, 'eb_speed_min_mps': float(values.min()),
            'eb_speed_max_mps': float(values.max()), 'eb_speed_profile_is_outer_bound': False,
            'eb_speed_limit_ahead_m': float(reference.s[limited[0]]) if len(limited) else None,
            'eb_speed_profile_mps': signed.tolist()}
        return SpeedProfile(signed, curve_limit, steering_limit, diagnostics)
