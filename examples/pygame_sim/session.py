"""Single-owner simulation and bounded UI mailbox; no SDL dependency."""
from collections import deque
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from time import monotonic, perf_counter

import numpy as np

from pyrobo import (ChassisDynamics, ControlMode, DWAEngine, MotionPolicy, NavigationConfig,
                    Navigator, RobotObservation, SafetyGuard, Supervisor, TeleopAdapter, Twist2D, pose2d)
from pyrobo.core.geometry import planar_state
from pyrobo.geometry.cache import SnapshotIndex
from pyrobo.planning import RRTConnect, VoronoiPlanner
from pyrobo.simulation import Simulator2D
from .config import build_parameters, make_world, parameters


@dataclass(frozen=True)
class ControlInput:
    throttle: float = 0.
    turn: float = 0.
    engaged: bool = False
    source: str = 'keyboard'
    focused: bool = True
    timestamp: float = field(default_factory=monotonic)


@dataclass(frozen=True)
class Action:
    kind: str
    value: object = None


@dataclass(frozen=True)
class Snapshot:
    state: object
    mode: ControlMode
    policy: MotionPolicy
    goal: object
    path: np.ndarray
    trail: np.ndarray
    motion: object
    progress: object
    navigation_status: str
    planner_status: str
    collided: bool
    elapsed_ms: float
    notice: str
    ticks: int
    fault: str = ''


def readonly(values):
    result = np.array(values, dtype=float).reshape(-1, 2)
    result.setflags(write=False)
    return result


class Session:
    def __init__(self, config):
        self.config = config
        self.world, start, goal = make_world(config)
        model, footprint, dynamics, dwa, planner, safety = build_parameters(config)
        self.model, self.footprint = model, footprint
        self.cloud = self.world.point_cloud(config['app']['point_spacing'])
        plant, index = ChassisDynamics(model, dynamics), SnapshotIndex()
        optimizer = DWAEngine(dwa, footprint, model=model, dynamics=plant, index=index, input_limits=safety)
        global_planner = (RRTConnect if config['backend']['planner'] == 'rrt_connect' else VoronoiPlanner)(planner, footprint)
        gate = SafetyGuard(plant, optimizer.command_space, footprint, index=index, config=safety)
        settings = parameters(NavigationConfig, {
            'lookahead': dwa.lookahead, 'goal_tolerance': dwa.goal_tolerance, 'yaw_tolerance': dwa.yaw_tolerance,
            **config['navigation']}, 'navigation')
        self.nav = Navigator(model=model, footprint=footprint, dynamics=plant, global_planner=global_planner,
                             local_optimizer=optimizer, settings=settings, index=index, safety=gate,
                             max_cloud_age=safety.max_cloud_age)
        teleop = config['teleop']
        self.policy = MotionPolicy(teleop['policy'])
        adapter = TeleopAdapter(optimizer.command_space, policy=self.policy, ttl=teleop['ttl'],
                               **{key: teleop[key] for key in ('max_turn_deviation', 'max_lateral_deviation', 'max_heading_deviation')})
        self.runtime = Supervisor(self.nav, teleop=adapter, mode=config['app']['start_mode'],
                                  authorized_sources=('ui', 'keyboard', 'virtual', 'gamepad'))
        self.sim = Simulator2D(self.world, start, model=model, footprint=footprint, dynamics=dynamics)
        if goal is not None:
            self.runtime.set_goal(goal, 0., source='ui')
        self.trail = deque([start.translation[:2].copy()], maxlen=config['app']['trail_length'])
        self.motion, self.ticks, self.elapsed_ms = None, 0, 0.
        self.planner_status = 'not planned'
        self.notice = 'Click a goal or use WASD / the stick to drive.'
        self.fault = ''

    def set_mode(self, mode):
        mode = ControlMode(mode)
        if self.runtime.mode == ControlMode.ESTOP and mode != ControlMode.ESTOP:
            self.notice = 'E-stop is latched. Release with R, then choose a mode.'
            return
        if mode == ControlMode.AUTO and self.nav.goal is None:
            self.notice = 'Click the map to set a goal first.'
            return
        self.runtime.set_mode(mode, source='ui')

    def apply(self, action):
        if action.kind == 'mode':
            self.set_mode(action.value)
        elif action.kind == 'estop':
            self.set_mode(ControlMode.ESTOP)
            self.notice = 'E-stop latched. R releases to PAUSED.'
        elif action.kind == 'release':
            if self.runtime.mode == ControlMode.ESTOP:
                self.runtime.release_estop(source='ui')
                self.notice = 'Released to PAUSED. Choose a mode or drive again.'
        elif action.kind == 'reset':
            self.__init__(self.config)
            self.set_mode(ControlMode.PAUSED)
            self.notice = 'Scene reset. Chassis and task restarted.'
        elif action.kind == 'policy':
            policy = MotionPolicy(action.value)
            if policy not in (MotionPolicy.TELEOP_STOP, MotionPolicy.TELEOP_AVOID):
                raise ValueError('manual policy must be stop or avoid')
            self.policy = policy
            self.notice = 'Manual assistance: ' + policy.value
        elif action.kind == 'goal':
            goal = pose2d(*action.value, frame=self.world.frame, body='goal')
            index = self.nav.index.get(self.cloud)
            if (self.world.collides(goal, self.footprint, padding=self.footprint.margin) or
                    index.clearances(planar_state(goal)[None], self.footprint, self.world.bounds)[0] <= 0):
                self.notice = 'Goal rejected: the rectangle needs more clearance.'
                return
            self.runtime.set_goal(goal, self.sim.state.timestamp, source='ui')
            self.set_mode(ControlMode.AUTO)
            if self.runtime.mode != ControlMode.ESTOP:
                self.notice = 'Goal accepted. WASD can take over; G resumes navigation.'
        else:
            raise ValueError(f'unknown action: {action.kind}')

    def tick(self, controls=None, actions=(), *, emergency=False, before_execute=None):
        controls = controls or ControlInput()
        started = perf_counter()
        for action in actions:
            self.apply(action)
        if emergency:
            self.apply(Action('estop'))
        if self.sim.collided or self.fault:
            return self.snapshot()
        fresh = controls.focused and monotonic() - controls.timestamp <= self.config['app']['input_timeout']
        if not fresh:
            self.set_mode(ControlMode.PAUSED)
            self.notice = 'Input suspended: focus lost or UI heartbeat expired.'
        elif controls.engaged and self.runtime.mode != ControlMode.ESTOP:
            self.set_mode(ControlMode.MANUAL)
        if self.runtime.mode == ControlMode.MANUAL:
            throttle, turn = (float(np.clip(v, -1., 1.)) for v in (controls.throttle, controls.turn))
            cfg, space = self.config['teleop'], self.nav.command_space
            lo, hi = space.bounds()
            speed = throttle * min(cfg['speed_limit'], hi[0] if throttle >= 0 else -lo[0])
            turn_limit = min(cfg['turn_limit'] or hi[1], hi[1])
            angular, steering = (turn * turn_limit, None) if space.can_spin else (0., turn * turn_limit)
            self.runtime.submit_teleop(Twist2D(speed, angular), self.sim.state.timestamp,
                                      steering=steering, source=controls.source, policy=self.policy)
        observation = RobotObservation(self.sim.state, self.cloud, self.world.bounds, self.config['app']['dt'])
        motion = self.runtime.step(observation)
        # A result computed before an emergency, focus loss or new UI action must
        # not move the simulation on behalf of an obsolete input.
        if before_execute is not None:
            interrupted, emergency = before_execute()
            if interrupted or emergency:
                self.set_mode(ControlMode.ESTOP if emergency else ControlMode.PAUSED)
                motion = self.nav.safety.stop(observation, reason='interactive_input_changed')
        self.motion = motion
        if self.runtime.mode == ControlMode.AUTO:
            output = self.nav.navigation_output(observation, motion)
            if output.diagnostics['replanned']:
                self.planner_status = str(output.diagnostics['planner_status'])
            self.nav.status = output.status
        # Even an unverified stop is a finite-force braking attempt. Independent
        # solid collision truth terminates motion if braking cannot prevent impact.
        self.sim.step(motion.command, observation.dt)
        self.ticks += 1
        self.trail.append(self.sim.state.pose.translation[:2].copy())
        if self.sim.collided:
            self.set_mode(ControlMode.ESTOP)
            self.notice = 'Solid collision. Home starts a new experiment.'
        self.elapsed_ms = (perf_counter()-started)*1000
        return self.snapshot()

    def snapshot(self):
        return Snapshot(self.sim.state, self.runtime.mode, self.policy, self.nav.goal, readonly(self.nav.path.xy),
                        readonly(self.trail), self.motion, self.nav.progress, str(self.nav.status), self.planner_status,
                        self.sim.collided, self.elapsed_ms, self.notice, self.ticks, self.fault)


class Worker:
    """One worker owns all backend state; SDL and drawing stay on the main thread."""
    def __init__(self, session):
        self.session = session
        self._lock, self._stop, self._urgent = Lock(), Event(), Event()
        self._controls, self._actions, self._revision = ControlInput(), deque(), 0
        self._snapshot = session.snapshot()
        self._thread = Thread(target=self._run, name='pyrobo-pygame-simulation', daemon=True)

    def start(self):
        self._thread.start()

    def publish(self, controls):
        with self._lock:
            self._controls = controls

    def enqueue(self, action):
        with self._lock:
            if action.kind == 'estop':
                self._urgent.set()  # retained even if the ordinary mailbox is full
            if action.kind == 'goal':
                self._actions = deque(a for a in self._actions if a.kind != 'goal')
            if len(self._actions) >= 64:
                return action.kind == 'estop'
            self._actions.append(action)
            self._revision += 1
            return True

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def _run(self):
        while not self._stop.is_set():
            started = perf_counter()
            with self._lock:
                controls, actions, revision = self._controls, tuple(self._actions), self._revision
                self._actions.clear()
                emergency = self._urgent.is_set()
                self._urgent.clear()
            def before_execute():
                with self._lock:
                    latest = self._controls
                    interrupted = (self._stop.is_set() or self._revision != revision or not latest.focused or
                                   monotonic()-latest.timestamp > self.session.config['app']['input_timeout'] or
                                   (controls.engaged and not latest.engaged))
                    return interrupted, self._urgent.is_set()
            try:
                snapshot = self.session.tick(controls, actions, emergency=emergency, before_execute=before_execute)
            except Exception as error:
                # Keep the window and Reset action usable after a backend error.
                self.session.fault = f'{type(error).__name__}: {error}'
                self.session.set_mode(ControlMode.ESTOP)
                snapshot = self.session.snapshot()
            with self._lock:
                self._snapshot = snapshot
            # No catch-up bursts: expensive planning slows simulated time.
            self._stop.wait(max(0., self.session.config['app']['dt']-(perf_counter()-started)))

    def close(self):
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=3.)
