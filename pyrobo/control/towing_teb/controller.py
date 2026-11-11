"""Standalone TEB controller: optimize a band, execute its first command."""
from dataclasses import replace
from time import perf_counter

import numpy as np

from ...core import (Diagnostics, Footprint, MotionPolicy, MotionResult, MotionStatus,
                     SafetyAssessment)
from ...core.geometry import planar_state, wrap_angle
from ...geometry.cache import SnapshotIndex
from ...models.dynamics import ChassisDynamics
from ...models.dynamics.prediction import BatchPredictor
from ...models.kinematics import DifferentialDrive, command_space_for
from ...safety import SafetyGuard
from ...safety.fallback import selected_trajectory, verified_stop
from ..dwa import DWAEngine
from .config import TowingTEBConfig
from .execution import check_command
from .optimizer import segment_motion, solve_band


from pyrobo.registry import register_local_optimizer


@register_local_optimizer('towing_teb', config=TowingTEBConfig, label='towing_teb · 牵引车 TEB', family='teb', teleop_config=True)
class TowingTEB:
    name = 'towing_teb'

    def __init__(self, config=None, footprint=None, *, model=None, dynamics=None,
                 command_space=None, index=None, input_limits=None, teleop_config=None):
        config = config or TowingTEBConfig()
        # 车体几何与运动模型由调用方独立注入，不由算法名称选择。
        footprint = footprint or Footprint()
        self.config, self.footprint = config, footprint
        self.model = (model or getattr(dynamics, 'model', None) or
                      getattr(command_space, 'model', None) or DifferentialDrive())
        self.command_space = command_space or command_space_for(self.model)
        self.plant = dynamics if hasattr(dynamics, 'integrate') else ChassisDynamics(self.model, dynamics)
        if self.plant.model != self.model or self.command_space.model != self.model:
            raise ValueError('dynamics and command space must use the same model')
        self.index = index or SnapshotIndex()
        self._predictor = BatchPredictor(self.plant, footprint, prediction_time=config.prediction_time,
                                       recording_dt=config.integration_dt, braking_timeout=config.braking_timeout)
        self._input_guard = SafetyGuard(self.plant, self.command_space, footprint, index=self.index, config=input_limits)
        # Only manual requests reach this delegate; autonomous planning never
        # invokes the old DWA NavigationPolicy, recovery logic or scoring.
        self._teleop = DWAEngine(teleop_config, footprint, model=self.model, dynamics=self.plant,
                                command_space=self.command_space, index=self.index, input_limits=input_limits)

        self.last_band = None

    def reset(self):
        self._teleop.reset()
        self.last_band = None

    def _prepare(self, observation, request):
        if request.policy == MotionPolicy.NAVIGATION:
            if request.path.frame != observation.frame or request.goal.wrd_frame != observation.frame:
                raise ValueError('goal and path frames must match the robot')
        world = self.index.get(observation.obstacles)
        reason = self._input_guard.input_reason(observation, request)
        if reason or request.policy == MotionPolicy.STOP:
            return world, self._stop(observation, world, reason=reason or 'stop_requested',
                mode=request.stop_mode, assessment=SafetyAssessment.INPUT_STALE if reason else None)
        if request.policy != MotionPolicy.NAVIGATION:
            result = self._teleop.optimize(observation, request, collision_world=world)
            return world, replace(result, diagnostics=replace(result.diagnostics,
                extra={**result.diagnostics.extra, 'optimizer': 'dwa', 'navigation_optimizer': self.name}))
        return world, None

    def _stop(self, observation, world, **kwargs):
        return verified_stop(observation, self._predictor, self.command_space, world,
                             extra={'optimizer': self.name}, **kwargs)

    def _seed_band(self, observation, request):
        """Resample the supplied route as an initial guess, never a path penalty."""
        cfg = self.config
        start, goal = planar_state(observation.chassis.pose), planar_state(request.goal)
        count = int(np.clip(np.ceil(cfg.prediction_time/cfg.dt_ref)+1, cfg.min_samples, cfg.max_samples))
        if np.linalg.norm(start[:2]-goal[:2]) <= cfg.goal_tolerance and self.command_space.can_spin:
            poses = np.column_stack((np.tile(start[:2], (count, 1)),
                start[2]+np.linspace(0., float(wrap_angle(goal[2]-start[2])), count)))
            final = True
        else:
            route = request.remaining_path if request.remaining_path is not None else request.path
            # Direct optimize() callers need not supply progress. Trim only the
            # initialization to the nearest segment; Navigator supplies it normally.
            xy = route.xy
            if request.remaining_path is None and len(xy) > 1:
                delta = np.diff(xy, axis=0)
                t = np.clip(np.sum((start[:2]-xy[:-1])*delta, axis=1)/
                            np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0., 1.)
                projected = xy[:-1]+t[:, None]*delta
                segment = int(np.argmin(np.linalg.norm(projected-start[:2], axis=1)))
                xy = np.vstack((projected[segment], xy[segment+1:]))
            xy = np.vstack((start[:2], xy))
            xy = xy[np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-8]]
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            length = min(distance[-1], max(cfg.lookahead, self.model.max_speed*cfg.prediction_time))
            samples = np.linspace(0., length, count)
            points = np.column_stack([np.interp(samples, distance, xy[:, j]) for j in range(2)])
            delta = np.diff(points, axis=0)
            yaw = np.arctan2(delta[:, 1], delta[:, 0])
            if cfg.allow_reverse and delta[0] @ np.array([np.cos(start[2]), np.sin(start[2])]) < 0:
                yaw += np.pi
            final = bool(length >= distance[-1]-1e-8)
            poses = np.column_stack((points, np.r_[yaw, goal[2] if final else yaw[-1]]))
            poses[0] = start
            poses[:, 2] = start[2]+np.unwrap(wrap_angle(poses[:, 2]-start[2]))
        travel = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)/self.model.max_speed
        turn = np.abs(wrap_angle(np.diff(poses[:, 2])))/self.model.max_yaw_rate
        intervals = np.clip(np.maximum(travel, turn), cfg.dt_min+1e-9, cfg.dt_max-1e-9)
        return poses, intervals, final

    def optimize(self, observation, request):
        """局部规划入口，实际目标函数在 optimizer.solve_band() 的 residual()。

        外部每周期传 RobotObservation 与 MotionRequest：chassis 是实测位姿、
        速度、转角与加速度；obstacles.points 是世界系完整 XYZ 点云；bounds 是
        工作边界；dt 是下一执行周期。request.path / remaining_path / local_target /
        goal 提供全路径、剩余路径、前视目标、终点；policy 与 timestamp+ttl 区分
        模式和时效。详情和独立调用示例见 ../README.md。

        模型/尺寸/物理从 self.model / footprint / plant.config 获取；碰撞索引从
        self.index.get(observation.obstacles) 获取。返回 MotionResult：原生 command、
        status、safety_assessment、下一周期执行及其制动尾段的 trajectory、diagnostics。
        last_band 只是优化结果，不能当作真实物理执行轨迹。reset() 清本地缓存。
        """
        started = perf_counter()
        world, handled = self._prepare(observation, request)
        if handled is not None:
            self.last_band = None
            return handled
        seed, intervals, final = self._seed_band(observation, request)
        band = solve_band(seed, intervals, observation, world, self.footprint,
            self.command_space, self.plant.config, self.config, goal_velocity=(0., 0.) if final else None)
        self.last_band = band
        v, w, _ = segment_motion(band.poses, band.intervals)
        turn = w[0] if self.command_space.can_spin else np.arctan2(
            self.model.wheelbase*w[0]*v[0], v[0]*v[0]+1e-12)
        lo, hi = self.command_space.bounds()
        if not self.config.allow_reverse:
            lo[0] = 0.
        search = np.clip([[v[0], turn]], lo, hi)
        if self.command_space.can_spin:
            wheel_speed = abs(search[0, 0])+.5*self.model.track*abs(search[0, 1])
            search /= max(1., wheel_speed/self.model.max_wheel_speed)
        extra = {'teb_status': band.status, 'teb_nodes': len(band.poses),
                 'teb_evaluations': band.evaluations, 'teb_iterations': band.iterations,
                 'teb_cost': band.cost, 'teb_initial_cost': band.initial_cost,
                 'teb_duration_s': float(band.times[-1]), 'teb_warm_started': False}
        # One command extracted from the optimized band; no DWA rescoring,
        # multi-preview/scaled candidates, turn latch or recovery state machine.
        return self._command_result(observation, world, search[0], started, extra)

    def _command_result(self, observation, world, search, started, extra):
        # TEB has one command extracted from the band, not a DWA candidate set.
        extra = {'optimizer': self.name, 'trajectory_kind': 'execution_then_brake', **extra}
        native, valid = self.command_space.to_native_batch(search[None])
        if valid[0]:
            batch, clearance, verified = check_command(self._predictor,
                self.plant.codec.pack(observation.chassis), native[0], observation.dt, world, observation.bounds)
        else:
            verified = False
        if not verified:
            reason = 'command_execution_infeasible' if valid[0] else 'command_not_representable'
            result = self._stop(observation, world, reason=reason)
            return replace(result, diagnostics=replace(result.diagnostics,
                elapsed_s=perf_counter()-started, candidate_count=1, feasible_count=0,
                extra={**result.diagnostics.extra, **extra, 'controller_mode': 'blocked'}))
        diagnostics = Diagnostics(elapsed_s=perf_counter()-started, candidate_count=1, feasible_count=1,
            clearance_m=clearance, selected_command=tuple(native[0]),
            extra={**extra, 'search_command': tuple(search)})
        return MotionResult(self.command_space.decode_one(native[0]), MotionStatus.TRACKING,
            SafetyAssessment.SAFE_UNDER_MODEL,
            selected_trajectory(batch, 0, observation, self.plant.codec), diagnostics)


TowingTEBEngine = TowingTEB
