"""Standalone DWA controller; all sampling/selection decisions are local."""
from dataclasses import replace
from time import perf_counter

import numpy as np

from ...core import (Diagnostics, Footprint, MotionPolicy, MotionResult, MotionStatus,
                     SafetyAssessment)
from ...core.geometry import planar_state
from ...geometry.cache import SnapshotIndex
from ...models.dynamics import ChassisDynamics
from ...models.dynamics.prediction import BatchPredictor
from ...models.kinematics import DifferentialDrive, command_space_for
from ...safety import SafetyGuard
from ...safety.fallback import selected_trajectory, verified_stop
from ..dwa import DWAEngine
from .config import RefDWAConfig
from .optimizer import objective, velocity_window
from .rollout import evaluate_commands


from pyrobo.registry import register_local_optimizer


@register_local_optimizer('ref_dwa', config=RefDWAConfig, label='ref_dwa · 基础 DWA', family='dwa', teleop_config=True)
class RefDWA:
    name = 'ref_dwa'

    def __init__(self, config=None, footprint=None, *, model=None, dynamics=None,
                 command_space=None, index=None, input_limits=None, teleop_config=None):
        config = config or RefDWAConfig()
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

    def reset(self):
        self._teleop.reset()

    def _prepare(self, observation, request):
        if observation.dt > self.config.prediction_time:
            raise ValueError('control dt must not exceed prediction_time')
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

    def optimize(self, observation, request):
        """每周期唯一入口；返回 MotionResult，不在这里执行命令。

        observation.chassis 提供实测 pose、velocity、steering、加速度历史；
        observation.obstacles.points[:, :2] 是世界系点云 XY，原始 XYZ 保留；
        observation.dt 是下一执行周期，timestamp/frame/bounds 是时间/坐标系/边界。
        request.path、remaining_path、local_target、goal 由导航器传入；
        request.policy 区分导航/手动/停车，timestamp+ttl 决定请求期限。
        非导航请求由本控制器私有分支处理，不进入这个基础目标函数。

        self.model / footprint / plant.config 给出车体与物理参数；
        self.index.get(observation.obstacles) 给出当前点云的碰撞查询。
        result.command 是 WheelCommand 或 AckermannCommand；trajectory 是真实
        plant 预测（含制动段），diagnostics 为数值诊断。详见 ../README.md。
        """
        started = perf_counter()
        world, handled = self._prepare(observation, request)
        if handled is not None:
            return handled
        state = self.plant.codec.pack(observation.chassis)
        current = self.command_space.state_coordinates(state[None])[0]
        lower, upper = self.command_space.bounds()
        if not self.config.allow_reverse:
            lower[0] = 0.
        acceleration = np.array([self.plant.config.drive_acceleration,
            self.plant.config.yaw_acceleration if self.command_space.can_spin else self.model.max_steering_rate])
        goal = planar_state(request.goal)
        at_goal = np.linalg.norm(state[:2]-goal[:2]) <= self.config.goal_tolerance
        # Final yaw is an interface requirement; use the same heading objective.
        if at_goal and self.command_space.can_spin:
            lower[0] = upper[0] = 0.
        search = velocity_window(current, lower, upper, acceleration, observation.dt,
                                 (self.config.linear_samples, self.config.angular_samples))
        if not len(search):
            return self._stop(observation, world, reason='empty_dynamic_window')
        native, command_ok = self.command_space.to_native_batch(search)
        batch, clearance, feasible = evaluate_commands(self._predictor, state, native,
            observation.dt, world, observation.bounds)
        evaluated = native, batch, clearance, feasible & command_ok
        target = planar_state(request.local_target if request.local_target is not None else request.goal)[:2]
        terminal = self.plant.codec.poses(evaluated[1].states[:, evaluated[1].prediction_end])
        costs, terms = objective(terminal, search, evaluated[2], target, self.model.max_speed,
                                self.config, goal_yaw=goal[2] if at_goal else None)
        return self._select_result(observation, world, search, evaluated, costs, terms, started)

    def _select_result(self, observation, world, search, evaluated, costs, terms, started, *, extra=None):
        native, batch, clearance, feasible = evaluated
        extra = {'optimizer': self.name, **(extra or {})}
        if not feasible.any():
            result = self._stop(observation, world, reason='no_feasible_candidate')
            return replace(result, diagnostics=replace(result.diagnostics,
                elapsed_s=perf_counter()-started, candidate_count=len(search), feasible_count=0,
                extra={**result.diagnostics.extra, **extra, 'controller_mode': 'blocked'}))
        index = int(np.argmin(np.where(feasible, costs, np.inf)))
        diagnostics = Diagnostics(elapsed_s=perf_counter()-started, candidate_count=len(search),
            feasible_count=int(feasible.sum()), clearance_m=float(clearance[index]),
            selected_command=tuple(native[index]), cost_terms={k: float(v[index]) for k, v in terms.items()},
            extra={**extra, 'search_command': tuple(search[index]), 'candidate_id': index})
        return MotionResult(self.command_space.decode_one(native[index]), MotionStatus.TRACKING,
            SafetyAssessment.SAFE_UNDER_MODEL,
            selected_trajectory(batch, index, observation, self.plant.codec), diagnostics)


RefDWAEngine = RefDWA
