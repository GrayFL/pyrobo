# 无界面 2D 仿真与导航例程

`World2D` 提供物理边界、实心矩形障碍与点云；`Simulator2D` 使用共享 `ChassisDynamics` 保存实际 `ChassisState`。旧导入名 `RobotState` 指向同一状态类型。所有算法与 Matplotlib、ROS、Jupyter 无关。

```python
from pyrobo import Ackermann, Footprint, NavigationInput, Navigator
from pyrobo.simulation import Simulator2D, demo_world

world, start, goal = demo_world()
nav = Navigator(model=Ackermann(), footprint=Footprint.long())
sim = Simulator2D(world, start, model=nav.model, footprint=nav.footprint,
                  dynamics=nav.dynamics.config)
cloud = world.point_cloud()
for _ in range(1000):
    s = sim.state
    out = nav.step(NavigationInput(s.timestamp, s.pose, s.velocity, goal,
                                  cloud, world.bounds, dt=0.1, chassis=s))
    if out.status in ("arrived", "emergency_stop"):
        break
    sim.step(out.command, 0.1)
    if sim.collided:
        break
```

`sim.step` 接收模型原生指令，也允许 Twist2D，经 `model.from_twist` 适配。收到的是速度 / 转角设定值，实际运动由有限驱动力、阻力、力矩和运动学共同积分。`command_age` 默认为零，表示新命令；在长时间 step 中也会触发超时软停。手动与 DWA 共享同一 plant，不重置状态。

## 地图与碰撞

- `world.point_cloud(spacing=0.04)` 沿完整边界采样已知静态点云，不填充栅格或实体内部。默认点半径 0.03 m 覆盖该采样间隙。
- `world.lidar(pose, timestamp=..., beams=360, max_range=8)` 计算射线首个命中，包含遮挡；返回机器人坐标系点云，使用 `scan.transformed(pose)` 转为世界系。

例程使用完整已知地图。将局部扫描用于未知环境需要调用方维护历史地图；Navigator 不自动积累点云。

实体碰撞使用独立的分离轴定理，检查偏移矩形与实心障碍、工作空间边界。每个物理子步按中点 twist 的实际车体运动上界检查两端膨胀矩形，覆盖期间扫掠。它不调用规划器点云碰撞函数。碰撞时保留最后安全位姿、速度归零、`collided=True`，后续 step 报错。这是仿真终止语义，不是接触求解。

## 实验日志与手动接管

`run_navigation(world, start, goal, navigator=..., dt=0.1, max_steps=1000, control_input=...)` 重置导航器并通过 Supervisor 运行闭环。回调 `control_input(step_index, chassis_state, supervisor)` 调用 `set_mode`、`submit_teleop` 来切换控制权和提交期望。手动时只维护导航进度，只有当前来源的意图交给 DWA；交还 AUTO 不重置状态或路径。

旧 `command_override(step_index, chassis_state, navigation_output)` 为弃用的实验兼容接口，会发出 DeprecationWarning；旧调用仍先运行自动优化，然后由 SafetyGuard 独立检查覆盖命令，不能绕过安全检查。新代码使用 control_input。

返回 `SimulationRun`，含 `poses`、`timestamps`、实测 `velocities` / `steering`、导航 `outputs`、实际选定的 `commands`、`manual` 来源标记，以及上一执行周期末的力 / 力矩 `efforts`。每条日志在输入时间戳对齐；最后一条命令可能因到达 / 步数预算而未执行。遇到到达、紧急状态、实体碰撞或步数预算结束，务必检查 `arrived` / `collided`，预算耗尽不算到达。

```bash
python -m pyrobo.simulation --planner rrt_connect --model differential --footprint compact
python -m pyrobo.simulation --planner voronoi --model ackermann --footprint long
python -m pyrobo.simulation --model ackermann --scenario handoff --plot /tmp/handoff.png
```

`detour` 为交错墙绕行，`handoff` 在直线路径上注入三秒手动倒车，随后恢复 DWA。`summary()` 报告到达、碰撞、反向帧数、手动帧数、重规划次数及计算耗时。可选 `--plot` 保存路径与矩形轮廓、速度、转角、路径进度、力 / 力矩；核心仿真不导入绘图库。例程失败退出码为 1。
