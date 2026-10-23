# 轻量 2D 仿真

`World2D` 保存物理矩形边界和轴对齐实体障碍；`Simulator2D` 保存当前 `RobotState(timestamp, pose, velocity)`，接受速度设定值并按加速度约束推进。所有算法接口与 Matplotlib、Jupyter、ROS 无关。

```python
from pyrobo import NavigationInput, Navigator
from pyrobo.simulation import Simulator2D, demo_world

world, start, goal = demo_world()
sim, nav = Simulator2D(world, start), Navigator()
known_map = world.point_cloud(spacing=0.04)
for _ in range(1000):
    s = sim.state
    result = nav.step(NavigationInput(s.timestamp, s.pose, s.velocity,
                                     goal, known_map, world.bounds, dt=0.1))
    if result.status in ("arrived", "emergency_stop"):
        break
    sim.step(result.command, dt=0.1)
    if sim.collided:
        break
```

## 两类点云

- `world.point_cloud(spacing)`：按线段长度采样完整障碍边界，是**已知静态地图**。没有栅格，不填充障碍内部。默认 0.04 m 间距配合算法的 0.03 m 点半径覆盖相邻采样间隙。
- `world.lidar(pose, timestamp=..., beams=360, max_range=8)`：射线与线段求交，仅保留最近有效回波，包含遮挡，返回机器人坐标系点云。`scan.transformed(pose)` 转到世界系。无回波不生成最大量程伪障碍。

Notebook 主闭环使用完整已知地图；传感器视图展示局部激光观测，明确区分两者。将局部扫描用于未知环境需要额外的点云记忆/地图模块，当前 Navigator 不会把每帧扫描自动融合。添加动态障碍时，应更新完整快照和仿真实体几何；当前算法只对新快照重新规划，不预测物体运动。

实体碰撞使用分离轴定理，检查车体矩形与实心障碍以及物理边界，**不复用规划器的点云碰撞结果**。运动细分为至多 0.02 s，并检查带区间运动余量的两端车体。触发碰撞后设置 `collided=True`、保留最后一个安全位姿、终止运动，后续 step 报错；它是碰撞终止模型，不是接触力学仿真。

`run_navigation(world, start, goal, navigator=..., dt=..., max_steps=...)` 提供无界面实验：重置导航器、保存与输入时间戳对齐的状态/决策，遇到到达、紧急状态、碰撞或步数上限结束。返回 `SimulationRun`，可访问 `poses`、`timestamps`、`velocities`、`outputs` 和 `summary()`。状态日志截止到最后一次控制决策，碰撞步之后的中间位姿不追加为新的控制帧。`max_steps` 耗尽时 `arrived=False`，调用方应检查结果。

运行 `python -m pyrobo.simulation` 使用 RRT-Connect，`python -m pyrobo.simulation --planner voronoi` 使用 Voronoi／D* Lite；两者均以 DWA 完成闭环，输出结果与本机延迟分位数，失败时退出码为 1，不导入绘图库。也可传入 `Navigator(planner=VoronoiConfig(...))` 自定义预算。绘图和交互回放见本地 [Notebook](../../examples/navigation_2d.ipynb)，通过 scipykit 完成。

