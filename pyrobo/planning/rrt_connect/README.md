# RRT-Connect：点云上的连续空间寻路

`RRTConnect.plan(start, goal, obstacles, bounds) -> PlanResult` 接收当前起终点 `Pose`、`PointObstacles` 和物理边界。输出 `PoseArray` 路径、状态、扩展次数和耗时；不需要历史观测。`Navigator.step` 为它提供统一时间戳接口和路径缓存。

## 算法

先检查起终点以及直达线段；直达失败时从两端生长随机树。交替扩展一棵树，再沿新节点反复连接另一棵树。找到连接后做随机快捷连接，最后按指定间距插值。插值只是路径输出密度，不是占据栅格。

全局碰撞轮廓为 `hypot(length/2, width/2) + footprint.margin + config.clearance` 的外接圆，障碍点另带 `point_radius`。KD-tree 选取线段包围圆内的原始障碍点，再计算精确的线段到点距离；不以离散的边采样代替碰撞检测。工作空间边界按车体半径内缩，线段端点在内缩的矩形内即可保证整段不越界。

```python
from pyrobo import PointObstacles
from pyrobo.planning import RRTConnect, RRTConnectConfig
from pyrobo.simulation import demo_world

world, start, goal = demo_world()
planner = RRTConnect(RRTConnectConfig(seed=7, time_budget=0.1))
result = planner.plan(start, goal, PointObstacles(world.point_cloud()), world.bounds)
print(result.status, result.path.position)
```

## 参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `step_size` | 0.65 m | 每次扩展最大长度 |
| `max_extensions` | 1800 | 包括失败尝试和 connect 内循环的扩展上限 |
| `time_budget` | 0.045 s | 每次扩展/快捷连接前检查的协作式预算 |
| `goal_bias` | 0.15 | 朝另一棵树根节点采样的概率 |
| `shortcut_attempts` | 60 | 快捷连接次数上限，与搜索共享时间预算 |
| `path_spacing` | 0.15 m | 输出相邻路径点的最大距离 |
| `clearance` | 0.12 m | 在车体外接圆安全边距之外的跟踪余量 |
| `seed` | 7 | 每个实例的随机数种子 |

树最近邻使用 NumPy 精确扫描，适合预算内的小树；障碍点查询使用 SciPy KD-tree。每次调用新建树，失败重试会继续随机数序列；`reset()` 恢复种子。相同种子且不触发时间截止时可复现，真实计时截止可能导致不同路径。

`status` 为 `success`、`invalid_start`、`invalid_goal` 或 `budget_exhausted`。失败返回空 `PoseArray`；预算耗尽不等于问题不可解。路径点的 yaw 仅是相邻线段切线方向，最终目标朝向由 DWA 单独控制。该规划器搜索 XY，依靠外接圆允许原地转向，不是完整 SE(2) 或动力学规划器。

时间预算不是硬截止：初始检查、一次树扩展、输出整理和 Python 调度仍有开销。复杂任务可增大预算或在上层异步运行全局规划。狭窄通道可能需要面向矩形 SE(2) 的后续规划器，不能通过缩小真实车体来绕过几何约束。

参考：[RRT-Connect 官方文档及原论文 DOI](https://ompl.kavrakilab.org/core/classompl_1_1geometric_1_1RRTConnect.html)、[OMPL 规划终止与可行性说明](https://ompl.kavrakilab.org/planners.html)。未引入 OMPL 二进制依赖，代码为针对本库接口的独立实现。

