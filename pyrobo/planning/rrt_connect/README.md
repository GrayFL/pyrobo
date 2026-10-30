# RRT-Connect：点云上的连续空间寻路

`RRTConnect.plan(start, goal, obstacles, bounds) -> PlanResult` 接收当前起终点 `Pose`、`PointObstacles` 和物理边界。输出 `PoseArray` 路径、状态、扩展次数和耗时；不需要历史观测。`Navigator.step` 为它提供统一时间戳接口和路径缓存。

## 算法

先检查起终点以及直达线段；直达失败时从两端生长随机树。交替扩展一棵树，再沿新节点反复连接另一棵树。找到连接后做随机快捷连接，最后按指定间距插值。插值只是路径输出密度，不是占据栅格。

默认启用障碍表面的桥接采样：部分非目标采样尝试取邻近墙面点之间的中点，引导树进入窄通道。矩形姿态先检查候选中点，每条实际边仍检查平移和旋转；保留均匀采样，未生成有效中点时回退。它不是通道连通性证明，也不保证每种地图都更快。可用 `bridge_sampling=False` 做同预算对照。

全局碰撞轮廓与 DWA 共用 `Footprint` 的实际矩形，并额外加入 `config.clearance` 跟踪余量。直线平移的扫掠体是加长的定向矩形，按原始点到矩形的距离和矩形边界检查；不是把矩形替换成外接圆。转角按旋转矩形的区间运动上界自适应细分，未能证明安全的区间保守拒绝。

起终点按实际 yaw 验证。两棵树记录各自入射 / 出射朝向，扩展检查父节点的旋转；连接、快捷连接和输出完整路径均检查转角，不能通过删掉航点切进障碍。非对称参考点偏移的边检查两个方向。直线窄通道不再因车长的外接圆半径被直接排除。

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
| `time_budget` | 0.045 s | 每次扩展/快捷连接前检查的协作式预算；`None` 只使用次数预算 |
| `goal_bias` | 0.15 | 朝另一棵树根节点采样的概率 |
| `shortcut_attempts` | 60 | 快捷连接次数上限，与搜索共享时间预算 |
| `path_spacing` | 0.15 m | 输出相邻路径点的最大距离 |
| `clearance` | 0.12 m | 在矩形安全边距之外的跟踪余量 |
| `seed` | 7 | 每个实例的随机数种子 |
| `bridge_sampling` | True | 点云表面桥接偏置，保留均匀采样 |

树最近邻使用 NumPy 精确扫描，适合预算内的小树；障碍点查询使用 SciPy KD-tree。每次调用新建树，失败重试会继续随机数序列；`reset()` 恢复种子。相同种子且不触发时间截止时可复现，真实计时截止可能导致不同路径。

确定性运行显式配置 `time_budget=None`、整数 `seed`、`max_extensions` 和
`shortcut_attempts`。此模式的搜索和快捷连接只按逻辑次数停止；真实 wall 耗时仍记录为
诊断，不参与路径选择。重复实验应以相同输入调用 `reset()`，恢复随机数状态。
默认 `time_budget=.045` 的行为不变。`AdaptiveClearancePlanner` 目前要求 wall 时间预算，
传入 `time_budget=None` 会明确拒绝；锁步装配使用 `adaptive_clearance=False` 的固定余量规划。

`status` 为 `success`、`invalid_start`、`invalid_goal` 或 `budget_exhausted`。失败返回空 `PoseArray`；预算耗尽不等于问题不可解。路径点的 yaw 仅是相邻线段切线方向，最终目标朝向由 DWA 单独控制。该规划器搜索 XY，验证了矩形的直线与旋转包络，但不是完整 SE(2) 或动力学规划器。旋转包络通过不代表阿克曼可以原地旋转；Navigator 预留终点入射段，DWA 再验证汽车实际候选。汽车通常需要更大的跟踪余量，例程按最小转弯半径配置。

时间预算不是硬截止：初始检查、一次树扩展、输出整理和 Python 调度仍有开销。复杂任务可增大预算或在上层异步运行全局规划。狭窄通道可能需要面向矩形 SE(2) 的后续规划器，不能通过缩小真实车体来绕过几何约束。

参考：[RRT-Connect 官方文档及原论文 DOI](https://ompl.kavrakilab.org/core/classompl_1_1geometric_1_1RRTConnect.html)、[OMPL 规划终止与可行性说明](https://ompl.kavrakilab.org/planners.html)。未引入 OMPL 二进制依赖，代码为针对本库接口的独立实现。

桥接采样思路参考 [Bridge Test](https://adacomp.org/wp-content/uploads/2016/01/icra03.pdf)，这里按原始点云表面及矩形方向实现，不使用占据栅格。`PlanResult.diagnostics['bridge_samples']` 记录有效偏置样本数；重复试验见[导航基准](../../../examples/navigation_benchmark/README.md)。

