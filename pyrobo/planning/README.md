# 全局规划与间距适配

`RRTConnect` 和 `VoronoiPlanner` 实现 `GlobalPlanner`，接收 Pose、原始点云索引与物理边界，返回 PlanResult。两者搜索 XY 路线并验证矩形扫掠及转角，不保证非完整约束下的汽车可达性。

具体规划器的 `config.clearance` 始终是硬约束，叠加在车体 margin 和点半径之外。若需要“空旷处留宽、窄道允许靠近”的策略，显式使用组合：

```python
from pyrobo import Footprint, Navigator
from pyrobo.planning import AdaptiveClearancePlanner, RRTConnect, RRTConnectConfig

body = Footprint(.5, .5, margin=.08)
planner = AdaptiveClearancePlanner(RRTConnect,
    RRTConnectConfig(clearance=.4, time_budget=1.), body, minimum_clearance=0.)
nav = Navigator(footprint=body, global_planner=planner)
```

此时 `.4 m` 是首选额外间距，`minimum_clearance` 才是额外间距的硬下限。初始尝试先按起终点可用空间限制首选值，失败再尝试中间值和硬下限；三次尝试共用原时间预算，不缩小实际长宽、offset、margin 或点半径。调用方显式设置的硬下限不会被降低。诊断保留 preferred / effective / minimum_clearance 及每次尝试状态。

缓存最多四个不同间距的规划器实例，reset 会清除。某次不可抢占计算仍可能超出软预算。降低额外间距只能释放真实可通行的空间，不能为过大的车体创造通道。

Web / Pygame 和 RSim 的配置默认启用该组合；可关闭 `backend.adaptive_clearance` / `StackConfig.adaptive_clearance` 恢复固定间距。直接传入具体规划器配置的既有库 API 保持硬间距语义。

阿克曼的 `NavigationConfig.goal_approach_distance=None` 根据车型和前视先选择入射长度，再按矩形可用空间缩短；正数固定长度，0 关闭。自动模式已经接近终点时可直接重新连接目标，避免回头追逐入射起点。所有连接仍受同样的碰撞检查；这项几何适配不能替代 SE(2) / 曲率约束规划。

算法细节：[RRT-Connect](rrt_connect/README.md)、[Voronoi / D* Lite](voronoi/README.md)。实际成功率和失败样本见[报告](../../reports/navigation/README.md)。
