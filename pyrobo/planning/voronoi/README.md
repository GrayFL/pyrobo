# Voronoi 点云路网与 D* Lite

`VoronoiPlanner.plan(start, goal, obstacles, bounds) -> PlanResult` 将历史项目的
`voronoi_planner/VDPlanner` 合入当前库。输入位姿统一为 `graphmap.pose.Pose`，
障碍为 `PointObstacles`，输出路径为 `PoseArray`。全局路网与 DWA 使用同一份当前点云快照，
不需要 2D 占据栅格，也不默认下采样。

```python
from pyrobo import Navigator, VoronoiConfig

nav = Navigator(planner=VoronoiConfig())
# result = nav.step(NavigationInput(...))
# 也可直接用 VoronoiPlanner(config, footprint).plan(...) 获取全局路径。
```

## 构图与搜索

1. 检查起点、终点及直达线段。碰撞使用原始点云的 XY 投影，检查实际 yaw 下的矩形，安全边距为
   `footprint.margin + clearance`；点障碍另有 `point_radius`。
2. 按 `boundary_step` 在物理 `Bounds2D` 四条边上生成虚拟点，与障碍点一起构造
   SciPy Voronoi 图。边界点用于封闭路网，实际边界碰撞仍使用解析矩形约束。
   完全重复的 XY 点在构图时去重；碰撞索引及原始 XYZ 均保留。
3. 排除无穷远边、非有限顶点和间距不足的顶点。先判断 ridge 的 `-1` 标记，
   再做顶点索引映射，避免旧实现把无穷远端误当成数组末尾。
4. 每条路网边都检查沿边方向平移的完整扫掠矩形，包括参考点纵向偏移，并验证两个方向。起终点作为真实节点，分别连接到至多
   `connection_neighbors` 个最近顶点；每条连接也必须通过相同检查。
5. 边成本为长度乘以安全间距惩罚。端点／中点采样只影响偏好，不能替代整段碰撞检查。
   D* Lite 以欧氏距离为启发式，搜索成本始终不小于欧氏距离。
6. 提取路径后检查每个转角及起终 yaw 的连续旋转矩形包络；如果转角不可行，保守禁用该出边，
   在同一预算内修复 D* Lite。下次规划先恢复这些与入射方向有关的禁边。这个处理可能舍弃
   从另一方向进入才可行的边，不是完整的 SE(2) 搜索。
7. 路径插值保留每一个拐角和真实起终点；`path_spacing` 限制输出线段长度，
   不跨越拐角做插值或平滑，以免切入障碍。

## 缓存和预算

一个实例只保存一份路网、搜索代价和对应的 XY 快照。同一地图与目标下，起点移动时
重连起点并复用 D* Lite；障碍内容、目标位置或 yaw、边界、坐标系、点半径或规划配置变化时
重建路网。障碍快照可以增加或删除点，不积累历史地图。新快照重建也允许发现新开放的通道。
`reset()` 清除全部缓存。`Navigator` 在目标 yaw 改变时也会清理状态。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `boundary_step` | 0.5 m | 虚拟边界点间距 |
| `max_edge_length` | 15 m | 路网边及端点连接最大长度，直达路径不受此限 |
| `connection_neighbors` | 32 | 每个端点最多检查的近邻数 |
| `path_spacing` | 0.15 m | 输出路径最大点间距 |
| `clearance` | 0.12 m | 矩形安全边距之外的跟踪余量 |
| `safety_distance` / `safety_weight` | 0.5 m / 1.0 | 余量不足时的软成本，不改变碰撞轮廓 |
| `time_budget` | 0.2 s | 单次规划协作式预算 |
| `max_sites` | 20000 | 输入点数加虚拟边界点数上限（去重前） |
| `max_edge_checks` | 50000 | 单次路网边和端点连接检查上限 |
| `max_expansions` | 10000 | 单次 D* 队列处理上限 |

SciPy/Qhull 构图、KD-tree 查询和单次数组运算不能中途抢占；大小上限与前后时间检查
提供软实时控制。直达路径和起终点校验不经过路网预算；预算不构成硬实时保证。
超出点数上限会显式失败，调用方可提高预算或明确选择降采样并评估覆盖损失。

| `PlanResult.status` | 含义 |
| --- | --- |
| `success` | 得到本次快照下有效的完整路径 |
| `invalid_start` / `invalid_goal` | 起终点不满足矩形间距或边界约束 |
| `no_path` | 当前有限路网没有连接，不能据此证明连续空间无解 |
| `budget_exhausted` | 时间、点数、边检查或搜索预算不足 |
| `degenerate_geometry` | Qhull 无法构造有效图 |

失败返回空路径，绝不返回上一次成功路径。`extensions` 表示本次 D* 队列处理次数；
`diagnostics` 提供 `reused_graph`、`sites`（本次构图点数）、`vertices`、`edges` 和
`edge_checks`。导航器会把这些字段放入 `NavigationOutput.diagnostics["planning"]`。

## 历史实现迁移

保留了“障碍点云 → Voronoi 路网 → 安全成本 → D* Lite”的流程，并修复了历史实现的
无穷边索引、起终点连接缺少碰撞检查、只采样边中点、旧路径残留以及重新初始化时
搜索状态未完全清理等问题。D* 队列改为带版本的堆并定期压缩，支持成本增加与降低。

旧版的自动凸包外扩改为调用方明确提供的物理边界；隐式体素降采样和控制台打印被移除。
旧 `VDPlanner.plan(obs_pcds, pos_curr, pos_goal)` 的数组／增量接口不再作为公共接口。
历史目录位置和本机验证方式见根目录 `PROJECT.md`。

该规划器使用定向矩形，支持直线窄通道，但仍不做完整 SE(2) / 汽车可达性搜索。点云表面不编码实体内部占据；
起终点必须来自有效自由空间。局部执行、最终朝向对齐和安全制动由 DWA／Navigator 完成。
