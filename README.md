# pyrobo

面向机器人控制算法的 Python 工具库。算法与 ROS、GUI、仿真前端解耦，以“当前时刻输入 → 当前时刻输出”为接口组织。首版提供连续空间寻路和点云避障，支持默认 **0.8 m × 0.8 m** 的差速 / 单轮车模型。

- **RRT-Connect**：有计算预算的双向采样寻路，输出连续空间折线路径。
- **Voronoi + D* Lite**：从历史项目迁入的点云路网规划，校验全线段和起终点连接，支持复用路网搜索。
- **DWA**：融合历史算法的弧长前视、朝向限速和转向滞回，在可达速度窗口内选择速度，检查实际矩形车体、运动区间和制动尾段。
- **Navigator**：组合全局寻路与局部控制，缓存并逐帧校验路径，支持目标变化、障碍变化和停滞后的重规划。
- **Simulator2D**：加速度受限运动、独立实体碰撞检查、理想 2D 激光雷达与已知点云地图。

## 安装

需要 Python ≥ 3.11、NumPy ≥ 1.26.4、SciPy ≥ 1.17，以及提供 `Pose` / `PoseArray` 接口的 **graphmap.pose v3**。请先按 graphmap 项目的安装说明配置该库，并确认：

```bash
python -c "from graphmap.pose import Pose, PoseArray"
python -m pip install -e .
```

`graphmap` 由使用方提供，未在依赖中填写未经确认的 PyPI 同名发行包。开发环境已通过源码路径提供 graphmap 时，无需重复安装。核心算法不依赖 Torch、Matplotlib 或 scipykit。

运行 Notebook 还需自行配置 **scipykit**（`from scipykit.task.cv import *`），以及：

```bash
python -m pip install -e '.[dev,examples]'
python -c "from scipykit.task.cv import *"
```

已有科研环境安装前请检查依赖解析结果，避免改变核心科学计算库版本。

## 实时接口

```python
import numpy as np
from graphmap.pose import Pose
from pyrobo import Bounds2D, NavigationInput, Navigator, PointCloud, Twist2D, VoronoiConfig

robot = Pose(translation=[0, 0, 0], rotation=[0, 0, 0], degrees=False,
             wrd_frame="world", ego_frame="base")
goal = Pose(translation=[3, 1, 0], rotation=[0, 0, 0], degrees=False,
            wrd_frame="world", ego_frame="goal")
cloud = PointCloud(np.array([[1.5, 0.0], [1.5, 0.05]]), frame="world", timestamp=0.0)
navigator = Navigator(planner=VoronoiConfig())  # 点云 Voronoi + DWA

result = navigator.step(NavigationInput(
    timestamp=0.0,
    pose=robot,
    velocity=Twist2D(0.0, 0.0),  # 当前实测速度
    goal=goal,
    obstacles=cloud,
    bounds=Bounds2D(-2, 5, -3, 4),
    dt=0.1,                    # 下一控制周期，不是计算耗时
))
print(result.status, result.command, result.safe)
```

每帧传入新的当前观测；`timestamp` 必须严格递增，输出保留相同时间戳。`command.linear` 是前向速度设定值（m/s），`command.angular` 是偏航角速度设定值（rad/s）。执行器应按配置的加速度限幅跟踪，不能把输出当作瞬时速度跳变。重复实验前调用 `navigator.reset()`。

`result.path`、`result.trajectory` 均为 **graphmap.pose.PoseArray**；前者是参考路径，后者含制动尾段。`diagnostics` 提供规划状态、耗时、候选数量和保守间距；`planner` 标识所选算法，`planning` 包含本次全局搜索信息，`controller_mode` 区分跟随、转向和转向制动。算法最多保留一个路网、当前路径、转向／时间／进度信息和随机数状态，不自动积累点云。

`Navigator()` 保持原来的 RRT-Connect 默认行为，也可显式传入 `RRTConnectConfig(...)`。传入 `VoronoiConfig(...)` 即使用历史算法整合后的完整导航链。两者均使用同一 `NavigationInput`／`NavigationOutput`，可通过 `Footprint` 和 `DWAConfig` 配置车体及运动能力。单独规划可使用 `VoronoiPlanner.plan(...)`，单独局部控制可使用 `DynamicWindow.step(...)`。

| 状态 | 含义 |
| --- | --- |
| `navigating` / `aligning` | 沿路径行驶 / 在目标位置对齐朝向 |
| `braking` / `arrived` | 在目标处减速 / 位置、朝向和停止条件均满足 |
| `no_path` | 未得到可用路径（路网不连通、预算不足、退化几何或起终点间距不足）；返回制动指令，后续可重试，具体原因见 `planner_status` |
| `blocked` | 局部候选不可行，返回经检查的制动指令 |
| `emergency_stop` | 当前模型下连制动轨迹也无法通过检查；`safe=False`，由上层处理 |

## 坐标系与点云

- 右手系：**x 向前、y 向左、z 向上**；单位米、秒、弧度，正偏航绕 +z 逆时针。
- 公共位姿统一使用 graphmap，不另建位姿类型。`Pose` 原生欧拉角默认是角度；本库构造时明确使用 `degrees=False`。辅助函数 `pose2d(x, y, yaw)` 接收弧度。
- 机器人与目标为 `z=0`、零 roll/pitch、`scale=1` 的平面位姿，且世界坐标系标签必须与点云一致。
- `PointCloud` 接收 `(N,2)` 或 `(N,3)`；保留原始 XYZ 数据。几何查询使用 XY 投影和 KD-tree，默认**不栅格化、不降采样**。
- 传感器点云通过 `cloud.transformed(T_world_sensor)` 转换；支持带真实 3D 外参的传感器。可以给 `Navigator(z_range=(zmin, zmax))` 在世界系筛选高度，原始点云仍保留。默认所有高度都参与投影，请自行剔除地面。
- `obstacles` 是当前完整障碍快照，不是点云增量。实时观测设置采集时间；默认拒绝超过 0.3 s 的观测。已知静态地图可用 `timestamp=None`。
- 可选 `cloud.select_voxels(size)` 返回每个体素中的一个原始点，源点云不变。它不是占据栅格，且丢点可能影响安全覆盖，须相应增大 `point_radius` 或继续用原云碰撞检测。

## 示例与验证

[examples/navigation_2d.ipynb](examples/navigation_2d.ipynb) 本地示例包含 Voronoi＋DWA 绕墙闭环、U 形障碍、scipykit 绘图、速度/耗时/间距指标、时间滑块和 HTML 动画（Notebook 与输出按本仓库约定忽略）。重新运行时请选择具备上述依赖的内核；切换配置可比较 RRT-Connect。

```bash
python -m pyrobo.simulation                    # RRT-Connect + DWA
python -m pyrobo.simulation --planner voronoi   # Voronoi + D* Lite + DWA
python -m pytest -q                            # 几何、控制、重规划和两种算法的闭环测试
```

进一步阅读：[RRT-Connect](pyrobo/planning/rrt_connect/README.md)、[Voronoi / D* Lite](pyrobo/planning/voronoi/README.md)、[局部控制](pyrobo/control/dwa/README.md)、[仿真器](pyrobo/simulation/README.md)、[本地开发上下文](PROJECT.md)。

## 适用范围

这是面向算法开发的**软实时**实现：RRT 有扩展次数和协作式时间预算，Voronoi 有点数、边检查、搜索次数和时间预算，DWA 有固定采样规模；Qhull 构图不能中途抢占，Python 调度、输入点数和单次几何运算仍会影响耗时，不能承诺硬实时截止期限。

默认规划适用于已知或由外部维护的静态点云地图。没有点不等于已知自由空间；只有当前激光扫描时，未观测区域、遮挡及历史地图的处理需由调用方负责。点云必须足够覆盖障碍表面，并设置合理的点半径（默认 0.03 m）。实体内部的占据语义、动态物体速度预测、SLAM、重定位、倒车、汽车转向约束和硬件急停不在首版范围内。

全局规划使用车体外接圆，可能拒绝实际矩形能通过的窄通道；RRT-Connect 不保证最短路径，Voronoi 在有限路网上搜索，两者失败都不能证明连续空间无路。局部控制仍可能陷入局部停滞。`safe=True` 仅表示在**当前点云、运动模型及配置**下通过了预测检查，不代表未来环境不变。

算法依据：[Kuffner & LaValle 的 RRT-Connect 论文与官方实现说明](https://ompl.kavrakilab.org/core/classompl_1_1geometric_1_1RRTConnect.html)、[Fox 等人的 DWA 论文](https://publications.ri.cmu.edu/the-dynamic-window-approach-to-collision-avoidance)。Voronoi／D* Lite 与 DWA 的历史迁移范围、接口变化和几何修正见对应算法 README。

