# pyrobo

面向机器人运动控制的 Python 库：原始点云地图 → 全局路径 → 可回退的航点进度 → 模型空间 DWA → 力驱动底盘。算法与 ROS、GUI、绘图库解耦，公共位姿统一使用 `graphmap.pose.Pose` / `PoseArray`。

- **矩形车体**：任意长宽、朝向和纵向参考点偏移；提供 0.5 × 0.5 m、1 × 0.5 m 两个版型。全局规划、局部预测、路径验证和实体仿真都检查矩形。
- **RRT-Connect** 与 **Voronoi + D* Lite**：直接在点云上生成有序航点，不默认栅格化或降采样。
- **差速 / 阿克曼**：差速可原地旋转；阿克曼的偏航由车速和实际转角决定，支持倒车。
- **共享动力学**：速度伺服产生驱动力和转向力矩，考虑摩擦、平方阻力、转向回中、执行器限幅和轮端摩擦预算。DWA 预测与手动执行使用同一个 NumPy 批量积分器。
- **路径进度**：按弧长投影和前视插值，在抵达转角航点之前切换目标；保留完整路径，倒退时恢复历史段，交叉路段按进度连续性选取分支。

## 安装

需要 Python ≥ 3.11、NumPy ≥ 1.26.4、SciPy ≥ 1.17，以及提供 Pose v3 接口的 **graphmap** 源码库。先按 graphmap 项目说明安装或配置其源码路径，再运行：

```bash
python -c "from graphmap.pose import Pose, PoseArray"
python -m pip install -e .
# 开发测试及可选绘图
python -m pip install -e '.[dev,examples]'
```

不自动安装未经确认的 PyPI 同名 graphmap 包。核心不依赖 Torch、Matplotlib、scipykit；绘图例程仅在使用 `--plot` 时导入 Matplotlib。已有科研环境请先检查安装解析结果，避免更新核心科学计算库。

## 完整导航示例

```python
from pyrobo import Ackermann, DifferentialDrive, Footprint, NavigationInput, Navigator, VoronoiConfig
from pyrobo.simulation import Simulator2D, demo_world

model = Ackermann(wheelbase=0.5)  # 换为 DifferentialDrive(track=0.4) 即差速
footprint = Footprint.long()     # 或 Footprint.compact()、Footprint(0.7, 0.45)
world, start, goal = demo_world()
# XY 全局路径为汽车转弯预留跟踪余量，DWA 再验证实际运动。
clearance = 0.12 + 0.4 * model.minimum_turn_radius if isinstance(model, Ackermann) else 0.12
planner = VoronoiConfig(clearance=clearance, time_budget=2.0)
nav = Navigator(model=model, footprint=footprint, planner=planner)
sim = Simulator2D(world, start, model=model, footprint=footprint,
                  dynamics=nav.controller.plant.config)
cloud = world.point_cloud()

for _ in range(1000):
    s = sim.state
    out = nav.step(NavigationInput(
        timestamp=s.timestamp, pose=s.pose, velocity=s.velocity,
        goal=goal, obstacles=cloud, bounds=world.bounds, dt=0.1, chassis=s,
    ))
    if out.status in ("arrived", "emergency_stop"):
        break
    sim.step(out.command, 0.1)
    if sim.collided:
        break
print(out.status, sim.collided)
```

`Navigator()` 默认差速与 RRT-Connect。传 `model=Ackermann()` 时默认使用更长的预测时间、前视距离和 RRT 跟踪余量；显式传入的控制器、模型、动力学或规划配置按传入值使用。其他全局规划器也能通过 `DynamicWindow.step(..., path=...)` 与 `PathTracker` 接入。

**输出是模型指令**：差速返回 `WheelCommand(left, right, mode)`，轮速单位 m/s；阿克曼返回 `AckermannCommand(speed, steering, mode)`，单位 m/s、rad。`Twist2D` 表示实测车体速度；旧代码读取 `out.command.linear/angular` 需迁移为模型指令。发送速度话题可用 `model.from_twist(Twist2D(...))`；阿克曼会拒绝零线速度的非零偏航要求。

`ChassisState` 携带实际位姿、速度、转角、转角速度和上一步加速度。每次都传实际状态，使 DWA 预测与执行一致；阿克曼必须提供它，停车时也不能从 Twist 推断转角。差速省略时兼容只提供 Twist 的调用，假定未提供的加速度为零。手动与自主模式切换时继续使用同一个状态，不清零速度或转角。

时间戳必须严格递增；`dt` 是下一执行周期。点云为当前完整快照，默认拒绝未来时间戳及超过 0.3 s 的观测；静态地图可不带时间。目标、边界、世界系变化会清路径缓存；新障碍使剩余路径失效时重规划。`reset()` 用于重放或离散重定位。

`out.path` 保留完整路径，`out.trajectory` 包含制动尾段；每次只执行本周期指令。`diagnostics` 提供 `progress_m`、`remaining_m`、前视目标 `target`、当前 `segment`、规划状态、候选数、间距和计算耗时。

| 导航状态 | 含义 |
| --- | --- |
| `navigating` / `aligning` | 跟随路径 / 末端朝向对齐；阿克曼仍需行驶 |
| `braking` / `arrived` | 制动 / 位置、朝向和停止条件均满足 |
| `no_path` | 暂无全局路径，返回制动指令；详见 `planner_status` |
| `blocked` | 局部候选不可行，返回经检查的制动指令 |
| `emergency_stop` | 预测制动碰撞或不能在预算内停住，`safe=False` |

## 碰撞与坐标约定

右手系 x 向前、y 向左、z 向上，米 / 秒 / 弧度。`pose2d` 使用弧度；直接构造 graphmap 欧拉角时务必指定 `degrees=False`。导航位姿要求 z=0、零 roll/pitch、scale=1 和显式世界系；传感器外参允许完整 3D 变换。

`Footprint(length, width, margin=0.08, offset_x=0.0)` 定义车体坐标系中的矩形，`offset_x` 是车体中心相对运动参考点的纵向偏移。差速参考点为轮轴中点；阿克曼为后轴中点，如车体中心在后轴前 0.25 m，可用 `Footprint.long(offset_x=0.25)`。默认几何中心与参考点重合。

`PointCloud` 接收 Nx2 / Nx3，保留原始 XYZ。碰撞使用 XY 投影和 KD-tree，可通过 `z_range` 筛选高度。`cloud.transformed(T_world_sensor)` 变换传感器点云；`select_voxels` 是显式选择原点的降采样，丢点需重新评估覆盖。

直线上的矩形平移使用完整扫掠矩形，转角使用带连续区间覆盖的旋转矩形；外接圆只用于邻域查询和运动距离上界。点半径默认 0.03 m，安全边距与全局跟踪余量另外计入。示例表面间距 0.04 m。

## 运行例程

```bash
python -m pyrobo.simulation
python -m pyrobo.simulation --planner voronoi --model ackermann --footprint long
python -m pyrobo.simulation --planner rrt_connect --model ackermann --footprint compact
python -m pyrobo.simulation --model differential --scenario handoff
python -m pyrobo.simulation --model ackermann --scenario handoff --plot /tmp/pyrobo-handoff.png
python -m examples.manual_control
python -m pytest -q
```

`detour` 展示交错墙绕行、两种全局规划器及两种车体。`handoff` 在导航中插入三秒手动倒车，再交回 DWA，展示路径进度回退；`manual_control` 展示速度话题适配、滑行、倒车和急停。`--plot` 保存路径、矩形车体、实测速度、转角、进度和力 / 力矩图。失败例程以非零退出码结束。

进一步阅读：[运动学与动力学](pyrobo/README.md)、[DWA 与路径跟踪](pyrobo/control/dwa/README.md)、[RRT-Connect](pyrobo/planning/rrt_connect/README.md)、[Voronoi / D* Lite](pyrobo/planning/voronoi/README.md)、[仿真](pyrobo/simulation/README.md)、[例程](examples/README.md)、[开发上下文](PROJECT.md)。

## 范围

全局算法搜索 XY 折线，验证矩形平移与转角碰撞，但不求解完整 SE(2) / 汽车可达路径；阿克曼末端预留直线入射段，无法容纳时返回 `invalid_goal_approach`。复杂停车、极窄弯道或局部极值仍可能需要更合适的全局规划器或重定位，预算失败不能证明连续空间无路。DWA 所有实际候选均受所选模型约束。

动力学面向平地低速虚拟底盘，不模拟轮胎侧滑、载荷转移或接触力学；摩擦圆限制当前状态的执行器预算，并报告参考包络越界，不能保证瞬态实际横向加速度永远不越界。`safe=True` 表示当前点云和配置模型下通过预测检查。点云需覆盖障碍表面，表面点不编码实体内部；未知空间、SLAM、动态障碍预测和硬件延迟由上层处理。预算为软实时，Qhull、单次几何运算和 Python 调度不能硬抢占。
