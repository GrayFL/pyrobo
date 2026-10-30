# pyrobo

面向机器人运动控制的 Python 库：全局导航 / 手动意图 → 控制权管理 → 模型空间 DWA → 独立安全门 → 力驱动底盘。算法与 ROS、GUI、绘图库解耦，公共位姿统一使用 `graphmap.pose.Pose` / `PoseArray`。

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
                  dynamics=nav.dynamics.config)
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

`Navigator()` 默认差速与 RRT-Connect。传 `model=Ackermann()` 时默认使用更长的预测时间、前视距离和 RRT 跟踪余量；显式传入的控制器、模型、动力学或规划配置按传入值使用。可以通过 `global_planner=` 和 `local_optimizer=` 注入其他算法；旧 `DynamicWindow.step(..., path=...)` 入口仍可使用。

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

## 手动辅助与控制权

```python
from pyrobo import ControlMode, MotionPolicy, RobotObservation, Supervisor, Twist2D

runtime = Supervisor(nav)
runtime.set_goal(goal, sim.state.timestamp)
runtime.set_mode(ControlMode.MANUAL)
runtime.submit_teleop(Twist2D(.5, 0.), sim.state.timestamp,
                      policy=MotionPolicy.TELEOP_AVOID)
result = runtime.step(RobotObservation(sim.state, cloud, world.bounds, dt=.1))
sim.step(result.command, .1)
runtime.set_mode(ControlMode.AUTO)
```

`TELEOP_STOP` 只缩小纵向参考，保持差速角速度或阿克曼转角；`TELEOP_AVOID` 允许有界转向偏离，两者都禁止擅自加速或反向。默认手动请求 0.3 s 超时即制动。PAUSED 保留任务；ESTOP 锁存，必须显式 `release_estop()`，解除后进入 PAUSED。

手动期间只维护导航进度，不额外计算自动 DWA。完整底盘状态和历史路径跨模式保留。最终安全门独立复核实际下发命令的一周期运动和制动；无解、陈旧输入、模型/适配故障均返回停止原因与安全评估。`MotionResult.trajectory` 含相对时间、绝对时戳和制动尾段；兼容的 `NavigationOutput` 仍提供 PoseArray。

## 碰撞与坐标约定

右手系 x 向前、y 向左、z 向上，米 / 秒 / 弧度。`pose2d` 使用弧度；直接构造 graphmap 欧拉角时务必指定 `degrees=False`。导航位姿要求 z=0、零 roll/pitch、scale=1 和显式世界系；传感器外参允许完整 3D 变换。

`Footprint(length, width, margin=0.08, offset_x=0.0)` 定义车体坐标系中的矩形，`offset_x` 是车体中心相对运动参考点的纵向偏移。差速参考点为轮轴中点；阿克曼为后轴中点，如车体中心在后轴前 0.25 m，可用 `Footprint.long(offset_x=0.25)`。默认几何中心与参考点重合。

`PointCloud` 接收 Nx2 / Nx3，保留原始 XYZ。碰撞使用 XY 投影和 KD-tree，可通过 `z_range` 筛选高度。`cloud.transformed(T_world_sensor)` 变换传感器点云；`select_voxels` 是显式选择原点的降采样，丢点需重新评估覆盖。

直线上的矩形平移使用完整扫掠矩形，转角使用带连续区间覆盖的旋转矩形；外接圆只用于邻域查询和运动距离上界。点半径默认 0.03 m，安全边距与全局跟踪余量另外计入。示例表面间距 0.04 m。

## Pygame 交互测试

在仓库根目录运行，可通过 WASD、屏幕摇杆或 USB 手柄驾驶；点击地图设定导航目标，拖动指定到达朝向。支持切换全局算法、车型与矩形尺寸，并显示路径、预测轨迹和安全状态。

```bash
python -m pip install -e '.[interactive]'
python -m pyrobo.apps.pygame_sim --config pyrobo/apps/configs/pygame.toml
python -m pyrobo.apps.pygame_sim --planner voronoi --model ackermann --set footprint.length=1.0
```

G 恢复自动导航，Space 暂停，E 急停，R 解除到暂停，Home 重置；手动输入默认停障辅助，按 2 切换辅助绕障。详见[交互测试台说明](pyrobo/apps/pygame_sim/README.md)及[配置模板](pyrobo/apps/configs/pygame.toml)。pygame 为可选依赖，不进入核心库。

## Web 地图编辑与局域网测试

```bash
python -m pip install -e '.[web]'
python -m pyrobo.apps.web_sim --host 0.0.0.0 --port 8000
```

本机打开 `http://127.0.0.1:8000`，其他设备使用服务器的局域网 IP。支持矩形、多边形、自由画墙与独立折线墙，任意摆放车辆和目标、切换算法 / 车型 / 参数，以及键盘、屏幕摇杆和点击导航。每个窗口独立运行，场景库跨设备共享；完整场景可保存为 JSON，并作为无界面回归用例重放。

```bash
python -m pyrobo.apps.web_sim.replay pyrobo/apps/scenes/polygon_corridor.json --steps 700
```

详见 [Web 运动实验室说明](pyrobo/apps/web_sim/README.md)。页面无需 Node 构建或外部 CDN，FastAPI / Uvicorn 不进入核心算法依赖。

交互例程默认将全局额外间距作为首选值自动适配，保留矩形、碰撞 margin 和点半径；页面可切换为固定硬间距。阿克曼终点入射距离留空时根据可用空间自动选择。可用[窄道回归场景](pyrobo/apps/scenes/narrow_ackermann_adaptive.json)测试两个全局规划器，算法语义见[规划组合说明](pyrobo/planning/README.md)。

## 多进程导航与延迟测试

可选的 `pyrobo.integrations.rsim.NavigationStack` 使用真实 RSim 将全局规划、DWA、执行检查分到独立进程；传递模型原生命令和完整状态，拒绝过期结果，并在执行端单独复核有效期及制动。仿真物理持续前进，可以注入计算卡顿、里程计和地图中断。

```bash
python -m benchmarks.navigation.rsim_platform --scenario narrow_ackermann \
  --model ackermann --planner rrt_connect --control-delay-every 8 --control-delay .4 \
  --output reports/narrow-delayed.json
python -m benchmarks.navigation.planning --repeats 20 --budget .5 \
  --output reports/global.json
```

安装和端口契约见 [RSim 集成](pyrobo/rsim/README.md)，复现实验见 [导航测试平台](benchmarks/navigation/README.md)；[实测报告](reports/navigation/README.md)同时保留成功和失败用例。

## 运行例程

```bash
python -m pyrobo.simulation
python -m pyrobo.simulation --planner voronoi --model ackermann --footprint long
python -m pyrobo.simulation --planner rrt_connect --model ackermann --footprint compact
python -m pyrobo.simulation --model differential --scenario handoff
python -m pyrobo.simulation --model ackermann --scenario handoff --plot /tmp/pyrobo-handoff.png
python -m examples.manual_control
python -m examples.assisted_teleop --model both
python -m pytest -q
```

`detour` 展示交错墙绕行、两种全局规划器及两种车体。`handoff` 在导航中插入三秒手动倒车，再交回 DWA，展示路径进度回退；`manual_control` 展示速度话题适配、滑行、倒车和急停。`--plot` 保存路径、矩形车体、实测速度、转角、进度和力 / 力矩图。失败例程以非零退出码结束。

`assisted_teleop` 展示两种车型的无障碍、严格停障、辅助绕障和无可行制动场景。正式手动控制使用 `Supervisor.submit_teleop` 发送期望，由共享 DWA 求解；`run_navigation(control_input=...)` 可编排接管过程。旧 `command_override` 已弃用，仍受最终安全门约束。

进一步阅读：[架构与 API 迁移](docs/architecture.md)、[性能记录](benchmarks/README.md)、[运动学与动力学](pyrobo/README.md)、[DWA 与路径跟踪](pyrobo/control/dwa/README.md)、[RRT-Connect](pyrobo/planning/rrt_connect/README.md)、[Voronoi / D* Lite](pyrobo/planning/voronoi/README.md)、[仿真](pyrobo/simulation/README.md)、[例程](examples/README.md)、[开发上下文](PROJECT.md)。

## 范围

全局算法搜索 XY 折线，验证矩形平移与转角碰撞，但不求解完整 SE(2) / 汽车可达路径；阿克曼末端预留直线入射段，可通过 `NavigationConfig.goal_approach_distance` 单独设置长度，无法容纳时返回 `invalid_goal_approach`。复杂停车、极窄弯道或局部极值仍可能需要更合适的全局规划器或重定位，预算失败不能证明连续空间无路。DWA 所有实际候选均受所选模型约束。

动力学面向平地低速虚拟底盘，不模拟轮胎侧滑、载荷转移或接触力学；摩擦圆限制当前状态的执行器预算，并报告参考包络越界，不能保证瞬态实际横向加速度永远不越界。`safe=True` 表示当前点云和配置模型下通过预测检查。点云需覆盖障碍表面，表面点不编码实体内部；未知空间、SLAM、动态障碍预测和硬件延迟由上层处理。预算为软实时，Qhull、单次几何运算和 Python 调度不能硬抢占。


## 可选 RSim 设备与跨提供端导航

公开运行时路径为 `pyrobo.rsim`，包含 `devices`（连接）、`drivers`（工厂）、
`components`（执行）和 `assemblies`（组合）。需要真实 RSim >=0.4；
纯算法和内存物理模型不依赖 RSim。`pyrobo.integrations.rsim` 在 0.1 过渡期
委托同一实现，计划 0.2 退役。

`MemoryChassisFactory.build()` 只构造未运行组件，`shared()` 返回启动配方，
`device()` 只连接。完整端口只在内存工厂本地定义，公开接口族是
`chassis-ports-v1`；可用 `rsim.devices.ChassisPortsDevice` 消费本库或
MujocoSim 提供端。默认不允许运动，仿真运动也须明确 `motion_enabled=True`。

`NavigationInputs` 显式转换定位和测量状态；`NavigationCommandOutput` 保留
命令身份、主机期限及可选的固定源时间终点。`StackConfig.clock` 区分源环境时间与主机安全时间，
`StackConfig.scheduling` 独立选择默认 `realtime` 或显式 `environment` 调度。
同一导航组合支持本地、线程和进程执行，包装层没有引擎选择分支。

```bash
python -m pyrobo.apps.navigation_provider --key demo --clock simulation \
  --execution thread --scheduling environment --duration 20 --output navigation.json
```

该消费程序连接已经启动的设备；无需安装 MuJoCo 或 MujocoSim。默认示例
使用固定预建空房地图、明确的平面定位投影及相同操作限速，不是 SLAM 验收。
安装边界、字段、停止模式降级、reset 与兼容要求见
[运行时设备与导航](pyrobo/rsim/README.md)。

内存提供端显式使用 `MemoryChassisFactory(scheduling='accelerated', speedup=2)`
时，物理步长固定，倍率只调整主机节奏；`pause()` 退役命令，`resume()` 和有效倍率变化
进入新 episode 并保留物理时间，只有 `reset()` 将时间归零。环境导航默认读取最新源帧，
重复帧不计算，实际消费间隔作为控制 dt；计算过载会跳帧或安全停止。倍率是配置目标，
报告中的实测 RTF 才是实际吞吐，不能将配置 10 倍等同于实现 10 倍闭环。

完整交互工具现位于 `pyrobo.apps.pygame_sim`、`pyrobo.apps.web_sim`，
配置、Web 静态资源及小型回归场景随包管理。旧 examples 入口只做兼容委托；
导航基准工具位于 `benchmarks/navigation`。无须从仓库根目录启动包内应用。
