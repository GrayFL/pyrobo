# Pygame 机器人测试台

在一个窗口中测试全局规划、DWA、运动学与力驱动动力学。支持 RRT-Connect / Voronoi + D* Lite、差速 / 阿克曼、任意矩形车体，以及手动停障 / 辅助绕障。窗口显示实际车体、完整全局路径、局部预测（含制动尾段）、行驶轨迹、前视目标与控制诊断。

## 启动

在仓库根目录运行；graphmap 的准备方式见[根 README](../../README.md)。pygame 是可选依赖，核心库不导入它。

```bash
python -m pip install -e '.[interactive]'
python -m pyrobo.apps.pygame_sim
python -m pyrobo.apps.pygame_sim --config pyrobo/apps/configs/pygame.toml
python -m pyrobo.apps.pygame_sim --planner voronoi --model ackermann --set footprint.length=1.0
python -m pyrobo.apps.pygame_sim --map empty --policy teleop_avoid --speed 0.5
python -m pyrobo.apps.pygame_sim --help
```

默认启动为 PAUSED，地图带有一个示范目标。按 G 前往该目标，或点击地图设置新目标；按 WASD 可直接接管。`--autostart` 会立即执行预设目标。

## 操作

| 输入 | 行为 |
| --- | --- |
| W / S 或 ↑ / ↓ | 前进 / 倒车速度期望；释放后请求零速 |
| A / D 或 ← / → | 差速为角速度，阿克曼为转角；阿克曼零速时只动转向机构 |
| 右侧圆形摇杆 | 按住并拖动：向上前进、向下倒车、向左右转向；松开归零 |
| USB 手柄 | 默认左摇杆轴 1 / 0，按住按钮 0（通常 A）允许接管 |
| 地图左键点击 | 设置位置目标；默认到达朝向为当前车位指向目标的方向 |
| 地图左键拖动 | 按下的位置为目标，拖动方向为到达朝向 |
| G / Auto | 恢复完整历史路径上的自动导航 |
| M / Manual | 进入手动模式 |
| Space / 右键 / Pause | 有限力制动，保留导航任务 |
| E / E-stop | 锁存急停，普通手动输入或 G 不能解除 |
| R / Release | 解除急停到 PAUSED，再选择模式或重新操作 |
| 1 / 2 | 手动严格停障 / 有界辅助绕障 |
| P | 显示或隐藏供规划器使用的点云 |
| Home / Reset | 重启整个实验，恢复初始位姿并清空任务进度与轨迹 |
| Esc / 关闭窗口 | 退出 |

手动输入会接管 AUTO；松开控制不会自动回到 AUTO。需要按 G 恢复导航。切换控制权时保留实际速度、转角、加速度和完整路径；只有显式 Reset 重启状态。

键盘优先于虚拟摇杆，虚拟摇杆优先于手柄。模式切换、急停解除后需要已按住的控制先归零 / 松开，防止残留输入立即接管。窗口失焦、活动手柄断开或 UI 心跳过期会暂停；恢复焦点不会自动恢复导航。

地图目标会按实际矩形、边距和点云校验，拒绝落在障碍内部或边界外的目标。全局路线仍是 XY 路线；阿克曼的目标朝向可能要求较大转弯空间，拖动设定朝向有助于测试这一点。

## 配置与覆盖顺序

完整可编辑模板：[pygame.toml](../configs/pygame.toml)。优先级为：内置默认值 → 配置文件 → 专用启动选项 → `--set`。同一字段重复 `--set` 时最后一个生效。

```bash
python -m pyrobo.apps.pygame_sim --config pyrobo/apps/configs/pygame.toml \
  --model ackermann --planner voronoi \
  --set footprint.length=1.0 --set footprint.width=0.5 \
  --set dynamics.mass=30.0 --set dynamics.longitudinal_drag.quadratic=2.5 \
  --set dwa.prediction_time=2.8 --set voronoi.time_budget=2.0
```

`--set` 的右侧采用 TOML 值语法。字符串需要保留引号，例如 `--set 'teleop.policy="teleop_avoid"'`。未知字段、无效算法名、非法数值、出生点碰撞以及超过看门狗期限的控制周期会在打开窗口前报错。

| 表 | 参数来源 |
| --- | --- |
| `backend` | `planner`、`model`、`local_optimizer`；当前局部优化器为 `dwa` |
| `app` | 窗口大小、显示 fps、仿真 dt、点云采样间距、轨迹长度、UI 超时、初始模式 |
| `map` | `preset = detour / empty / custom`，可覆盖 bounds / boxes / start / goal |
| `footprint` | Footprint：length / width / margin / offset_x |
| `differential` / `ackermann` | 对应运动学模型全部构造参数，两个表可共存，便于 CLI 切换 |
| `dynamics` | DynamicsConfig；三类 drag 可用嵌套表配置 |
| `dwa` | DWAConfig 全部参数；含采样数、预测时长、代价、批大小 |
| `rrt_connect` / `voronoi` | 相应全局规划器配置，包含预算、间距、种子等 |
| `navigation` | NavigationConfig：lookahead / goal_tolerance / yaw_tolerance / goal_approach_distance |
| `safety` | SafetyConfig：状态 / 地图年龄、周期、制动预算 |
| `teleop` | 策略、速度上限、转向上限、TTL 与最大允许偏离 |
| `joystick` | 启用、设备索引、轴号、反向、死区、使能按钮 |

省略 DWA 预测时间 / 前视距离时沿用车型默认值：差速 1.8 s / 1.3 m，阿克曼 2.4 s / 1.6 m。省略全局 clearance 时保留车型转弯余量。速度与力的实际限制由模型和 dynamics 决定；DWA 的配置不直接覆盖显式构造的物理参数。

`navigation.goal_approach_distance` 省略时使用车型默认终点入射长度，正数指定长度（m），`0` 关闭入射段。阿克曼终点前空间不足时可缩短，但仍须通过矩形碰撞检查。全局 `clearance` 是叠加在车体安全边距之外的额外间距，窄道中可单独调整。

全局规划软预算默认为 RRT-Connect 0.5 s、Voronoi 2 s。预算过小可能显示 `budget_exhausted` 并等待重试；可提高对应 `time_budget`。规划在工作线程进行，期间窗口继续响应输入。

`teleop.turn_limit=0` 表示使用车型上限；正数单位为差速 rad/s、阿克曼 rad。前后向速度均受模型上限约束。全部坐标为米，角度为弧度。

自定义地图示例：

```toml
[map]
preset = "custom"
bounds = [-5.0, 8.0, -4.0, 4.0] # xmin, xmax, ymin, ymax
start = [-3.0, 0.0, 0.0]         # x, y, yaw
# goal 可省略，随后在窗口中点击
boxes = [
  [0.0, 0.8, -1.0, 1.0],        # xmin, xmax, ymin, ymax
  [3.0, 3.6, 1.5, 3.0],
]
```

地图使用 World2D 实体矩形与边界；算法收到沿表面采样的原始点云。当前例程不导入外部点云文件或提供地图编辑器。实体碰撞真值独立于点云规划，碰撞后运动终止，按 Home 重启实验。

## 手柄映射

默认上推 / 左推对应负轴值，因此两个 invert 开关为 true。不同设备可调整轴号与按钮：

```bash
python -m pyrobo.apps.pygame_sim \
  --set joystick.throttle_axis=3 --set joystick.turn_axis=2 \
  --set joystick.deadman_button=4 --set joystick.deadzone=0.15
```

`device_index` 为连接设备的选择索引；运行中按 pygame 的 instance ID 处理拔插。配置了不存在的轴 / 按钮时不产生运动输入，窗口显示原因。`deadman_button=-1` 可取消按钮使能要求，此时摇杆离开死区即接管。接口依据 [pygame joystick 文档](https://www.pygame.org/docs/ref/joystick.html)；实物设备需确认自己的轴向与按钮映射。

## 运行与测试

SDL 事件和绘制在主线程；一个工作线程独占 Navigator、Supervisor 与 Simulator。人工期望进入同一个 DWA，输出经过独立 SafetyGuard。UI 不修改位姿或直接覆盖底盘命令。

工作线程使用固定仿真 dt，不在耗时规划后连续追赶补帧。规划很慢时仿真时间会变慢，界面仍可处理操作；执行前检查新急停、失焦及新的操作，废弃过时结果并制动。控制延迟显示在右侧，这个例程不承担实机实时控制。未知异常显示为故障并冻结实验，Reset 可重新开始。

无显示环境也能运行完整控制、渲染与截图：

```bash
python -m pyrobo.apps.pygame_sim --headless --autostart --steps 80 \
  --planner voronoi --model ackermann --set footprint.length=1.0 \
  --screenshot /tmp/pyrobo-workbench.png
python -m pytest tests/test_pygame_session.py tests/test_pygame_ui.py -q
```

headless 使用 SDL dummy 显示与确定性的同步仿真步；常规窗口使用工作线程。`--steps` 限制当前实验的控制步数，headless 默认 100；`--screenshot` 保存最后一帧。退出时输出 JSON 状态，碰撞或后台异常返回非零退出码。

验证包含真实规划器闭环到达、键盘 / 虚拟摇杆事件、模拟手柄拔插、人工接管、失焦与超时、急停锁存、耗时优化期间急停以及坐标转换。UI 测试在未安装 pygame 的环境跳过，纯配置与后端测试仍运行。实体手柄的设备兼容性需要在对应硬件上验证。
