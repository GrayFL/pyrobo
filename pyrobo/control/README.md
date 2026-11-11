# 局部规划算法开发接口

`ref_dwa` / `ref_teb` 是按经典数学结构整理的最小可运行实现，`towing_dwa` /
`towing_teb` 分别由它们复制。四个目录具有独立的配置类、控制器和目标函数，修改 towing
不会改变 ref。原有 `dwa` / `teb` / `eb_dwa` 继续可用。

## 从哪里开始修改

| 算法 | 核心代码 | 内容 |
| --- | --- | --- |
| ref_dwa | [optimizer.py](ref_dwa/optimizer.py) | `velocity_window()` 动态窗口；`objective()` 三项评分 |
| ref_teb | [optimizer.py](ref_teb/optimizer.py) | `solve_band()` 内的 `residual()`：位姿和时间的最小二乘 |
| towing_dwa | [optimizer.py](towing_dwa/optimizer.py) | ref_dwa 核心的独立副本 |
| towing_teb | [optimizer.py](towing_teb/optimizer.py) | ref_teb 核心的独立副本 |

每个目录的 `config.py` 定义可保存的参数；`controller.py` 实现对外入口与本算法的执行流程；
`optimizer.py` 放数值优化。DWA 的 `rollout.py` 管候选预测与可行性，TEB 的 `execution.py`
只检查本周期首段命令与制动。它们都是各算法的内部模块，可按需替换或增删；没有公共
基类，也没有必须实现的 `prepare/evaluate/result` 流程。同家族可自行继承，当前 towing
保留独立副本以便分别开发。基础实现不包含旧算法的路径距离评分、转向状态机、换向惩罚、
停滞脱困或 TEB 的多预览/缩速候选及 DWA 二次评分。TEB 的全局路线仅用来初始化 band，
没有贴合路线的代价项。保留 NumPy 批量预测、KD-tree、稀疏 Jacobian、LSMR 与稀疏结构缓存。

这是经典结构的 Python 最小实现，不是原论文或 ROS 插件的完整移植。DWA 的圆形机器人/
理想圆弧距离改用本库矩形车体与共享 plant 的轨迹净空；TEB 固定本周期节点数，只优化
单条 band，不做多拓扑搜索、自动重采样或热启动。TEB 时间代价为 `sum(dt_i**2)`，
默认不使用墙钟截断（`time_budget=None`）。复杂窄路、局部极值可能停滞，留给后续开发。

## 对外提供什么

核心协议见 [interfaces.py](interfaces.py)：

```python
def optimize(self, observation: RobotObservation, request: MotionRequest) -> MotionResult:
    ...
```

这是结构化协议，不要求继承任何类。逐周期调用只要求 `optimize(obs, req)`；内部可以是
速度采样、轨迹优化或其他方法，外部不访问 `_prepare`、`_seed_band` 等私有方法。

| 方法 | 调用约定 |
| --- | --- |
| `optimize(obs, req) -> MotionResult` | 唯一必需入口；读取观测和请求、提出下一周期命令，不推进真实底盘 |
| `reset() -> None` | 可选生命周期钩子；清理本地缓存/热启动/历史控制，支持重复调用；无状态算法可省略 |

内置算法通过注册表构造时，构造器接受 `config`、`footprint` 及关键字 `model`、`dynamics`、
`command_space`、`index`、`input_limits`、`teleop_config`，这是工厂装配约定，非优化协议。
它们暴露 `config`、`model`、`footprint`、`plant`、`command_space` 供现有 Web/Session
装配。`Navigator(local_optimizer=...)` 也接受只实现 `optimize` 的对象；此时在 Navigator
构造器显式提供相符的模型、车体、动力学等依赖。新算法不必拥有 `predictor` 或候选数组。
无需继承旧 `DWAEngine`，无需启动线程或读取 Web/ROS 状态。

`MotionResult` 的主要字段：

| 输出 | 约定 |
| --- | --- |
| `command` | 差速为 `WheelCommand(left, right, mode)`，左右轮地速 m/s；汽车为 `AckermannCommand(speed, steering, mode)`，m/s、rad |
| `status` | `MotionStatus`，例如 `TRACKING`、`BRAKING`、`EMERGENCY_STOP` |
| `safety_assessment` | `SafetyAssessment`；`result.safe` 由它计算，不自行赋布尔值 |
| `trajectory` | `Trajectory` 或 `None`；提供时使用 graphmap `PoseArray`、递增相对 `times`、`start_time`、可选状态数组 `states`、`prediction_end`，明确轨迹含义及覆盖时域 |
| `diagnostics` | `Diagnostics`：耗时、候选数、可行数、净空、`cost_terms`、失败原因；`extra['optimizer']` 标识实际算法 |

当前 DWA 返回保持候选到预测时域末尾再制动的物理预测；当前 TEB 返回首段命令执行 `obs.dt`
再制动的物理预测，`diagnostics.extra['trajectory_kind']='execution_then_brake'`。
TEB 的 `last_band` 是另一个内部优化结果，不能冒充真实 plant 执行轨迹；候选数为 1
仅描述它提出一条命令，不要求 TEB 采用候选选优。两者当前都选择复用 `verified_stop()`
作有限力制动退化，这是实现选择，不是必须继承的算法步骤。

## 能获得什么，如何读取

以下数据由 `Navigator` / `Supervisor` 每周期作为函数参数注入：

| 信息 | 读取位置 |
| --- | --- |
| 实测位姿 | `observation.chassis.pose`；`planar_state(pose)` 转为 `[x, y, yaw]` |
| 实测线/角速度 | `observation.chassis.velocity.linear / angular` |
| 转向与动态历史 | `chassis.steering / steering_rate / acceleration / yaw_acceleration / steering_acceleration` |
| 当前障碍快照 | `observation.obstacles.points` 为只读 Nx3；`points[:, :2]` 为 XY；`timestamp=None` 表示静态地图 |
| 观测时刻、坐标系、边界 | `observation.timestamp / frame / bounds` |
| 下一执行周期 | `observation.dt`，不是前后观测的时间差 |
| 完整全局路径 | `request.path` 是 `Path`，`.xy` 为 Nx2，`.poses` 为 PoseArray |
| 剩余路径与进度 | `request.remaining_path / remaining_distance`，可选；不得破坏完整路径 |
| 局部目标、终点 | `request.local_target / goal` 为 Pose，前者可选 |
| 模式与请求期限 | `request.policy / timestamp / ttl / source`；`request.fresh(now)` 判断时效 |
| 手动期望 | `request.desired` 为 Twist2D，另有 `steering` 和允许偏离量；手动/停车请求可能没有路径 |
| 车体与运动上限 | 构造时注入的 `self.footprint / self.model / self.command_space.bounds()` |
| 物理与预测 | `self.plant.config`；`plant.codec.pack(chassis)`；按需构造 `BatchPredictor` 或设计自己的预测器，并非协议必需属性 |
| 矩形碰撞查询 | `world = self.index.get(observation.obstacles)`；`world.clearances(poses, self.footprint, observation.bounds)` |

单位统一为 m、s、rad，x 前、y 左；位姿位于显式世界系，参考点为差速轴中点或汽车后轴中点。
传感器数据由外部用 `PointCloud.transformed(T_world_sensor)` 转到同一世界系后送入。
算法无法从这些接口取得障碍物未来轨迹、牵引挂车关节状态或未观测空间信息；需要时应新增明确输入。

## 独立调用与装配

```python
import numpy as np
from pyrobo import (Bounds2D, ChassisState, MotionPolicy, MotionRequest, Navigator,
                    Path, PointCloud, RefDWA, RefDWAConfig, RobotObservation, pose2d)
from pyrobo.core.geometry import path_poses

optimizer = RefDWA()  # 可改为 RefTEB / TowingDWA / TowingTEB
state = ChassisState(pose2d(0, 0), timestamp=0.)
obs = RobotObservation(state, PointCloud(np.empty((0, 3))), Bounds2D(-5, 5, -4, 4), dt=.1)
req = MotionRequest(MotionPolicy.NAVIGATION, obs.timestamp,
    path=Path(path_poses([[0, 0], [3, 0]])), goal=pose2d(3, 0))
result = optimizer.optimize(obs, req)

# 实际导航建议通过 Navigator / Supervisor，以保留独立最终执行检查。
nav = Navigator(controller=RefDWAConfig())
# 自定义实现也可：Navigator(local_optimizer=optimizer)
```

`ReferenceAdapter` 已移除。各算法的输入检查、优化、验证和结果打包完全由本目录控制器
安排：DWA 检查各候选的完整时域和制动；TEB 优化 band 后，检查首段命令执行一个周期及
制动，不把首段速度恒定外推整个 band 时域。band 后续速度/曲率可以改变。
共享几何/动力学工具可按需调用，不提供跨算法的强制求解模板。

`Navigator` 负责全局路径、进度和到达判定；`Navigator.step` / `Supervisor.step` 在
`optimize` 返回后调用独立 `SafetyGuard` 复核实际执行命令，不信任算法自己报告的安全或
轨迹。直接调用 `optimize` 不会隐式经过这个外部安全门。各控制器当前保留私有的 DWA
手动委托，自主分支不调用旧 DWA 导航策略或 rollout 评估。原有三个算法目录未修改。

## 车体与 Web 测试

四个算法均支持差速和阿克曼。构造时分别传入 `model=` 与 `footprint=`；算法名称不决定
车型或几何。未传车体时统一使用 `Footprint()`：长 0.5、宽 0.5、margin 0.08、offset 0。
例如长 0.79、宽 0.54、差速轴距前沿 0.10 m 的车体，可与任一算法组合：

```text
offset_x = 0.10 - 0.79/2 = -0.295 m
前沿 x = +0.10 m；后沿 x = -0.69 m；左右边沿 y = ±0.27 m
```

在代码中传入 `Footprint(length=.79, width=.54, offset_x=-.295)` 即可。
轮距、质量、速度/力限幅等其余参数沿用已有默认值；towing 尚未加入挂车模型或牵引约束。

Web「局部规划」可选择四个名称，切换时保留当前车型、全部尺寸与中心偏移；车型选择始终可用。
「车体中心相对驱动轴」显示偏移；差速轴画在位姿原点。
选择后点击「应用并重置」，再点「自动导航」。四个独立配置表与场景一起保存/导入/导出，
TEB 面板编辑当前选中的 TEB 参数，其余参数可在高级 JSON 中修改。
仅切换算法、参数无需重启；修改 Python 源码后需重启 Web 后端并刷新页面。

手动开发时也可在仓库根目录启用源码自动重载（需要 web 可选依赖）：

```bash
python -m uvicorn pyrobo.apps.web_sim.server:create_app --factory \
  --host 0.0.0.0 --port 8090 --reload --reload-dir pyrobo
```

保存 `.py` 文件触发重载并重置后端会话，之后刷新页面或点「重新连接」。
场景库沿用应用的默认保存目录，需保留的实验先保存场景。

新增算法在实现类上使用 `pyrobo.registry.register_local_optimizer` 装饰器登记配置类、工厂和算法族；模块导入时自动注册，Web/Pygame 从目录生成选择器。包内新算法由应用启动扫描导入；外部模块可用 `PYROBO_PLUGINS` 指定。详细示例见 [Web 自动发现说明](../apps/web_sim/README.md#算法与车辆模板自动发现)。
Web 的工作量上限属于交互服务资源限制，不进入独立优化函数。
