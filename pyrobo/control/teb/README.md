# Timed Elastic Band

`TimedElasticBand`（别名 `TEBEngine`）实现 `LocalMotionOptimizer`，用于差速与阿克曼局部导航。全局路线仍由 RRT-Connect / Voronoi 提供，DWA 仍为默认算法。无需 ROS、g2o、Torch 或新增 Python 依赖。

```python
from pyrobo import Ackermann, Footprint, Navigator, TEBConfig, TimedElasticBand

nav = Navigator(model=Ackermann(wheelbase=.5),
                footprint=Footprint.long(offset_x=.25),
                controller=TEBConfig(lookahead=1.6))
```

也可注入 `local_optimizer=TimedElasticBand(...)`。沿用 `Navigator.step(NavigationInput(..., chassis=state))` 或 `Supervisor`，传完整实测状态。输出为模型原生命令，公共执行轨迹使用带时间的 `Trajectory` / graphmap `PoseArray`。

## 算法与安全边界

局部窗口重采样为有界 `(x,y,yaw)` 节点，固定两端，联合优化内部位姿与相邻正时间间隔。残差包含总时间、路径偏离、矩形障碍间距、非完整运动学、速度与加速度；差速增加轮速限制，阿克曼增加最小转弯半径、从实测转角开始的转向速率和倒车方向。限制来自现有模型与动力学配置。

SciPy 稀疏 Jacobian 的局部连接结构按节点数缓存。NumPy 批量计算节点和段中点距离，使用完整当前点云的 KD-tree，不隐式栅格化或降采样。`Footprint.offset_x`、margin、点半径和边界与现有几何一致。相同路线、目标、模式和坐标约定下，按经过的时间移动、重采样上一 band；新点云始终重新计算。只保留一条 band，reset、路径/目标变化、过大跳变、时间回退和手动切换后重新初始化。

从 band 的时间速度和路径前视生成至多 16 条跟踪参考，使用共享有限力 plant 批量验证，按 band 跟踪与现有到达策略选优。跟随路径时，非零速度的减速参考保持曲率：差速同步缩放线速度和角速度，阿克曼缩放车速并保持转角。差速另外保留零线速度的原地转向候选，用于起步和航向修正；对齐及恢复模式沿用各自的转向意图。不枚举 DWA 速度网格。实际指令必须通过完整预测与制动尾段、一个执行周期后制动两项连续矩形扫掠检查，最终安全门再独立复核。无可行指令时返回验证过的停车；无法证明制动安全时 `safe=False`。

`last_band` 是只读 NumPy 优化诊断，含 `poses`、`intervals`、`times` 和代价。其软约束和离散避障残差不构成执行安全证明；`MotionResult.trajectory` 才是实际获选指令经 plant 产生的预测，包含制动尾段。每周期只执行当前指令。

手动 `TELEOP_STOP` / `TELEOP_AVOID` 显式委托既有 DWA，保留不擅自加速、反向及有界转向偏离的语义；Web 的 `dwa` 参数继续用于手动辅助，诊断显示实际算法。

## 参数与诊断

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `prediction_time` / `integration_dt` | 2.4 / 0.08 s | plant 预测窗口 / 最大记录间隔 |
| `lookahead` | 1.3 m | 前视，阿克曼应用装配默认 1.6 m |
| `min_samples` / `max_samples` | 5 / 16 | 节点数下限 / 上限 |
| `dt_ref` / `dt_min` / `dt_max` | 0.3 / 0.06 / 1.5 s | 初始化密度 / 优化后每段时间范围 |
| `max_iterations` | 12 | least_squares 主残差评估上限，不含差分 Jacobian |
| `time_budget` | 0.06 s | 优化软预算；`None` 仅限制次数 |
| `obstacle_clearance` | 0.12 m | 真实轮廓及 margin 外的软间距偏好 |
| `warm_start` / `allow_reverse` | true / true | 复用 band / 允许倒车 |

`*_weight` 调整残差权重，不能降低真实碰撞轮廓或指令上限。预算耗尽使用已评估的最低代价 band，再执行全部物理验证。预算不含初始化、指令预测和最终安全门，单次 SciPy/KD-tree 运算不能硬抢占。优化后的 band 总时长可与 `prediction_time` 不同。

`diagnostics.extra` 提供 `optimizer`、`teb_status`（`converged` / `iteration_limit` / `time_budget`）、节点数、残差调用数、初始/最终代价、总时长和复用标记。`candidate_count` 是物理验证参考数。预算中断时 `teb_iterations=0` 表示 SciPy 未返回最终主评估计数，实际调用数仍在 `teb_evaluations`。

单条 band 不搜索多个拓扑、不预测动态障碍，也不是全局 SE(2) 停车规划器；不合适的 XY 路线、极窄空间和局部极值仍可能阻塞。`converged` 不代表到达或安全。

## 应用与参考

Web 使用 `backend.local_optimizer="teb"` 与独立 `teb` 参数表；旧场景默认 DWA，schema_version 保持 1。Pygame 支持 `--local-optimizer teb`。RSim 使用 `StackConfig(controller=TEBConfig(...))`，沿用现有状态、命令、操作限速及期限接口连接 Memory / MuJoCo；锁步要求 `time_budget=None`。

```bash
python -m pyrobo.apps.lockstep_navigation --local-optimizer teb --execution process --episodes 2
python -m benchmarks.navigation.teb --repeats 30 --output reports/teb-latency.json
python -m pytest -q tests/test_teb.py tests/test_rsim_teb.py
python -m pyrobo.apps.web_sim.replay pyrobo/apps/scenes/teb_narrow_road.json --steps 650
```

[窄路差速回归场景](../../apps/scenes/teb_narrow_road.json)保留原始墙体、起终点、0.5 m 方形车体和 0.08 m 碰撞边距。全局额外净距固定为 0，以隔离局部 TEB 在连续弯道的减速跟踪；两种全局规划器均需实际到达、停止且所有指令通过安全门。该用例不验证自适应全局搜索的预算分配，也不代表阿克曼倒车机动已解决。


参考 BSD 许可的 [TU Dortmund teb_local_planner](https://github.com/rst-tu-dortmund/teb_local_planner) 数学建模与文档，独立实现 Python 求解和本库动力学执行层，没有复制 C++ 代码或引入其依赖。具体参考：[band 顶点](https://github.com/rst-tu-dortmund/teb_local_planner/blob/noetic-devel/include/teb_local_planner/timed_elastic_band.h)、[运动学与转弯半径](https://github.com/rst-tu-dortmund/teb_local_planner/blob/noetic-devel/include/teb_local_planner/g2o_types/edge_kinematics.h)、[速度与精确弧长](https://github.com/rst-tu-dortmund/teb_local_planner/blob/noetic-devel/include/teb_local_planner/g2o_types/edge_velocity.h)。论文见[上游引用列表](https://github.com/rst-tu-dortmund/teb_local_planner#citing-the-software)，包括 Rösmann 等的 ECMR 2013 稀疏优化及 IROS 2017 汽车运动学优化论文。
