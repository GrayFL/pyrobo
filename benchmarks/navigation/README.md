# 导航基准与真实时间仿真

本目录提供以下实验，结果保留失败样本、参数、源代码摘要和环境版本。

EB-DWA 的保存场景、逐周期指标与同场景算法对照使用：

```bash
python -m benchmarks.navigation.eb_dwa --scenes all --logical --output reports/eb-cases
python -m benchmarks.navigation.eb_dwa --scenes narrow_road --optimizers dwa teb eb_dwa \
  --planners rrt_connect voronoi --seeds 7 11 23 --output reports/eb-narrow
python -m benchmarks.navigation.teb --include-eb --repeats 30 --output reports/eb-latency.json
```

`--logical` 关闭 EB 和 RRT 的墙钟预算，保留评估/迭代/扩展上限；专门的预算故障场景仍保留其预算。Voronoi 本身不使用 seed，因此不同 seed 行是重复运行。实际 Web 服务采用相同 `SceneSession`；回放不会在计算时推进物理时间，不能证明硬实时。

当前 EB 默认 `time_budget_s=None`，所以 `--logical` 对普通 EB 场景额外关闭的是 RRT 墙钟预算。最终 20 场景、重复窄路、60 ms 失败对照与串行性能见 [EB-DWA 验收报告](../../docs/eb_dwa_results.md)；可跟踪原始汇总为 [eb_dwa_results.json](eb_dwa_results.json)。每个 `.scene.json` 都可直接交给 `python -m pyrobo.apps.web_sim.replay`，或导入网页复测。

每次运行保留完整展开场景、SHA256、所有逐周期数据以及失败结果。`local_ms` 是局部算法与候选验证；`tick_ms` 包括全局导航、独立安全门和实体步进；`eb_ms` / `dwa_ms` / `speed_profile_ms` 是内部阶段；全局搜索耗时另列。回放在记录器中的额外几何查询不计入这些阶段，包含于 `wall_seconds_with_metrics`。性能对照应串行执行。

净空为完整矩形（含 margin、offset）、点半径和边界的查询值，另外记录独立实体碰撞。实际曲率取 `omega/v`，只统计 `abs(v)>=.05 m/s`；变化率仅统计相邻移动点且间距大于 .005 m。直道/弯道速度分别选 `abs(curvature)<.15` / `>=.4` 的移动点。制动起始距离是首次纵向减速时到前方全局折线转角（角差 >.25 rad）的距离，无相应点时为 null。横向误差分别相对当前全局路径与当前 EB 参考，不能把两种误差混作同一指标；EB 改变路径形状本就可能增大全局横向距离。统计中保留重规划和所有停止周期。

| 入口 | 用途 | 物理时间 |
| --- | --- | --- |
| `planning` | 相同预算、多种子、两种全局规划器的成功率与路径质量 | 不执行车辆 |
| `control` | DWA + 全局路线 + 动力学闭环，前视距离 / 车型 / 速度 / 矩形对比 | 固定 dt；求解时物理暂停 |
| `rsim_platform` | 实际 RSim 多进程、计算卡顿与传感器失效 | 随单调时钟持续前进 |
| `teb` | 同一实测状态和原始点云下，DWA / TEB（`--include-eb` 增加 EB-DWA）延迟与内存 | 固定输入，不推进被测控制器的物理 |
| `eb_dwa` | 保存 Web 场景的三算法对照、故障验收与逐周期指标 | 固定 dt；求解时物理暂停 |

同步闭环的计算耗时是真实测量，但不能用它证明卡顿时的物理安全。只有第三类实验用于这项结论。

## TEB 局部决策测量（2026-11-09）

`python -m benchmarks.navigation.teb --repeats 30 --output reports/teb-latency.json` 可复现对比。两算法使用同一实测状态序列、1 × 0.5 m 矩形（offset_x=0.25 m）、2.4 s 预测窗口及完整 1000 / 10000 点直道墙面。每组预热一次，再记录 30 次决策。包含优化、物理预测及制动验证，不含全局规划和独立最终安全门。计时时无其他测试或仿真并行；Python 3.13.14、NumPy 2.5.1、SciPy 1.18.0，BLAS 单线程。

| 模型 / 点数 | DWA p50 / p95 / p99 · ms | TEB p50 / p95 / p99 · ms |
| --- | --- | --- |
| 差速 / 1000 | 50.62 / 51.50 / 52.57 | 60.49 / 70.36 / 73.63 |
| 差速 / 10000 | 167.20 / 169.95 / 170.03 | 71.14 / 83.26 / 84.77 |
| 阿克曼 / 1000 | 36.62 / 36.81 / 36.95 | 62.37 / 69.97 / 72.38 |
| 阿克曼 / 10000 | 95.85 / 96.89 / 97.36 | 72.63 / 82.06 / 85.20 |

该工况下 TEB 验证 12–16 条参考，DWA 验证 137–141 条候选。1000 点时稀疏求解固定开销使 TEB 较慢；10000 点时减少物理候选及其矩形邻域查询带来收益，不能概括为所有地图均比 DWA 快。全部输出通过模型安全检查；原始报告保留逐次延迟、求解状态、制动尾段距离、最小间距、配置与源码摘要。

内存另用 tracemalloc 测一个完整决策，为避免追踪开销触发提前退出，该次 TEB 关闭墙钟预算、保留次数上限。Python 跟踪峰值在本直道工况为 TEB 约 0.11 MiB；DWA 1000 点为差速 11.77 / 阿克曼 5.11 MiB，10000 点为 119.57 / 46.65 MiB。它不包含进程 RSS 或全部原生库分配，不能当成任意地图的内存上界。两算法仍使用完整点云。

本轮全量 pytest 373 项通过；最终 Web 布尔诊断序列化及手动模式共享点云缓存修复后，TEB 核心、RSim 和 Web 专项 34 项复验通过。实际 Edge 浏览器验证两车型到达、参数往返和手机布局；偏移长车体回放及 Memory/MuJoCo 受控导航通过。源码树本地证据位于 `assets/teb-20261109/`，最终专项日志为 `final-cache-focused.log/xml`，最终性能原始文件为 `latency-final.json`（最后的手动缓存修改不涉及所测自动导航路径）；早期 `latency.json` 的内存测量受墙钟预算影响，不用于上表结论。

## 地图和指标

`scenarios.py` 定义直道、偏置窄门、交错墙、U 形陷阱、连续绕桩、仓库、不可通行隔墙、用户「窄道阿克曼001」。地图为连续几何实体；全局 / 局部算法接收原始表面点云，实体碰撞由独立仿真几何检查。

全局报告包含每次成功 / 状态、p50 / p95 / p99 / 最大耗时、超预算量、扩展次数、长度、间距和完整路线。成功率只计独立几何检查通过的路线；无解隔墙应报告 0 条路线。几何验证以不超过 4 cm 平移、2° 旋转的密度检查矩形，不等同于汽车可达性证明。

全局 `minimum_margin_clearance_m` 是返回航点处的点云矩形间距；闭环同名指标在记录的实际位姿上计算，均已扣除车体 margin 和点半径。这些采样指标不是整条连续扫掠轨迹的最小实体距离，实体碰撞结果单独记录。`planning` / `control` 正常完成实验即返回 0，验收必须读取报告的成功 / 到达 / 碰撞字段。

闭环记录实际到达、实体碰撞、最大 / 平均行驶速度、间距、路径误差、转向变化、倒车和停车帧数。`turn_reversals` 在实际转角 / 角速度超过 0.03 后统计变号；弯道的正常左右转也会计入，不能把这个数字直接称为“画龙次数”。

```bash
python -m benchmarks.navigation.planning --repeats 20 --budget .5 \
  --approach 1 --no-bridge-sampling --output reports/uniform.json
python -m benchmarks.navigation.planning --repeats 20 --budget .5 \
  --approach 1 --output reports/bridge.json
python -m benchmarks.navigation.planning --scenarios narrow_ackermann \
  --clearance .4056296013484 --adaptive --output reports/adaptive.json
python -m benchmarks.navigation.control --models differential ackermann \
  --steps 700 --output reports/closed-loop.json
python -m benchmarks.navigation.control --scenarios straight offset_gate \
  --models ackermann --lookahead-scales .75 1 1.25 --max-speed .25 \
  --steps 1200 --output reports/low-speed.json
```

全局对照默认具体规划器的额外 clearance 为硬约束；`--adaptive` 才将它解释为首选间距，硬下限始终包含原矩形 margin 和点半径。`--approach` 留空自动适配，正数固定入射段长度。闭环默认使用自动间距组合和自动入射段。

## RSim 延迟实验

安装 / 端口 / 部署说明见 [RSim 导航组件](../../pyrobo/rsim/README.md)。默认全局、DWA、执行各一个 worker，加上主进程物理仿真，报告中应看到四个不同 PID。`--local` 可验证相同组件图在单进程的接口行为。

```bash
python -m benchmarks.navigation.rsim_platform --scenario straight \
  --control-delay-every 1 --control-delay .1 --output reports/delay100.json
python -m benchmarks.navigation.rsim_platform --scenario narrow_ackermann \
  --control-delay-every 8 --control-delay .4 --output reports/delay400.json
python -m benchmarks.navigation.rsim_platform --scenario staggered --model differential \
  --planning-delay .8 --sensor-pause-period 5 --sensor-pause .45 \
  --output reports/sensor-pause.json
python -m benchmarks.navigation.rsim_platform --scenario straight \
  --map-pause-period 5 --map-pause .55 --output reports/map-pause.json
python -m benchmarks.navigation.rsim_platform \
  --scene pyrobo/apps/scenes/narrow_ackermann_adaptive.json --planner voronoi \
  --output reports/saved-scene.json
```

`--scene` 导入网页保存的地图、初始 / 目标位姿、车体、车型、动力学和导航设置。`--model` / `--planner` / `--max-speed` / `--clearance` / `--approach` 可显式覆盖相关设置；报告 `config` 是最终生效参数，`scene` 保留来源场景。使用内置 `--scenario` 时，来源场景仅提供地图及起终点，配置采用测试平台默认值，不能把来源 JSON 中的旧配置误读为实际运行参数。

延迟在真实 worker 中阻塞。默认结果期限 0.3 s：400 ms 后返回的结果会被丢弃，执行进程继续处理新状态并制动，不把仿真时间冻结。传感器暂停只停发布观测，真值与物理继续运行。地图暂停实验将地图标为动态，并保留最后一次获取时间；正常静态地图不应人为过期。

报告包含真实控制 / 全局 / 执行延迟、超过 100 / 400 ms 的数量、执行端到期次数、旧结果拒绝、陈旧观测制动原因、物理最大更新间隔、每帧状态和各组件 PID。CLI 成功到达且无实体碰撞返回 0，未到达也保存报告并返回 1。

## 绘图与复现

```bash
python -m benchmarks.navigation.plot reports/bridge.json --output reports/maps.png
python -m benchmarks.navigation.plot reports/delay400.json --output reports/delay400.png
python -m benchmarks.navigation.plot reports/closed-loop.json --output reports/closed-loop.png
python -m scripts.benchmark_control --output reports/control-performance.json
python -m benchmarks.navigation.efficiency --output reports/early-rejection.json
```

`efficiency` 在接近制动边界的固定输入上，成对比较完整候选预测与先拒绝不安全候选的完整控制调用；同时验证停车命令。它用于隔离提前拒绝的收益，不是旧版本完整控制器的耗时对比。一般场景性能另由 `scripts.benchmark_control` 测量。

报告保存在调用方指定路径；仓库内的实测数据及限制见 [本轮结果](../../reports/navigation/README.md)。性能对照应在相同环境、相同预算与负载下运行；不要把一次成功当作普遍成功保证。RRT 随机种子可重复，墙钟预算和系统调度仍会导致边缘样本差异。

方法参考 [OMPL benchmark](https://ompl.kavrakilab.org/benchmark.html)、[DWA 原论文](https://publications.ri.cmu.edu/the-dynamic-window-approach-to-collision-avoidance) 和 [Nav2 Regulated Pure Pursuit](https://github.com/ros-navigation/navigation2/tree/main/nav2_regulated_pure_pursuit_controller)。窄道采样参考 [Bridge Test](https://adacomp.org/wp-content/uploads/2016/01/icra03.pdf)，实现按点云表面构造候选中点，仍逐边验证矩形。
