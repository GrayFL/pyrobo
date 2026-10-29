# 导航基准与真实时间仿真

本目录提供三类实验，结果均保留失败样本、参数、源代码摘要和环境版本。

| 入口 | 用途 | 物理时间 |
| --- | --- | --- |
| `planning` | 相同预算、多种子、两种全局规划器的成功率与路径质量 | 不执行车辆 |
| `control` | DWA + 全局路线 + 动力学闭环，前视距离 / 车型 / 速度 / 矩形对比 | 固定 dt；求解时物理暂停 |
| `rsim_platform` | 实际 RSim 多进程、计算卡顿与传感器失效 | 随单调时钟持续前进 |

同步闭环的计算耗时是真实测量，但不能用它证明卡顿时的物理安全。只有第三类实验用于这项结论。

## 地图和指标

`scenarios.py` 定义直道、偏置窄门、交错墙、U 形陷阱、连续绕桩、仓库、不可通行隔墙、用户「窄道阿克曼001」。地图为连续几何实体；全局 / 局部算法接收原始表面点云，实体碰撞由独立仿真几何检查。

全局报告包含每次成功 / 状态、p50 / p95 / p99 / 最大耗时、超预算量、扩展次数、长度、间距和完整路线。成功率只计独立几何检查通过的路线；无解隔墙应报告 0 条路线。几何验证以不超过 4 cm 平移、2° 旋转的密度检查矩形，不等同于汽车可达性证明。

全局 `minimum_margin_clearance_m` 是返回航点处的点云矩形间距；闭环同名指标在记录的实际位姿上计算，均已扣除车体 margin 和点半径。这些采样指标不是整条连续扫掠轨迹的最小实体距离，实体碰撞结果单独记录。`planning` / `control` 正常完成实验即返回 0，验收必须读取报告的成功 / 到达 / 碰撞字段。

闭环记录实际到达、实体碰撞、最大 / 平均行驶速度、间距、路径误差、转向变化、倒车和停车帧数。`turn_reversals` 在实际转角 / 角速度超过 0.03 后统计变号；弯道的正常左右转也会计入，不能把这个数字直接称为“画龙次数”。

```bash
python -m examples.navigation_benchmark.planning --repeats 20 --budget .5 \
  --approach 1 --no-bridge-sampling --output reports/uniform.json
python -m examples.navigation_benchmark.planning --repeats 20 --budget .5 \
  --approach 1 --output reports/bridge.json
python -m examples.navigation_benchmark.planning --scenarios narrow_ackermann \
  --clearance .4056296013484 --adaptive --output reports/adaptive.json
python -m examples.navigation_benchmark.control --models differential ackermann \
  --steps 700 --output reports/closed-loop.json
python -m examples.navigation_benchmark.control --scenarios straight offset_gate \
  --models ackermann --lookahead-scales .75 1 1.25 --max-speed .25 \
  --steps 1200 --output reports/low-speed.json
```

全局对照默认具体规划器的额外 clearance 为硬约束；`--adaptive` 才将它解释为首选间距，硬下限始终包含原矩形 margin 和点半径。`--approach` 留空自动适配，正数固定入射段长度。闭环默认使用自动间距组合和自动入射段。

## RSim 延迟实验

安装 / 端口 / 部署说明见 [RSim 导航组件](../../pyrobo/integrations/rsim/README.md)。默认全局、DWA、执行各一个 worker，加上主进程物理仿真，报告中应看到四个不同 PID。`--local` 可验证相同组件图在单进程的接口行为。

```bash
python -m examples.navigation_benchmark.rsim_platform --scenario straight \
  --control-delay-every 1 --control-delay .1 --output reports/delay100.json
python -m examples.navigation_benchmark.rsim_platform --scenario narrow_ackermann \
  --control-delay-every 8 --control-delay .4 --output reports/delay400.json
python -m examples.navigation_benchmark.rsim_platform --scenario staggered --model differential \
  --planning-delay .8 --sensor-pause-period 5 --sensor-pause .45 \
  --output reports/sensor-pause.json
python -m examples.navigation_benchmark.rsim_platform --scenario straight \
  --map-pause-period 5 --map-pause .55 --output reports/map-pause.json
python -m examples.navigation_benchmark.rsim_platform \
  --scene examples/scenes/narrow_ackermann_adaptive.json --planner voronoi \
  --output reports/saved-scene.json
```

`--scene` 导入网页保存的地图、初始 / 目标位姿、车体、车型、动力学和导航设置。`--model` / `--planner` / `--max-speed` / `--clearance` / `--approach` 可显式覆盖相关设置；报告 `config` 是最终生效参数，`scene` 保留来源场景。使用内置 `--scenario` 时，来源场景仅提供地图及起终点，配置采用测试平台默认值，不能把来源 JSON 中的旧配置误读为实际运行参数。

延迟在真实 worker 中阻塞。默认结果期限 0.3 s：400 ms 后返回的结果会被丢弃，执行进程继续处理新状态并制动，不把仿真时间冻结。传感器暂停只停发布观测，真值与物理继续运行。地图暂停实验将地图标为动态，并保留最后一次获取时间；正常静态地图不应人为过期。

报告包含真实控制 / 全局 / 执行延迟、超过 100 / 400 ms 的数量、执行端到期次数、旧结果拒绝、陈旧观测制动原因、物理最大更新间隔、每帧状态和各组件 PID。CLI 成功到达且无实体碰撞返回 0，未到达也保存报告并返回 1。

## 绘图与复现

```bash
python -m examples.navigation_benchmark.plot reports/bridge.json --output reports/maps.png
python -m examples.navigation_benchmark.plot reports/delay400.json --output reports/delay400.png
python -m examples.navigation_benchmark.plot reports/closed-loop.json --output reports/closed-loop.png
python -m scripts.benchmark_control --output reports/control-performance.json
python -m examples.navigation_benchmark.efficiency --output reports/early-rejection.json
```

`efficiency` 在接近制动边界的固定输入上，成对比较完整候选预测与先拒绝不安全候选的完整控制调用；同时验证停车命令。它用于隔离提前拒绝的收益，不是旧版本完整控制器的耗时对比。一般场景性能另由 `scripts.benchmark_control` 测量。

报告保存在调用方指定路径；仓库内的实测数据及限制见 [本轮结果](../../reports/navigation/README.md)。性能对照应在相同环境、相同预算与负载下运行；不要把一次成功当作普遍成功保证。RRT 随机种子可重复，墙钟预算和系统调度仍会导致边缘样本差异。

方法参考 [OMPL benchmark](https://ompl.kavrakilab.org/benchmark.html)、[DWA 原论文](https://publications.ri.cmu.edu/the-dynamic-window-approach-to-collision-avoidance) 和 [Nav2 Regulated Pure Pursuit](https://github.com/ros-navigation/navigation2/tree/main/nav2_regulated_pure_pursuit_controller)。窄道采样参考 [Bridge Test](https://adacomp.org/wp-content/uploads/2016/01/icra03.pdf)，实现按点云表面构造候选中点，仍逐边验证矩形。
