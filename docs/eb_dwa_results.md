# EB-DWA 验收与性能记录

开发分支 `codex/eb-dwa`；原始输入、逐周期轨迹、截图、测试日志和失败尝试保存在 `assets/refactor-eb_dwa/phase5/`。全部汇总与源码摘要见 [机器可读结果](../benchmarks/navigation/eb_dwa_results.json)，复现命令和指标口径见 [基准说明](../benchmarks/navigation/README.md)。

最终回归覆盖 **487 项，均通过，无跳过**。全量运行为 486 passed /1 failed，941.07 s；3 条既有弃用警告。唯一失败是本次新增锁步测试仍假设 EB 默认启用墙钟预算。保留显式 60 ms 拒绝断言并补充新默认 None 可接纳断言后，2 项预算契约复测通过；未更改控制代码或锁步约束。两次原始 JUnit 均保留，最终通过节点与当前完整收集的 487 项逐项匹配。默认预算调整前的最终 EB 专项 93 项通过。没有安装依赖，也没有修改 RSim、MujocoSim、graphmap。

## 预算与已知限制

默认 `time_budget_s=None`：每次更新最多 350 次真实残差计算、每个拓扑最多 24 次主评估、最多 8 个周期续算。节点、拓扑次数、验证区间和缓存均有上限。时间预算仍可显式设置；它是软截止，不保证一次 SciPy/几何调用能立即中断。

60 ms 实验 12 次中 8 次到达、4 次安全停车但未到达，因此没有继续将 60 ms 作为默认。默认窄路串行对照中，EB-DWA 局部 p95 为 649 / 680 ms（RRT / Voronoi），最大约 879 ms，明显超过 100 ms 控制周期。异步 RSim 执行端仍必须检查命令期限；本报告的同步 Web 结果不能证明硬实时。EB 只修复局部 XY 几何，没有完整 SE(2) 搜索、任意汽车换向规划或动态障碍预测。

## 保存场景

20/20 达到各自的保存预期，均无实体碰撞。17 个到达；另外 3 个验证预算耗尽、候选制动或局部几何失败。点云冻结/恢复场景到达，但陈旧期间仍报告 `input_stale` 和 `all_commands_verified=False`。几何不可行实验观察 8 秒，要求最终至少连续停稳 1 秒，并记录几何拒绝与上层重规划；不要求从起点起永不移动，也不证明全局无路。

| 场景 | 步数 | 预期 | 实际到达 | 上层规划次数 |
| --- | ---: | --- | --- | ---: |
| ackermann_corner | 136 | arrived | True | 2 |
| ackermann_impossible | 80 | infeasible | False | 12 |
| ackermann_reverse | 90 | arrived | True | 0 |
| budget | 8 | stopped | False | 0 |
| corner | 155 | arrived | True | 0 |
| corner_wall | 211 | arrived | True | 0 |
| differential_spin | 128 | arrived | True | 0 |
| dynamic_obstacle | 149 | arrived | True | 1 |
| long_car_gate | 129 | arrived | True | 0 |
| long_wall | 178 | arrived | True | 1 |
| narrow_road | 324 | arrived | True | 1 |
| no_candidate | 1 | braking | False | 0 |
| obtuse | 165 | arrived | True | 0 |
| single_obstacle | 154 | arrived | True | 1 |
| stale_cloud | 107 | arrived | True | 0 |
| straight | 129 | arrived | True | 0 |
| two_obstacles | 171 | arrived | True | 1 |
| u_trap | 348 | replan | True | 1 |
| wall_clearance | 464 | arrived | True | 6 |
| zigzag | 122 | arrived | True | 0 |

本组关闭 EB 和 RRT 墙钟预算，保留计算次数预算；预算故障例保留原有专门设置。靠墙终点用例经过退让与重规划才到达，不能描述为全程无恢复动作。

## 窄路重复与浏览器

| 全局规划器 | seed | 到达步数 | 最小观测矩形净空 · m |
| --- | ---: | ---: | ---: |
| rrt_connect | 7 | 324 | 0.01822 |
| rrt_connect | 11 | 418 | 0.02146 |
| rrt_connect | 23 | 319 | 0.02067 |
| voronoi | 7 | 321 | 0.02085 |
| voronoi | 11 | 321 | 0.02085 |
| voronoi | 23 | 321 | 0.02085 |

以上 6/6 到达，全部命令验证、无碰撞。Voronoi 不使用 seed，其三行是相同确定性路网的重复。保留原始窄路地图、起终点、0.5 × 0.5 m 差速车体、0.08 m margin 与动力学；额外全局净距显式为 0。这不是任意车型或任意额外膨胀配置都可通过的证明。

真实 Edge 扩展测试通过，2260 个自动导航快照安全，包含差速弯道、汽车倒车/直角以及两种规划器的窄路；另复验最终默认配置。保存/导出/导入、速度参数、算法切换和 390 px 手机视口均通过；桌面、运行中参考曲线和手机截图已实际查看。Pygame 实际 headless CLI 30 步无碰撞/fault；实际 replay CLI 的汽车弯道和三个故障窗口均达成预期。

## 60 ms 对照（未筛去失败）

| 场景 | 全局 | seed | 到达 | 步数 | 碰撞 |
| --- | --- | ---: | --- | ---: | --- |
| ackermann_corner | rrt_connect | 7 | False | 800 | False |
| ackermann_corner | rrt_connect | 11 | True | 161 | False |
| ackermann_corner | rrt_connect | 23 | True | 184 | False |
| ackermann_corner | voronoi | 7 | True | 195 | False |
| ackermann_corner | voronoi | 11 | True | 152 | False |
| ackermann_corner | voronoi | 23 | True | 136 | False |
| narrow_road | rrt_connect | 7 | True | 451 | False |
| narrow_road | rrt_connect | 11 | True | 419 | False |
| narrow_road | rrt_connect | 23 | True | 503 | False |
| narrow_road | voronoi | 7 | False | 1000 | False |
| narrow_road | voronoi | 11 | False | 1000 | False |
| narrow_road | voronoi | 23 | False | 1000 | False |

该组用于发现预算敏感性，运行时有其他功能测试并行，不用于串行性能比较。失败均保存完整场景和轨迹。默认配置采用计算次数预算，不能将这里的 60 ms 失败隐藏到平均成功率中。

## 串行性能

以下两个实验均在本任务的测试、浏览器和其他仿真结束后串行运行。用户原有 8000 服务保持运行；BLAS 单线程。固定状态延迟：同一实测状态序列、2.4 s 窗口、完整 1000/10000 点、1 × 0.5 m / offset .25 m 车体，预热一次后测 30 次。局部调用含预测和制动验证，不含全局和独立最终安全门；该直道工况主要测试 EB 缓存，不代表每周期修复的开销。

| 模型 | 点数 | 算法 | p50 / p95 / max · ms | Python 峰值 · MiB |
| --- | ---: | --- | --- | ---: |
| DifferentialDrive | 1000 | dwa | 49.98 / 50.56 / 52.97 | 11.77 |
| DifferentialDrive | 1000 | teb | 60.17 / 69.90 / 74.49 | 0.11 |
| DifferentialDrive | 1000 | eb_dwa | 63.36 / 82.70 / 83.24 | 11.57 |
| DifferentialDrive | 10000 | dwa | 167.59 / 169.46 / 169.77 | 119.57 |
| DifferentialDrive | 10000 | teb | 70.41 / 82.85 / 84.82 | 0.11 |
| DifferentialDrive | 10000 | eb_dwa | 177.53 / 197.41 / 198.44 | 117.02 |
| Ackermann | 1000 | dwa | 36.07 / 36.36 / 37.04 | 5.11 |
| Ackermann | 1000 | teb | 61.63 / 69.33 / 71.97 | 0.11 |
| Ackermann | 1000 | eb_dwa | 49.34 / 69.16 / 70.23 | 5.13 |
| Ackermann | 10000 | dwa | 94.63 / 95.73 / 96.04 | 46.65 |
| Ackermann | 10000 | teb | 72.14 / 81.45 / 84.69 | 0.11 |
| Ackermann | 10000 | eb_dwa | 108.57 / 128.71 / 129.48 | 46.67 |

Python 内存另用 tracemalloc 测量，关闭墙钟预算以免追踪开销导致提前退出；不是进程 RSS。以下相同窄路/种子 7 的闭环使用各算法的默认控制配置（完整参数随场景保存），两个现有全局搜索均参与。所有样本保留，不以失败样本的短运行时间比较“更快”。

| 全局 | 局部 | 到达 | 步数 | 路长 · m | 局部 p50 / p95 / max · ms | 全局 p95 · ms |
| --- | --- | --- | ---: | ---: | --- | ---: |
| rrt_connect | dwa | True | 329 | 16.14 | 77.3 / 98.0 / 106.7 | 380.9 |
| voronoi | dwa | True | 341 | 17.16 | 58.6 / 99.4 / 108.1 | 697.1 |
| rrt_connect | teb | True | 424 | 16.08 | 65.3 / 90.6 / 94.0 | 384.0 |
| voronoi | teb | True | 406 | 17.05 | 62.6 / 85.5 / 95.1 | 723.3 |
| rrt_connect | eb_dwa | True | 324 | 16.11 | 108.5 / 648.6 / 810.7 | 382.3 |
| voronoi | eb_dwa | True | 321 | 16.45 | 133.8 / 680.0 / 878.5 | 724.2 |

## 指标与证据范围

本次串行六组对照均到达；EB-DWA 使用 324 / 321 步，DWA 为 329 / 341 步，TEB 为 424 / 406 步。EB-DWA 的计算耗时明显更高，不能从更少仿真步数推断更快的真实时间导航，也不能据此宣称普遍优于现有算法。实际轨迹和速度图保存在 `assets/refactor-eb_dwa/phase5/comparison.png`，已生成并查看。

JSON 另含 EB/DWA/速度参考/全周期耗时、活动节点、缓存/优化比例、候选数与可行率、最小净空、曲率和变化率、直道/弯道速度、制动起始距离、横向误差及转向速率。误差需区分全局路线与本周期 EB 曲线：EB 起点每周期重锚，后一项不是固定曲线的长期跟踪精度。净空按实际位姿对当时观测点云查询；实体碰撞由独立世界模型检查。

完整失败档案包含初始终点转弯、预算导致的反复重规划、短接入段过约束、靠墙到达恢复，以及浏览器暂停起步导致的原地转向停滞误判。最终只对航向误差持续减小计入 EB 转向进展；来回摆动不能无限推迟重规划。实验中尝试过缩短预测窗口和改变缓存接入切线，未解决问题的版本已撤回。

矩阵/扩展浏览器使用显式 None 预算；其后只把默认预算由 .06 改为 None。最终全量 pytest、预算契约补测、串行对照和默认浏览器复验均在该默认配置上执行。源码与每组输入摘要分别保留，不混淆测试版本。
