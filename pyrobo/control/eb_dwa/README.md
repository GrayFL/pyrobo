# Elastic Band + Kinodynamic DWA

实现依据与验收清单见 [设计文档](../../../docs/eb_dwa_design.md)，开发前结果见 [基线报告](../../../docs/eb_dwa_baseline.md)，闭环与性能见 [验收报告](../../../docs/eb_dwa_results.md)。几何修复、空间评分、速度参考及 Web/Pygame/RSim 入口使用同一车辆几何和物理执行链路；测试范围和未解决问题以验收记录为准。

## 控制器与装配

```python
from pyrobo import Ackermann, EBDWAConfig, ElasticBandDWA, Footprint, Navigator

nav = Navigator(model=Ackermann(), footprint=Footprint.long(offset_x=.25),
                controller=EBDWAConfig(prediction_time=2.4, track_lookahead_m=1.6))
```

`ElasticBandDWA.optimize` 实现 `LocalMotionOptimizer`。几何失败时只返回经过检查的制动；不会把未验证的参考交给采样器。DWA 的可选 `score_rollout` 插件读取共享批量预测，采样、有限力积分、完整持有及立即停止后的两项制动检查不变。最终 SafetyGuard 独立复核命令。手动意图委托既有 DWA，并保留共享点云索引。

导航通过 `repairs_global_path` 和 `consume_replan_request()` 两个可选能力保留原全局路线供局部修复；失败时明确请求下一周期全局重规划。未实现这些能力的旧控制器行为不变。几何、动力学候选和最终执行分别给出诊断；`safe=True` 不是任务到达。

差速初始转向也可产生任务进展：仅在 `turning` 且对参考的航向误差累计减小超过 .15 rad 时刷新平移停滞计时。来回摇摆不能不断延后重规划；旧 DWA/TEB 的停滞判定保持不变。

自适应覆盖距离来自共享 plant 的代表性最高前/后速度预测和当前状态制动预测，另加覆盖余量；不是可达空间的严格外包络。缓存稳定的优化曲线，滚动输出只改当前接入段，避免把累计跟踪误差反复写回路径；阿克曼接入长度考虑最小转弯半径。普通低净空软缺陷不会强迫整个长走廊在每次求解中全部优化。

调用方应在命令**实际执行之后**调用 `optimizer.record_execution(command, timestamp)`；下游覆盖后的真实命令才算反馈。传入 `None` 表示反馈缺失。Web/纯仿真自动回传实际执行命令；RSim 内存提供端在测量帧中附加 `executed_command={native, mode}`，消费端按同一源帧转送。无该可选字段的提供端（当前 MuJoCo）仍可运行，但不假设提案已执行，控制平滑项关闭，`eb_execution_feedback=False`。

`reset()` 清除几何、进度、候选、执行反馈和覆盖估计缓存。默认 `time_budget_s=None`，每次更新最多 350 次真实残差计算、每次拓扑最多 24 次求解器主评估，最多 8 周期续算。锁步 RSim 同样要求关闭墙钟预算。可显式设置软时间预算，但准备、单次计算和验证不能被抢占；60 ms 实验出现了安全停车而未到达的情况，不能把它当作稳定的默认值。修复周期可能超过控制周期，见验收中的实际计时。

## 速度参考

`speed_profile=True` 时，`SpeedPlanner` 为每个参考弧长样本生成带行驶方向的速度偏好。曲率限速取横向加速度、摩擦和模型偏航速率的较小约束；汽车另按 `d(arctan(wheelbase*curvature))/ds` 与转向速率限制速度，差速使用偏航加速度限制曲率变化。默认 `speed_lateral_fraction=.3` 只使用部分横向能力，给有限预测窗口和跟踪偏差留余量，不改变真实 plant 的限幅。

加速和减速响应由同一个 `BatchPredictor` 批量计算，包含当前配置的摩擦、驱动力、阻力、伺服和 jerk。表格采样正反速度、两侧转向极值和汽车转向速率；倒车独立计算。减速参考采用普通零速度命令，随后保留预测器的急停尾段，避免只按较短的急停距离规划普通减速。反向距离遍历提前降低弯道/终点前的速度，再从实测车速前向限制加速。仅缓存当前 dt 的正、反两个响应表，reset 清空。

终点速度偏好另按剩余长度与保持时间逐渐下降，避免恒定指令候选靠转向来拖延越过终点。预测时间本身保持配置值，评分仍读取全部保持阶段。终点朝向项只对进入位置容差的候选生效；已完全与目标朝向对齐的直线参考可提前约束朝向，避免不必要的末段转弯。

这些表格是采样近似，不能覆盖任意未来执行器状态；`eb_speed_profile_is_outer_bound=False` 明示这一点。它们影响评分与采样锚点，不替代完整候选预测、矩形扫掠、制动硬拒绝或最终安全门。`speed_profile=False` 可作对照实验，仍保留所有硬验证。

`max_coverage_m` / 活动区、`prediction_time`、跟踪预瞄是不同范围。跟踪距离为 `max(track_lookahead_m, abs(measured_speed)*track_lookahead_time_s)`，受当前参考长度截断；终点加入低速锚点。诊断给出实际前视 `eb_track_lookahead_m`、逐点速度 `eb_speed_profile_mps`、响应表复用和计算耗时，Web 可调整启用状态及横向能力比例。

## 几何模块

`EBPathRepairer.update(observation, global_path, goal=..., coverage_m=...)` 返回 `RepairResult`。成功时 `reference` 是 `EBReference`；失败时它是 `None`，并返回 `no_safe_rejoin`、`curvature_violation`、`collision`、`node_budget` 等明确原因。它不积分底盘、不生成控制命令。

`PathTracker` 保存完整全局路线和可回退进度。检测不仅包含矩形冲突，还包含低于期望的间距、急角、曲率、曲率变化和锯齿。检查完整原始边和旋转包络；相近缺陷合并为活动区。输出沿全局弧长扩展，覆盖进入窗口的全部冲突、重新接入区及尾部。`max_coverage_m` 限制全局路线的检查跨度；绕行后的实际参考长度另记为 `eb_coverage_m`。

重新接入点必须在全部相关冲突之后，并有至少 `rejoin_validation_m` 的安全后续路线。长墙两端可能各产生一个碰撞区间，但中间间距缺陷会将其合并为一个优化区。`preferred_clearance` 是软目标；`minimum_clearance` 是叠加于真实矩形、offset、margin 和点半径之上的硬下限，绝不会自动降低。

优化变量仅为活动节点 XY。起点、远处直线和重新接入后的尾部固定；曲线端点切线固定。三次 Hermite 曲线保证连续切线，残差包括按弧长尺度归一化的平滑、曲率能量与变化、车体间距、局部锚定和退化段惩罚。稀疏 Jacobian 利用四节点局部依赖；矩形距离在正对墙面时不可微，使用中心差分避免单侧扰动产生错误的远墙方向。

Jacobian 结构按节点/活动节点/采样区间缓存，最多 16 份，不缓存距离查询或几何验证结果。汽车重规划后的初始方向误差需要包含两次转弯的接入空间，活动区按转弯半径扩展。到全局终点只剩一个短连接段时，可采用经过实测方向和终点的圆弧切线构造 Hermite 曲线，随后仍验证整段真实曲率与矩形；最终到达依然要求实测位置、yaw 和速度全部满足容差。

优化代价不是安全证明。输出必须通过共享 `PointObstacles.clearances` 的完整矩形和物理边界检查。每个三次曲线区间使用导数上界约束平移、偏航和角点行程，只有间距下界为正才接受；不确定区间细分，超出深度/工作量上限则拒绝。阿克曼另验证整段曲率上界 `1 / minimum_turn_radius`。差速起始方向修正调用同一矩形旋转检查。外接圆仅用于角点运动界限。

无缺陷直线不运行优化。已有参考滚动重锚定并重新验证；点云侵入、覆盖耗尽或显式 `force=True` 触发修复。相同路线、目标、坐标、边界及几何条件下可复用上一曲线；倒退不删除原路径，换目标、路径、配置、明显重定位和时间回退会清除不兼容缓存。

每次更新限制残差评估总数，单个拓扑求解限制迭代数；软时间预算不能抢占 SciPy/点云查询或最终验证。所有真实耗时均记录。低预算下可在最多 `max_refinement_cycles` 个周期中续算，只保存一条有界中间曲线。未验证的中间曲线仅能作为下一次优化初值，绝不能作为输出参考。没有可行输出时返回失败；若已有验证过的初值，则可以明确返回 `validated_fallback`。`eb_solver_pending` 表示仍在续算，不表示参考已安全或任务已完成。

首选 nominal 初始化；失败后才尝试有上限的左右绕行初值。每次尝试的求解状态、实际评估次数、耗时与几何验证结果分别记录。错误全局拓扑和汽车换向规划不由 XY EB 保证解决，上层需要安全停车和重规划。

`EBReference.yaw` 是几何切线；`gear=-1` 时车头方向为切线加 π。左右 tube 是基于矩形净空的保守平移范围，宽度为正、横向误差向左为正。它们仅供软评分；最终控制仍必须验证真实物理轨迹。

## 独立验证

```bash
python -m pytest -q tests/test_eb_geometry.py tests/test_eb_scoring.py
```

测试读取 `pyrobo/apps/scenes/eb_dwa_*.json` 中的相同世界和参考路线。几何测试独立调用修复器；`test_eb_controller.py` 通过正式 `SceneSession` 验证完整执行。当前 DWA 默认评分、TEB、物理积分器和碰撞内核未修改。

## 空间匹配评分

`scoring.project_trajectories` 对整个 `(N,T,2)` 轨迹批次做精确线段投影。时间样本依次处理，每个候选以上一次弧长为中心、按实际位移扩展正反搜索窗口；避免交叉路线跳支。起始端的负向延长量用于识别真实倒退，超过末端的运动不继续获得进度奖励。重复全局节点在建立参考前去重，零长分母另有数值保护。

`trajectory_scores` 的横向、航向和走廊代价使用正向弧长增量做梯形积分；倒退单独惩罚。速度误差使用时间增量积分，进度奖励使用末端与起点的弧长差。另有终点越过惩罚，避免投影到末端后继续高速前冲。各项以独立数组返回，和为总评分。车辆的倒车车头方向是参考切线加 π，不能拿几何切线直接惩罚倒车姿态。

控制平滑项只使用调用方显式传入的 `previous_command` 执行反馈；不私自把上次候选当作已执行。`score_rollout` 读取共享 `RolloutBatch` 的保持指令阶段和真实 codec 速度，排除制动尾段及无效填充。碰撞、两项制动可达性、原生指令合法性等硬掩码仍由共享 DWA 物理评估器负责；评分模块没有修改可行性的权限。
