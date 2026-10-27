# 共享 DWA 内核

`DWAEngine.optimize(RobotObservation, MotionRequest)` 将导航、严格停障和辅助绕障放入同一批量流水线。返回 `MotionResult`：模型原生命令、带时间的获选轨迹、运动状态、安全评估与诊断。`DynamicWindow.step/stop` 继续提供原有签名和 `DWAResult`。

## 流水线

1. 策略生成允许的参考域与锚点。差速搜索 `(v,ω)` 并批量映射到轮速；阿克曼搜索 `(v,δ)`。手动策略不得超出期望速度幅值或自行反向。
2. ReachabilityEstimator 使用真实 track 积分器试探响应，用于增加锚点。经验响应范围不是严格外包络，不能拿下周期实际速度范围截断参考命令；不确定时保留完整允许域。
3. 用共享 ChassisDynamics 对 `(N,U)` 原生命令预测 `(N,T,S)` 状态。保留实际转角、转角速度与加速度历史。
4. 分别检查保持参考后的制动尾段与执行一周期后立即制动的尾段。未在预算内停住的候选不可行。
5. 矩形点云距离扣除车体边距、点半径和连续区间扫掠上界。半个累计角点运动距离覆盖记录点之间的运动；外接圆仅用于邻域和位移界限。
6. 将命令合法性、策略范围、连续碰撞、制动、可选物理约束组合为 `(N,)` 硬掩码，再向量化评分。只对获选行创建公开命令和轨迹对象。

NAVIGATION 按前视目标、路径、航向、间距和速度评分；差速保留原地转向迟滞。TELEOP_STOP 固定差速 ω 或阿克曼 δ。TELEOP_AVOID 允许有界转向偏离，并约束整段预测中相对用户期望圆弧的横向与航向偏差；用上一命令和平滑方向偏好抑制抖动。

无解时输出真实有限力 estop，并重新检查该停止动作。`safe=False` 表示无法用当前模型与观测证明安全；即使制动也可能存在碰撞风险。安全覆盖允许改变严格停障的角向参考，诊断明确给出 `safety_override` 和原因。运行入口仍需经过独立的 SafetyGuard。

## 数据与扩展

`MotionBatch` 对齐搜索命令、原生命令、预测、约束掩码、间距和代价；`evaluate_candidates` 可检查一个指定矩阵。候选顺序不改变，分块选择保留全局 `candidate_id`。`RolloutBatch` 提供相对时间、保持参考末端索引和逐行有效时间掩码。只有实际执行的第一周期需要下发，其余是滚动预测。

注入 `command_space`、`dynamics`、`reachability`、`policies`、`index` 可替换独立能力。新增更高维车型需配套策略；内核并不将其转换成通用 Twist。`models/dynamics/codec.py` 集中管理当前 S=10 布局，批量积分不逐候选实例化状态。

`DWAConfig(batch_size=0)` 默认整批；正整数可限制每批轨迹工作集。`collect_batch_diagnostics=True` 收集逐行物理标记；`reject_lateral_violation=True` 将横向包络超限作为硬拒绝。执行器限幅默认是实际响应的一部分，不因某一行受限而淘汰整批。

## 路径与参数

`PathTracker` 位于 `navigation.tracking`，旧路径 `control.dwa.path` 保留导出。它保存完整折线，按连续弧长投影并前视，在拐角前切换目标，倒退恢复历史段，交叉点按之前的进度区分分支。Navigator 通过 MotionRequest 传入本帧目标和剩余路径；独立 DWA 调用也可内部维护 tracker。

默认差速预测 1.8 s、前视 1.3 m；阿克曼 2.4 s、1.6 m。记录间隔 ≤0.08 s、物理子步默认 ≤0.02 s、采样参数 7/17、制动预算 4 s。候选数包含零参考与响应锚点，以诊断为准。显式模型/动力学决定运动能力；DWAConfig 的速度/加速度参数用于构造默认能力，angular_acceleration 也用于转向反向代价的归一化。

阿克曼必须提供完整实测 ChassisState。旧差速接口可省略，未提供的加速度按零处理。目标位置/yaw 默认容差 0.18 m/0.10 rad，阿克曼需通过行驶取得末端朝向。局部规划不保证任意停车姿态可达。

策略、运行时与迁移示例见 [架构说明](../../../docs/architecture.md)。项目思路参考 [DWA 原论文](https://publications.ri.cmu.edu/the-dynamic-window-approach-to-collision-avoidance)，实现使用原始点云和力驱动模型预测。
