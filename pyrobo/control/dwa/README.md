# 模型空间 DWA 与可回退路径进度

`DynamicWindow(config, footprint, model=..., dynamics=...).step(pose, velocity, goal, path, obstacles, bounds, dt, chassis=...)` 返回 `DWAResult`。输入输出位姿使用 graphmap，输出指令是差速 `WheelCommand` 或阿克曼 `AckermannCommand`，均可直接交给 `ChassisDynamics` / `Simulator2D`。速度话题和手动操作通过同一模型适配器进入执行端。

## 预测与碰撞

1. 以实测状态和伺服时间常数形成模型参考窗口。差速在左右轮速空间采样，并增加直行 / 纯转向参考；阿克曼在车速和转角空间采样。实际可达速度受共享 plant 的力、加速度、jerk、轮速、转角及转角速度约束。
2. 使用 `ChassisDynamics` 进行 NumPy 批量预测。保留转角、转角速度和加速度历史；同一控制周期的物理子步与实际 plant 完全一致。阿克曼不生成原地旋转候选。
3. 对保持候选后的制动尾段，以及执行一个周期后立即制动的尾段都检查碰撞。使用真正的有限力 estop；超过 `braking_timeout` 仍未停住的候选不可行。
4. 点云障碍转换到每个状态的矩形局部坐标，计算带符号距离，扣除点半径、车体安全边距，检查矩形角点相对物理边界的间距。圆形查询范围只用于筛选邻域，不作为最终碰撞形状。
5. 每段物理子步的中点 twist 为 plant 实际采用的圆弧。将 `(|v_mid| + footprint.radius*|w_mid|)*h` 在记录区间内累计，用半个总运动距离覆盖两端之间的扫掠区域；矩形旋转和倒车也受此检查。
6. 按前视目标、路径、朝向、间距和速度评分。差速保留原地转向滞回；阿克曼始终通过行驶转弯，在末段增加目标朝向成本。倒车候选具有较小惩罚，仍可用于恢复。

`trajectory` 包含制动尾段，不要求整条执行；每次仅发送当前指令，下一周期重新规划。无可行候选时返回 `mode='estop'` 的原生指令；`safe` 表示该真实制动轨迹是否安全且已停住，不代表瞬停。

阿克曼必须传入 `ChassisState`，因为停车时 Twist 不能确定转角。差速为兼容旧观测可省略该参数，此时假设未提供的加速度为零；物理闭环应始终传完整实测状态。`DWAConfig` 中速度 / 加速度参数仅用于创建默认模型 / 动力学，显式传入的模型和动力学是运动能力的唯一来源。

## 路径跟踪

`PathTracker.set_path(PoseArray)` 保存完整路径，`update(pose, lookahead)` 返回 `PathProgress`：前视 `target`、本帧 `remaining`、弧长 `distance`、`remaining_distance`、当前 `segment` 和目标 `target_segment`。

投影在上次弧长附近进行，搜索区间随实际位移增长；窗口正反方向对称，倒车后自然回到更早的段。交叉路径按弧长连续性区分支路，不使用“永远递增的航点索引”，也不删掉已走航点。前视目标沿折线插值，所以在抵达转角之前就会跨到下一段；不会为了每个中间航点停车。重复点、稀疏点、单点路径均受支持。

重新规划调用 `set_path`，离散重定位调用 `reset` 后重新设置路径。Navigator 持有 tracker 并传入进度；独立 DWA 调用内部也有 tracker。`remaining` 是当前视图，原始路径始终保留。

## 参数与范围

默认差速预测 1.8 s、前视 1.3 m；不提供显式 DWAConfig 的阿克曼为 2.4 s、1.6 m。轨迹记录间隔 ≤0.08 s，物理子步默认 ≤0.02 s；默认采样参数 7 / 17，加入锚点与轮速网格后候选数以结果为准。默认位置 / yaw 容差 0.18 m / 0.10 rad，制动预测预算 4 s。`allow_reverse=False` 禁止主动选择反向目标，但已有反向实测速度仍须先物理制动。

`mode` 为 `following`、`turning`、`turn_braking`、`aligning`、`braking` 或 `blocked`。只有差速使用原地转向模式。`reset` 清转向与跟踪状态，不改变任何外部底盘状态。

局部控制可能陷入局部极值；它不为任意汽车停车姿态保证可达。Navigator 为阿克曼终点预留沿目标 yaw 的直线入射段，并在无位置进展时重规划。规划器的跟踪余量和前视距离应匹配车辆尺寸与最小转弯半径。

参考：[Fox、Burgard、Thrun，The Dynamic Window Approach to Collision Avoidance](https://publications.ri.cmu.edu/the-dynamic-window-approach-to-collision-avoidance)。这是结合原始点云、模型控制空间和力驱动预测的项目实现，不是原论文的逐项复刻。
