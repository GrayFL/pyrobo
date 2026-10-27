# 运动学模型与共享力驱动动力学

`core/commands.py` 定义模型指令，`models/kinematics/` 定义运动学，`models/dynamics/` 提供 `ChassisDynamics`。旧 `kinematics.py` / `dynamics.py` 是兼容导出。DWA、手动控制和 `Simulator2D` 共用同一 NumPy 批量积分核心。实现参考用户原型 [ChassisDynamics](../assets/ChassisDynamics/chassis_sim_design.md)，保留原型的两类自由度、离散逆伺服和隐式阻力设计。

## 模型与指令

| 模型 | 指令 | 实际偏航 | 参考点 |
| --- | --- | --- | --- |
| `DifferentialDrive(track=0.4)` | `WheelCommand(left, right)`，轮地速 m/s | 车身偏航力矩积分得到 ω | 驱动轴中点 |
| `Ackermann(wheelbase=0.5)` | `AckermannCommand(speed, steering)`，m/s、rad | `ω = v*tan(δ)/wheelbase` | 后轴中点 |

差速轮组满足 `v=(left+right)/2`、`ω=(right-left)/track`；能代表两轮或等效的左右轮组。阿克曼是单轨自行车运动学，适用于等效四轮转向；不模拟真实两轮的平衡。原地可以转动转向机构，但车身不能旋转；倒车时同一转角产生反向偏航。

两者都支持 `model.command(speed, turn, mode=...)`，其中 `turn` 在差速是偏航目标，在阿克曼是转角目标。消息适配使用 `model.from_twist(Twist2D(v,w))`；阿克曼按**指令速度**求 `atan(L*w/v)`，零速非零偏航抛出错误。原生转角指令在零速仍有效。模型将参考指令限制在自身速度、轮速 / 转角和偏航上限内；非法实测状态报错。

`Footprint.offset_x` 显式区分矩形中心与运动参考点，不能把后轴速度直接当作前方车体中心的横向零速度。

## 状态与调用

```python
from pyrobo import Ackermann, ChassisDynamics, ChassisState, pose2d

plant = ChassisDynamics(Ackermann())
state = ChassisState(pose2d(0, 0, frame="map", body="rear_axle"))
command = plant.model.command(0.6, 0.3)
result = plant.step(state, command, dt=0.1)
state = result.state
print(state.velocity, state.steering, result.force, result.torque, result.flags)
```

状态显式传入、返回，不在 plant 中隐式持有。`pose` 是 graphmap.Pose，`velocity` 是 Twist2D；另含 `steering`、`steering_rate`、纵向 / 偏航 / 转向加速度和时间戳。保留这些字段才能无损切换控制来源。不能两套仿真器同时积分并覆盖位姿。

| `mode` | 行为 |
| --- | --- |
| `track` | 速度 / 转角伺服，主动补偿阻力，再施加有限力 |
| `coast` | 主动力和力矩为零；摩擦、风阻和阿克曼回中弹簧继续作用 |
| `soft_stop` | 零目标的平滑伺服，保留加速度 / jerk 整形 |
| `estop` | 最大可用制动力，禁止越过零速变成反向推进；阿克曼尽量保持当前转角 |

`step(..., command_age=...)` 使用同一仿真时钟，指令年龄超过 `timeout`（默认 0.3 s）转为 soft_stop；长周期也逐子步检查。急停模式不因超时解除，是否持续锁定由调用方负责。DWA 预测假定每周期更新指令，主动检查自己的紧急制动尾段。

## 共享物理

广义轴为 `M*qddot = U - K*q - D(qdot)`。纵向 M=质量、K=0；差速转向 M=车身偏航惯量、K=0；阿克曼转向 M=转向机构惯量，K 随车速平方增加。后者积分转角和转角速度，再推导车身 ω，不能再独立积分 ω。

`Drag` 返回有符号阻力：`coulomb*tanh(v/epsilon) + linear*v + quadratic*abs(v)*v`。平方项对倒车也反向，近零平滑化避免符号抖动。隐式轴积分使用有括区间的 Newton 解，纯阻力不能反向加速。这里不是具有静摩擦死区的库仑接触模型。

伺服根据速度误差 / 时间常数得到净加速度，限制加速、制动及 jerk；转角采用阻尼二阶伺服。离散逆计算维持该加速度所需的力，经过电机限幅及轮端共享摩擦预算，再积分真正可实现的运动。未饱和候选直接使用离散逆的解析结果；饱和候选才批量求隐式轴方程。位姿采用每个物理子步的中点 twist 精确圆弧积分。

正负力并不固定代表加速 / 制动：`F*v > 0` 是驱动，`F*v < 0` 是制动。倒车反向必须先减速，再由力推进；软停可补偿阻力，滑行则完全无主动力。

`DynamicsConfig` 默认质量 20 kg、驱动力 / 制动力 50 / 70 N、偏航力矩 3 Nm、转向力矩 1 Nm、物理子步 ≤0.02 s。摩擦预算为 `μ*m*g`，扣除当前横向需求；差速同时分配左右轮的纵向力和偏航力矩。转角硬限位、最大轮速 / 车速是显式状态保护，可能放松舒适 jerk；相关诊断包括 `reference_limited`、`effort_limited`、`speed_hard_projection`、`steering_stop_reaction`、`linear_jerk_relaxed`、`angular_jerk_relaxed`、`lateral_envelope_exceeded`、`tire_envelope_exceeded`。

这是一套低速、无侧滑的虚拟动力学。参考限速和摩擦分配不是实际瞬态包络不变性的证明；结果提供力、力矩、近似横向加速度、轮胎利用率和诊断。性能参数、模型限制和 DWA 必须与执行端配置一致。

## 批量预测接口

`plant.codec` 封装状态数组的列布局。`integrate(states, commands, dt)` 保持数组返回；传 `diagnostics=True` 得到 `(states, BatchDiagnostics)`，每类标记为 `(N,)`，可对应到具体候选。积分器本身不决定哪些限幅应淘汰候选。

`ReachabilityEstimator` 直接调用这个积分器，不重写阻力或伺服方程。它记录 track 的经验响应范围，保留完整模型/策略命令域并补充响应锚点；不能把下周期实际状态范围当作速度伺服参考范围。物理基线覆盖两种模型及四种命令模式，见 `tests/test_dynamics_baseline.py`。
