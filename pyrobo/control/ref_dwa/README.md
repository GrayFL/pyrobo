# ref_dwa

经典 DWA 的最小开发基线。`optimizer.py` 中 `velocity_window()` 实现速度上限与
单周期加速度窗口的交集；`objective()` 只含目标朝向、障碍净空、前进速度三项。
`controller.py` 是独立类，接入当前局部目标，私有 `rollout.py` 管候选物理预测与制动可行性。
内部采样、验证和选优可独立修改，没有跨算法适配基类。没有旧 DWA 的路径评分、
转向状态、停滞恢复、换向惩罚或额外响应采样锚点。

数学依据：[Fox、Burgard、Thrun，The Dynamic Window Approach to Collision Avoidance (1997)](https://www.cs.cmu.edu/~dfox/abstracts/colli-ieee.abstract.html)。
原论文的机器人模型与轨迹障碍距离在这里适配为矩形车体和有限力预测的净空；
并非论文实现的逐行复刻。三项评分均归一化后转为最小化代价，见 `objective()`。

默认 `RefDWAConfig()` 与 `Footprint()`；`towing_dwa` 是独立源码副本。
输入、输出、直接调用、坐标约定和 Web 测试见[开发接口](../README.md)。
