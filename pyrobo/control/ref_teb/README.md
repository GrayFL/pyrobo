# ref_teb

基础单 band TEB。`optimizer.py::solve_band()` 优化内部 SE(2) 位姿与正时间间隔；
`residual()` 保留时间、障碍、非完整运动学、速度、加速度及汽车转弯半径项。
全局路径只初始化 band，不作为贴线代价。固定起终点，使用实测初速度，实际终点加入
停车速度边界。`controller.py` 只从首段提取一条速度命令，不加入旧 TEB 的
多预览、缩速候选、DWA 评分、转向/恢复状态机。

数学依据：[TU Dortmund 官方 optimal_planner](https://github.com/rst-tu-dortmund/teb_local_planner/blob/noetic-devel/src/optimal_planner.cpp)、
[时间边](https://github.com/rst-tu-dortmund/teb_local_planner/blob/noetic-devel/include/teb_local_planner/g2o_types/edge_time_optimal.h)、
[加速度边](https://github.com/rst-tu-dortmund/teb_local_planner/blob/noetic-devel/include/teb_local_planner/g2o_types/edge_acceleration.h)。
只参考数学结构，独立 NumPy/SciPy 实现，无 ROS/g2o/C++ 代码复制。
时间残差使用 `sqrt(weight) * dt`，因此代价为 `weight * sum(dt**2)`。

保留本地稀疏 Jacobian、LSMR、批量几何查询；固定本周期节点数，无多拓扑搜索、
自动重采样或热启动，默认仅限制求解次数。节点/中点的障碍惩罚是软约束，最终命令另由
本目录 `execution.py` 使用 plant 检查首段命令执行一个控制周期及制动的完整矩形扫掠。
不把首段速度保持到整个 band 末尾，也不复用 DWA 候选评估/评分。`last_band` 保留优化
参考，返回的 `trajectory` 是本周期执行加制动的物理预测。失败会停车，不自动尝试其他命令。
`controller.py` 是独立类，没有通用算法基类；可自行修改内部求解/检查/输出流程。

默认 `RefTEBConfig()` 与 `Footprint()`；`towing_teb` 是独立源码副本。
输入、输出、直接调用、坐标约定和 Web 测试见[开发接口](../README.md)。
