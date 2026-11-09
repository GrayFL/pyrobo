# EB-DWA 开发前基线

基线提交 `d1adc1b70ffa11967bc0b9be1ae46d6e54d5d54d`，2026-11-09。完整 pytest：**375 passed / 0 failed / 0 skipped**，684.87 s，3 条既有弃用警告。真实导入来自当前四库源码；详细版本和每次延迟见 [机器可读基线](../benchmarks/navigation/eb_dwa_baseline.json)。

## 性能

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m benchmarks.navigation.teb --repeats 30 --output <report.json>`。完整测试结束后串行测量，本任务没有其他测试或仿真并行；用户原有 Web 服务保持运行。两算法同状态、同点云、1 × 0.5 m / offset 0.25 m 矩形、2.4 s 预测。包含局部优化、物理预测和制动检查，不含全局或独立最终安全门。

| 模型 | 点数 | 算法 | p50 / p95 / p99 · ms |
| --- | --- | --- | --- |
| DifferentialDrive | 1000 | dwa | 50.12 / 51.18 / 52.11 |
| DifferentialDrive | 1000 | teb | 60.09 / 69.61 / 72.86 |
| DifferentialDrive | 10000 | dwa | 166.64 / 171.00 / 177.15 |
| DifferentialDrive | 10000 | teb | 70.47 / 82.45 / 84.38 |
| Ackermann | 1000 | dwa | 36.22 / 36.36 / 36.41 |
| Ackermann | 1000 | teb | 61.65 / 69.23 / 71.63 |
| Ackermann | 10000 | dwa | 94.87 / 97.68 / 99.46 |
| Ackermann | 10000 | teb | 72.04 / 81.30 / 84.50 |

## Web 场景基线

通过原有 `SceneSession`、Supervisor、SafetyGuard 和独立实体碰撞真值执行。固定 RRT seed=7，保存展开的有效配置和逐周期轨迹。除切换全局/局部算法外保留源场景设置。DWA/TEB 使用各自场景默认预测时间；上方固定输入性能测试则统一预测窗口。此处每组合一次，不是稳定性或普遍成功率证明。

| 场景 | 全局 | 局部 | 到达步数 | 碰撞 | 全部指令通过验证 |
| --- | --- | --- | --- | --- | --- |
| teb_narrow_road.json | rrt_connect | dwa | 329 | False | True |
| teb_narrow_road.json | rrt_connect | teb | 432 | False | True |
| teb_narrow_road.json | voronoi | dwa | 341 | False | True |
| teb_narrow_road.json | voronoi | teb | 406 | False | True |
| teb_offset_detour.json | rrt_connect | dwa | 117 | False | True |
| teb_offset_detour.json | rrt_connect | teb | 157 | False | True |
| teb_offset_detour.json | voronoi | dwa | 148 | False | True |
| teb_offset_detour.json | voronoi | teb | 187 | False | True |

窄路用例是差速、额外全局净距固定为 0 的已保存回归；长车体用例为阿克曼，矩形 1 × 0.5 m / offset 0.2 m。真实 margin 均保持 0.08 m。这里没有验证原始自适应全局预算配置，也没有声称 TEB 的不稳定问题已解决。

原始本地证据：`assets/refactor-eb_dwa/baseline/` 下的 suite.log/xml、latency.json、展开的 `*-scene.json` 和完整 `*-trace.json`；8 次场景回放均保留，没有筛掉失败。测试与测量由现有 AIO 环境运行，没有安装或升级依赖。
