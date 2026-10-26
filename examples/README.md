# 底盘与导航例程

在已安装 pyrobo 的环境运行；从源码运行时在仓库根目录执行。依赖与坐标约定见[根 README](../README.md)。

## 两种全局规划器 × 两种底盘 × 两种矩形

```bash
python -m pyrobo.simulation --planner rrt_connect --model differential --footprint compact
python -m pyrobo.simulation --planner voronoi --model differential --footprint long
python -m pyrobo.simulation --planner rrt_connect --model ackermann --footprint compact
python -m pyrobo.simulation --planner voronoi --model ackermann --footprint long --plot /tmp/ackermann.png
```

入口实现是 `pyrobo/simulation/__main__.py`，完整闭环运行器是 `run_navigation`。默认障碍场景为交错墙；全局输出有序航点，PathTracker 提前给出转角后的前视目标，DWA 输出模型原生指令，力驱动 plant 生成实际状态并反馈。阿克曼的规划跟踪余量随最小转弯半径配置，终点有沿目标朝向的直线入射段。

## 手动倒车后恢复导航

```bash
python -m pyrobo.simulation --model differential --scenario handoff
python -m pyrobo.simulation --model ackermann --planner voronoi --scenario handoff --plot /tmp/handoff.png
```

第 35–64 帧交给手动反向速度指令，其余帧由 DWA 控制。实测路径进度会先增加、倒车时减少、恢复时再次增加；全局路径未被截断。`SimulationRun.manual` 标识命令来源，`commands` 保存实际选定的指令，底盘速度 / 转角 / 加速度状态一直保留。

## 纯手动动力学

```bash
python -m examples.manual_control
```

[manual_control.py](manual_control.py) 对两种模型依次运行速度话题跟踪、滑行、倒车和急停，打印实际速度、转角与力。`track` 零目标、`coast` 和 `estop` 含义不同，分别为平滑停、零主动力滑行和有限力紧急制动。可以把最后的 ChassisState 直接传给 Navigator 接着自主导航。

这些例程不依赖 Notebook 内核或 scipykit。使用 `--plot` 时需 Matplotlib；绘图输出路径由调用方指定。算法测试另覆盖稀疏 / 重复航点、交叉路径、转动扫掠、窄通道和预测 / 执行一致性。
