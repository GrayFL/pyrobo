# Web 运动实验室

FastAPI + Uvicorn 提供页面、场景库和 WebSocket；浏览器用 Canvas 绘图，不需要 Node 构建、CDN 或 pygame。每个浏览器窗口独立运行一辆车，局域网设备共享保存的场景库。

## 启动与局域网访问

在已配置 graphmap 的项目 Python 环境、仓库根目录执行：

```bash
# 已有 FastAPI / Uvicorn / websockets 时无需重复安装
python -m pip install -e '.[web]'
python -m examples.web_sim --host 0.0.0.0 --port 8000
```

本机访问 `http://127.0.0.1:8000`，局域网设备访问 `http://服务器局域网IP:8000`。启动器会打印本机及可检测到的 LAN 地址。设备需能够访问服务器对应端口；`0.0.0.0` 是监听地址，浏览器使用服务器实际 IP。

```bash
# 指定启动场景、共享目录、同时运行的实验数
python -m examples.web_sim --port 8000 \
  --scene examples/scenes/polygon_corridor.json \
  --scenes-dir examples/scenes/saved --max-clients 4

# 也可直接由 Uvicorn 启动工厂
uvicorn examples.web_sim.server:create_app --factory --host 0.0.0.0 --port 8000
```

默认最多 8 个独立实验，使用单个 Uvicorn 进程。此例程面向可信局域网，未提供账号登录；保存的场景对访问服务的设备可见。

## 绘制与导航

1. 点击「新建」得到空地图，或打开现有场景。默认示例为交错墙。
2. 选择「矩形障碍」拖动绘制；「自由画墙」可连续画曲线；「折线墙」逐点点击，回车或「完成折线」结束，至少两个点；「多边形」逐点点击，按回车或「完成多边形」闭合。支持凹多边形，拒绝自交多边形。自由画墙允许交叉，按距离筛选鼠标采样点，相邻点拓宽成四边形，节点补八边形接头，保存为若干简单多边形实体。独立「折线墙」保存原始顶点与线宽，不会自动闭合或填充内部。
3. 「放置车辆」设置初始位置；「设置目标」设置目标位置。按住拖动可指定朝向，也能在右侧精确输入 XY 与角度。XY 单位为米，界面角度为度，JSON 中为弧度。
4. 在右侧选择差速 / 阿克曼、RRT-Connect / Voronoi + D* Lite，设置任意矩形尺寸及参数。
5. 点击「应用并重置」构建实验，再点击「自动导航」。地图或物理参数发生变化时，显示未应用草稿，应用会从初始位姿重启。初始位置或目标与实体障碍相交时会拒绝应用并说明原因。

运行中选择「点击导航」可立即更换目标，保持实际底盘状态。拖动指定目标朝向；有未应用地图草稿时，会先应用草稿并从初始位置开始。中间航点由现有导航跟踪器连续跟踪，允许人工倒车后恢复进度。

编辑器提供撤销 / 重做、单个障碍删除、清空、适应地图、滚轮缩放与中键平移。手机使用触屏绘制和「移动视图」工具平移；工具栏可横向滑动。点云预览显示实体表面的采样点。碰撞真值使用矩形对实体几何检查，规划器接收表面点云，地图不栅格化。

界面显示全局路径、DWA 预测（含制动尾段）、实际行驶轨迹、前视点、车速、角速度、转角、规划结果、候选数量与安全判定。浅色虚线车体表示本次实验的初始位姿。

## 手动控制

| 输入 | 行为 |
| --- | --- |
| WASD / 方向键 | 前进、倒车和转向；立即接管自动导航 |
| 屏幕摇杆 | 鼠标或触屏拖动驾驶，松开请求零速 |
| G / 自动导航 | 恢复自动任务；已有草稿时先应用 |
| M / 手动 | 进入手动模式 |
| Space / 暂停 | 有限力制动，保留导航任务 |
| E / 急停 | 锁存急停 |
| R / 解除 | 解除到暂停状态，需要重新操作 |
| Home / 重置 | 恢复已应用场景的初始位姿 |

差速的横向输入是角速度；阿克曼是转角，零线速时只动转向机构。手动控制同样经过 Supervisor → DWA → SafetyGuard → Simulator，支持严格停障与辅助绕障，不直接改写位姿或底盘速度。

控制模式变化后，已保持的按键 / 摇杆须归零再接管。窗口失焦、页面隐藏、手柄断开或心跳过期会暂停。手动松开后不会自动恢复 AUTO，按 G 恢复。断线会终止该窗口的实验；本地草稿保留，重连后应用草稿重新开始。

USB 手柄采用浏览器 Gamepad API，默认轴 1 / 0、按钮 0 使能，完整参数中的 `joystick` 表可调整轴、反向、死区与按钮。浏览器可能要求 HTTPS 或 localhost 才开放手柄 API；普通 LAN HTTP 下可使用触屏摇杆和键盘。设备索引与按钮映射取决于浏览器和硬件，实物手柄需单独验证。参见 [Gamepad API](https://developer.mozilla.org/en-US/docs/Web/API/Gamepad_API)。

## 参数与场景保存

右侧可直接调整常用参数。「完整参数 JSON」提供所有车型、动力学、DWA、全局规划器、导航、安全门及手柄参数，编辑后点击「更新参数草稿」，校验通过再应用。`app.width` / `height` / `fps` 是与 Pygame 共用的配置字段；网页尺寸由浏览器决定。

切换车型时，界面自动选择对应预测时长 / 前视距离与全局转弯余量：差速 1.8 s / 1.3 m，阿克曼 2.4 s / 1.6 m；需要其他值可继续编辑。RRT / Voronoi 的默认软预算为 0.5 s / 2 s。算法仍为 XY 全局规划，复杂阿克曼场景可能因为转弯空间不足而不能到达，诊断状态会保留。

「导航与物理参数」提供**全局额外间距**、**间距策略**和**目标入射距离**。额外间距叠加在车体 `margin` 和点半径之外。默认 `backend.adaptive_clearance=true` 将它作为首选值，在共享时间预算内根据起终点和通道空间尝试减小；车体尺寸、margin 和点半径始终保持原值。选择固定间距时，它恢复为硬约束，过大可能导致 `invalid_start` / `no_path`。

`navigation.goal_approach_distance` 留空时先使用车型默认长度，再按矩形扫掠可用空间缩短；已进入终点接近区域时避免重新绕回车后的入射起点。正数固定长度，`0` 关闭入射段。显式长度不会自动缩短。自动适配仍无法容纳时返回 `invalid_goal_approach`；它不保证二维路线具备阿克曼可达性。

- **保存场景**：保存编辑器中的地图、初始位姿、目标、说明与完整有效参数；服务端会校验。默认目录为 `examples/scenes/saved/`，可用 `--scenes-dir` 修改。文件名使用独立 ID，每次保存新建用例。
- **场景库**：按时间列出已保存用例，加载后恢复初始状态。重启服务后仍可加载。
- **导出 / 导入 JSON**：可下载到当前设备、纳入版本管理或跨设备传递。导出保留当前草稿；导入会校验，成功后重置实验。
- **将当前车位设为初始位姿**：以当前车位重新测试时使用。保存默认保留明确设置的初始位姿。

JSON 格式为 `schema_version = 1`，顶层包含 `name`、`description`、`config`、`polygons` 和 `polylines`。`config.map` 保存边界、轴对齐矩形、初始位姿和可选目标；`polygons` 是填充多边形顶点坐标列表；`polylines` 每条记录为 `{"points": [[x,y], ...], "width": 0.3}`，可表示斜墙、折线和闭合线框。折线实体为带圆头的厚线段，矩形车体与线段的解析最短距离用于实体碰撞，规划点云由墙面采样生成。保存时展开模型 / 算法 / 物理的有效默认值，并保存 RRT 种子，便于重放和参数比较。用例不包含运行中的速度、历史轨迹或人工操作录制。

已提供完整用例 [polygon_corridor.json](../scenes/polygon_corridor.json)。保存目录默认被 Git 忽略；需要回归跟踪的 JSON 可导出到 `examples/scenes/` 下。

[窄道阿克曼001](../scenes/narrow_ackermann_001.json) 保留用户地图、起终点、0.5 × 0.5 m 车体及 0.08 m 碰撞边距，使用 Voronoi / D* Lite、0.08 m 全局额外间距、1.0 m 目标入射段。可通过「导入 JSON」复现，或运行下面的回放命令。这个用例也覆盖阿克曼静止时 DWA 同代价候选的转角选择。

[自动适配版本](../scenes/narrow_ackermann_adaptive.json)保留原始约 0.406 m 首选额外间距，开启自动间距和自动入射段；RRT / Voronoi 均已完成实际回放与浏览器到达验证，不需要缩小车体或手工试入射长度。对应浏览器脚本为 [browser_narrow_scene.mjs](../../tests/browser_narrow_scene.mjs)。

交互例程限制地图单边不超过 200 m、表面采样最多 100000 点、最多 256 个矩形、256 个多边形与 256 条折线（多边形和折线共 4096 顶点），场景文件最多 2 MB。也会拒绝过大的候选采样、积分工作量及规划预算，避免输入错误拖垮交互服务。这些是例程限制，未修改核心库范围。

## 无界面回放

```bash
python -m examples.web_sim.replay examples/scenes/polygon_corridor.json --steps 700
python -m examples.web_sim.replay examples/scenes/narrow_ackermann_001.json --steps 330
python -m examples.web_sim.replay examples/scenes/saved/*.json \
  --steps 900 --report /tmp/navigation-report.json
```

每个用例使用同一个 SceneSession、点云、动力学和独立实体碰撞检查。结果包含是否到达、碰撞、指令是否全部通过安全校验、仿真步数、规划状态和最小间距。全部到达且无碰撞、指令全部校验通过时退出码为 0，其余为 1。没有目标的纯手动场景会明确报错。规划预算依赖机器时间，跨硬件 / 库版本不保证逐位数值一致。

## 验证与结构

```bash
python -m pytest tests/test_web_sim.py tests/test_pygame_session.py tests/test_pygame_ui.py -q
```

浏览器回归脚本为 [browser_web_sim.mjs](../../tests/browser_web_sim.mjs)。在另一个终端启动服务，使用已有 Chromium / Chrome / Edge 和可选的 `playwright-core` 执行：

```bash
npm install --prefix /tmp/pyrobo-browser playwright-core --dry-run
npm install --prefix /tmp/pyrobo-browser playwright-core
PLAYWRIGHT_MODULE=/tmp/pyrobo-browser/node_modules/playwright-core/index.mjs \
BROWSER_EXECUTABLE=/path/to/chromium \
WEB_SIM_URL=http://127.0.0.1:8000 node tests/browser_web_sim.mjs
```

脚本实际绘制矩形、自由画墙、多边形和折线、摆放车辆与目标、保存加载、导入导出、验证两车型导航与急停，并在移动视口派发触摸事件验证屏幕摇杆。默认截图和导出用例保存在 `/tmp/pyrobo-web-browser`，可通过 `WEB_SIM_ARTIFACTS` 修改。Node 与 Playwright 仅用于开发验证，启动前端不需要它们。

`examples/sim_common/` 是与 Pygame 共享的配置、Session 和 Worker；`world.py` 增加多边形实体，`scene.py` 管理格式和持久化，`server.py` 处理传输，`static/` 为静态前端，`replay.py` 为测试用例入口。算法与物理仍在 pyrobo 核心库中。工作线程独占仿真，ASGI 事件循环负责连接与消息；固定 dt，耗时规划使仿真变慢，不连续追赶补帧。

服务生命周期与 WebSocket 接口参考 FastAPI 官方 [lifespan 文档](https://fastapi.tiangolo.com/advanced/events/) 和 [WebSocket 文档](https://fastapi.tiangolo.com/advanced/websockets/)。
