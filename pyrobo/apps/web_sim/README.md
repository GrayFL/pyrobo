# Web 运动实验室

## 局部规划算法开发基线

「局部规划」额外提供 `ref_dwa`、`ref_teb`、`towing_dwa`、`towing_teb`。
这四项使用经典基础目标函数，代码与旧 DWA/TEB 的导航策略独立；
核心函数、资料来源和输入输出说明见[局部规划开发接口](../../control/README.md)。
各目录独立实现 `config/controller/optimizer`，不依赖通用算法适配基类。
新 TEB 的局部预测显示本周期首段命令执行及制动；完整优化 band 留在控制器 `last_band`，
不将首段速度恒定外推整个规划时域来评估 band。

- 四者均支持差速与阿克曼，算法不绑定车体形状。车型、长宽、中心偏移及物理参数独立设置。
- 切换算法保留当前车体配置；未提供几何参数时统一使用 `Footprint()` 默认值。
- 四者各有同名配置表，TEB 面板及预测时长编辑当前算法。新 TEB 默认预算留空，仅限制求解次数。
- 选择后点「应用并重置」再导航。修改 Python 核心函数后需重启后端、刷新页面；修改参数只需应用。
- 加载场景保留其明确保存的车型和尺寸；例如长 0.79 m、宽 0.54 m、中心偏移 −0.295 m
  的车体可搭配任一算法。

`tests/browser_reference_optimizers.mjs` 检查四个入口、切换算法保留车体、towing 阿克曼导航、
参数隔离、保存/导入/导出、空场地闭环到达及 390px 手机布局，运行方式与其他浏览器专项相同。
源码自动重载的启动命令见[开发接口文档](../../control/README.md#车体与-web-测试)。

车体上的 `⊕` 标记为驱动轴中点（公共位姿原点），紫色点为几何中心，浅色虚线车体为初始车位。
差速车从静止按 A/D 原地旋转时，轴心坐标不变，几何中心按 `abs(offset_x)` 半径绕轴心运动；
`offset_x=-0.295` 时半径为0.295 m。`tests/browser_rotation_center.mjs` 使用真实 A/D 输入，
同时检查服务器位姿和实际绘制的矩形顶点，避免仅凭示意轮子位置判断旋转中心。

FastAPI + Uvicorn 提供页面、场景库和 WebSocket；浏览器用 Canvas 绘图，不需要 Node 构建、CDN 或 pygame。每个浏览器窗口独立运行一辆车，局域网设备共享保存的场景库。

「局部规划」支持 DWA、TEB、EB-DWA。EB-DWA 显示紫色几何参考及淡紫色软跟踪走廊，绿色仍为共享有限力模型的实际候选预测；软走廊不代替矩形碰撞检查。可调期望净空、跟踪前视、节点/迭代/软时间预算、提前减速开关及横向加速度使用比例，完整参数保存在 `config.eb_dwa`。详细诊断包含几何失败原因、复用情况、活动节点、参考速度及 EB/DWA 分项耗时。

场景库同时列出用户保存的场景与随包发布的内置案例。20 个 `eb_dwa_*.json` 通过同一个 `SceneSession` 在网页和 CLI 中重放。「可重复实验」JSON 的可选 `experiment` 字段包含粗糙 `reference_path`、初始完整运动状态、按 `time_s` 排序的 `events`、预期结果和最大步数。事件仅支持 `add_boxes`、`freeze_cloud`、`resume_cloud`；新增障碍同时进入实体碰撞世界与观测，冻结点云时保持原采集时间。重置恢复初始世界与事件队列，保存和导出保留实验定义。

固定参考只作为首次全局路线供局部算法处理，可以故意穿过障碍；失败后的重规划仍调用所选真实全局规划器。不能把实验参考当作无碰撞证书。编辑初始位置或目标时，应同步修改参考的首尾点或在实验设置中清除 `reference_path`。新建空白场景会清除实验设置。

replay 默认使用实验的 `max_steps`（普通场景为 700），并按 `expected` 判断是否符合验收预期。预算场景只观察首次 8 个周期；无可行候选场景只观察首周期的有限力制动。故障通过不等于到达：报告始终分别保留 `arrived`、`all_commands_verified` 与 `expected_met`。声明的点云冻结事件允许出现 `input_stale`，但必须触发经过检查的急停；恢复后到达仍不会把这些过期周期记为已验证输入。

```bash
python -m pyrobo.apps.web_sim.replay pyrobo/apps/scenes/eb_dwa_corner.json --steps 300
python -m pyrobo.apps.web_sim.replay pyrobo/apps/scenes/eb_dwa_ackermann_reverse.json --steps 150
```

## 启动与局域网访问

在已配置 graphmap 的项目 Python 环境、仓库根目录执行：

```bash
# 已有 FastAPI / Uvicorn / websockets 时无需重复安装
python -m pip install -e '.[web]'
python -m pyrobo.apps.web_sim --host 0.0.0.0 --port 8000
```

本机访问 `http://127.0.0.1:8000`，局域网设备访问 `http://服务器局域网IP:8000`。启动器会打印本机及可检测到的 LAN 地址。设备需能够访问服务器对应端口；`0.0.0.0` 是监听地址，浏览器使用服务器实际 IP。

```bash
# 指定启动场景、共享目录、同时运行的实验数
python -m pyrobo.apps.web_sim --port 8000 \
  --scene pyrobo/apps/scenes/polygon_corridor.json \
  --scenes-dir examples/scenes/saved --max-clients 4

# 也可直接由 Uvicorn 启动工厂
uvicorn pyrobo.apps.web_sim.server:create_app --factory --host 0.0.0.0 --port 8000
```

默认最多 8 个独立实验，使用单个 Uvicorn 进程。此例程面向可信局域网，未提供账号登录；保存的场景对访问服务的设备可见。

## 绘制与导航

1. 点击「新建」得到空地图，或打开现有场景。默认示例为交错墙。
2. 选择「矩形障碍」拖动绘制；「自由画墙」可连续画曲线；「折线墙」逐点点击，回车或「完成折线」结束，至少两个点；「多边形」逐点点击，按回车或「完成多边形」闭合。支持凹多边形，拒绝自交多边形。自由画墙允许交叉，按距离筛选鼠标采样点，相邻点拓宽成四边形，节点补八边形接头，保存为若干简单多边形实体。独立「折线墙」保存原始顶点与线宽，不会自动闭合或填充内部。
3. 「放置车辆」设置初始位置；「设置目标」设置目标位置。按住拖动可指定朝向，也能在右侧精确输入 XY 与角度。XY 单位为米，界面角度为度，JSON 中为弧度。
4. 在右侧选择差速 / 阿克曼、RRT-Connect / Voronoi + D* Lite、DWA / TEB，设置任意矩形尺寸及参数。
5. 点击「应用并重置」构建实验，再点击「自动导航」。地图或物理参数发生变化时，显示未应用草稿，应用会从初始位姿重启。初始位置或目标与实体障碍相交时会拒绝应用并说明原因。

运行中选择「点击导航」可立即更换目标，保持实际底盘状态。拖动指定目标朝向；有未应用地图草稿时，会先应用草稿并从初始位置开始。中间航点由现有导航跟踪器连续跟踪，允许人工倒车后恢复进度。

编辑器提供撤销 / 重做、单个障碍删除、清空、适应地图、滚轮缩放与中键平移。手机使用触屏绘制和「移动视图」工具平移；工具栏可横向滑动。点云预览显示实体表面的采样点。碰撞真值使用矩形对实体几何检查，规划器接收表面点云，地图不栅格化。

界面显示全局路径、获选指令的物理预测（含制动尾段）、实际行驶轨迹、前视点、车速、角速度、转角、规划结果、候选数量与安全判定。浅色虚线车体表示本次实验的初始位姿。

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

`backend.local_optimizer` 从注册表选择（默认 `dwa`），各自保存参数；「预测时长」只修改当前算法。TEB 默认预测 2.4 s，可设置节点数、迭代上限和优化软预算，预算留空为仅限制次数。Web 上限为 64 节点、100 次主评估及 1 s 预算，并检查物理预测工作量。手动辅助继续读取 `dwa` 参数。旧 schema_version=1 场景可直接加载，新保存使用 v2 文件与独立地形，补全已注册算法默认值。诊断显示实际算法、节点数、求解状态、残差调用数、代价及是否复用轨迹。详见 [TEB 算法说明](../../control/teb/README.md)。

右侧可直接调整常用参数。「完整参数 JSON」提供所有车型、动力学、DWA、TEB、全局规划器、导航、安全门及手柄参数，编辑后点击「更新参数草稿」，校验通过再应用。`app.width` / `height` / `fps` 是与 Pygame 共用的配置字段；网页尺寸由浏览器决定。

切换车型时，界面使用后端目录中的对应预测时长 / 前视距离与全局转弯余量：差速 1.8 s / 1.3 m，阿克曼 2.4 s / 1.6 m；需要其他值可继续编辑。RRT / Voronoi 的默认软预算为 0.5 s / 2 s。算法仍为 XY 全局规划，复杂阿克曼场景可能因为转弯空间不足而不能到达，诊断状态会保留。

「导航与物理参数」提供**全局额外间距**、**间距策略**和**目标入射距离**。额外间距叠加在车体 `margin` 和点半径之外。默认 `backend.adaptive_clearance=true` 将它作为首选值，在共享时间预算内根据起终点和通道空间尝试减小；车体尺寸、margin 和点半径始终保持原值。选择固定间距时，它恢复为硬约束，过大可能导致 `invalid_start` / `no_path`。

`navigation.goal_approach_distance` 留空时先使用车型默认长度，再按矩形扫掠可用空间缩短；已进入终点接近区域时避免重新绕回车后的入射起点。正数固定长度，`0` 关闭入射段。显式长度不会自动缩短。自动适配仍无法容纳时返回 `invalid_goal_approach`；它不保证二维路线具备阿克曼可达性。

- **场景与地形管理**：顶部「场景库」和「保存场景」打开同一个管理器。选择资源类型、分类目录和保存名称；文件名就是名称，禁止路径字符，大小写及 Unicode 等价重名会提示 409。覆盖必须明确选择；场景覆盖时可以修改名称，原文件随后移除。
- **目录分类**：原厂目录只读，可加载编辑后另存到用户目录。默认用户目录为 `$XDG_DATA_HOME/pyrobo/scenes/`（未设置时为 `~/.local/share/pyrobo/scenes/`），可用 `--scenes-dir` 修改。添加分类时可指定服务器上的文件夹，或留空在 `categories/分类名/` 下创建。目录位置记录到根目录 `.library.json`，重启保留；移除登记不删除磁盘文件。原有随机 ID 文件仍可列出、加载，明确覆盖时转为用户指定名称。
- **独立地形**：`maps/地形名.json` 只含边界、矩形、多边形和折线墙，不含车位、目标或算法参数。场景的 `config.map` 只保存 start/goal，通过 `terrain_ref: "maps/地形名.json"` 引用本目录地形；多个场景可共用一张地形。加载地形保留现有车辆和导航设置，在草稿中检查车位后应用。
- **保存与删除**：新场景自动创建尚不存在的地形；同名地形内容不同时提示先保存地形或换名，避免静默影响其他场景。在「地形」页加载后可覆盖地形；所有引用它的场景下次加载使用新地形。被引用地形不能删除；地图改名用另存方式，原引用保留。场景可单独删除。
- **导出 / 导入 JSON**：场景导出为 `schema_version=2` 的可移植文档，其中 `terrain` 内嵌地形，不依赖服务器文件路径；地图也可单独导出和导入。导入场景会校验并重置实验，导入地形仅更新草稿。磁盘上的引用式场景由 `load_scene_file(path)` 或 replay CLI 加载，跨设备复制目录时保留 `maps/`。
- **将当前车位设为初始位姿**：以当前车位重新测试时使用。保存默认保留明确设置的初始位姿，不保存运行中速度或操作录像。

`Scene` 的内存/编辑器表示继续展开旧版几何字段，统一通过 `Scene.document()` 输出 v2 文档；导出和磁盘文件不会重复存储 `config.map` 中的地形。参数保存包含有效默认值与 RRT 种子。折线记录为 `{"points": [[x,y], ...], "width": 0.3}`，实体碰撞仍使用厚线段解析距离。

### 算法与车辆模板自动发现

全局规划器、局部优化器、运动学模型与车辆模板的注册表位于 `pyrobo.registry`。实现类在自己模块内用装饰器登记，例如：

```python
from pyrobo.registry import register_local_optimizer

@register_local_optimizer("my_dwa", config=MyDWAConfig, label="我的 DWA", family="dwa", teleop_config=True)
class MyDWA:
    ...
```

对应入口为 `register_global_planner`、`register_kinematic_model`、`register_vehicle_template`。`config` 是可默认构造的 dataclass；车型和车辆模板省略 `config` 时使用自身 dataclass。局部工厂遵守[局部算法装配协议](../../control/README.md)；`teleop_config=True` 表示接收单独的手动 DWA 配置。`family` 选择现有快捷参数面板，其他参数统一通过完整参数 JSON 修改；新增算法族可不指定 family。全局工厂签名为 `(config, footprint)`；开启自适应间距需要兼容其配置约定。新运动学模型还须提供实际控制/动力学所需协议，注册本身不代替这些实现。

应用启动时导入 `pyrobo.planning`、`pyrobo.control`、`pyrobo.models.kinematics` 下的模块/包，新增包需从其 `__init__.py` 导入实现；独立插件可通过 `PYROBO_PLUGINS=my_package,my_other_package` 指定。导入即注册，重复名称报错。核心注册表不会扫描磁盘、启动线程或导入前端；插件只从服务器配置导入。`/api/catalog` 提供选项、标签、算法族和默认值，Web 与 Pygame CLI 从目录生成选项，无须改 HTML。

车体模板含默认小车（长0.5/宽0.5）、牵引车（长0.79/宽0.54/中心偏移−0.295）和长车体（长1.0/宽0.5）。模板只设置几何参数，不绑定车型或算法；改长宽、margin、offset_x 后显示「自定义」，场景保存完整值。

### 雷达发现未知障碍

绘图工具的「新障碍类型」可选已知地形或未知障碍，矩形、多边形、折线与自由画墙均支持；「切换已知/未知」点击已有障碍可转换。未知障碍以橙色显示，保存在场景 `unknown_obstacles` 中，独立地形文件不包含它们。

实体世界始终包含全部障碍。全局规划、局部控制与独立安全门只收到已知地图点云加当前雷达扫描；扫描使用真实位姿、有限量程、有限束数、射线首次回波和遮挡，世界坐标蓝点表示实际回波。观测带当前仿真时间；重置清除扫描状态。当前实现是单帧理想二维雷达，不积累历史点云或隐式执行 SLAM；离开视野后不保留额外地图记忆，稀疏扫描也不能保证未采样表面的安全。

原厂「雷达发现绕障」示例使用独立的「空场地」地图，机器人接近前方未知障碍后重新规划。`experiment.add_boxes` 保持原有语义：事件障碍立即加入已知地图；需要雷达发现的实体请使用 `unknown_obstacles`。冻结观测实验保留原时间戳，恢复后重新扫描。

已提供完整用例 [polygon_corridor.json](../scenes/polygon_corridor.json)。保存目录默认被 Git 忽略；需要回归跟踪的 JSON 可导出到 `examples/scenes/` 下。

[TEB 偏移长车体绕墙](../scenes/teb_offset_detour.json)可直接导入，或执行 `python -m pyrobo.apps.web_sim.replay pyrobo/apps/scenes/teb_offset_detour.json --steps 350`。它使用后轴参考点的 1 × 0.5 m 阿克曼车体，中心前移 0.2 m，碰撞 margin 保持 0.08 m。浏览器专项脚本 `tests/browser_teb.mjs` 验证算法切换、参数往返、两车型到达和手机布局，使用与现有 `browser_web_sim.mjs` 相同的 Playwright 环境变量。

[TEB 窄路差速连续弯道](../scenes/teb_narrow_road.json)保留场景库“窄路”的墙体、起终点和车体安全边距，设置全局额外净距为固定 0，以单独验证局部控制。导入后可切换两个全局规划器，或执行 `python -m pyrobo.apps.web_sim.replay pyrobo/apps/scenes/teb_narrow_road.json --steps 650`。该场景检查差速减速时线速度和角速度保持相同比例，并走完整条路线；它不覆盖自适应全局预算和阿克曼倒车规划。

[窄道阿克曼001](../scenes/narrow_ackermann_001.json) 保留用户地图、起终点、0.5 × 0.5 m 车体及 0.08 m 碰撞边距，使用 Voronoi / D* Lite、0.08 m 全局额外间距、1.0 m 目标入射段。可通过「导入 JSON」复现，或运行下面的回放命令。这个用例也覆盖阿克曼静止时 DWA 同代价候选的转角选择。

[自动适配版本](../scenes/narrow_ackermann_adaptive.json)保留原始约 0.406 m 首选额外间距，开启自动间距和自动入射段；RRT / Voronoi 均已完成实际回放与浏览器到达验证，不需要缩小车体或手工试入射长度。对应浏览器脚本为 [browser_narrow_scene.mjs](../../../tests/browser_narrow_scene.mjs)。

交互例程限制地图单边不超过 200 m、表面采样最多 100000 点、最多 256 个矩形、256 个多边形与 256 条折线（多边形和折线共 4096 顶点），场景文件最多 2 MB。也会拒绝过大的候选采样、积分工作量及规划预算，避免输入错误拖垮交互服务。这些是例程限制，未修改核心库范围。

## 无界面回放

```bash
python -m pyrobo.apps.web_sim.replay pyrobo/apps/scenes/polygon_corridor.json --steps 700
python -m pyrobo.apps.web_sim.replay pyrobo/apps/scenes/narrow_ackermann_001.json --steps 330
python -m pyrobo.apps.web_sim.replay examples/scenes/saved/*.json \
  --steps 900 --report /tmp/navigation-report.json
```

每个用例使用同一个 SceneSession、点云、动力学和独立实体碰撞检查。结果包含是否到达、碰撞、指令是否全部通过安全校验、仿真步数、规划状态和最小间距。普通场景要求到达且无碰撞、指令全部校验通过；显式 experiment 按 `expected` 验收，退出码由 `expected_met` 决定。`infeasible` 表示该有界实验出现几何拒绝和重规划，最后持续停稳至少 1 秒，允许此前沿已验证安全段推进；它不是全局无路的数学证明。过期输入仍保持 `all_commands_verified=False`，即使预期的制动与恢复事件通过。没有目标的纯手动场景会明确报错。规划预算依赖机器时间，跨硬件 / 库版本不保证逐位数值一致。

## 验证与结构

```bash
python scripts/test.py quick
python scripts/test.py related --area web --area registry
# 仅静态页面改动：
python scripts/test.py related --area web-ui
```

浏览器回归脚本为 [browser_web_sim.mjs](../../../tests/browser_web_sim.mjs)。在另一个终端启动服务，使用已有 Chromium / Chrome / Edge 和可选的 `playwright-core` 执行：

```bash
npm install --prefix /tmp/pyrobo-browser playwright-core --dry-run
npm install --prefix /tmp/pyrobo-browser playwright-core
PLAYWRIGHT_MODULE=/tmp/pyrobo-browser/node_modules/playwright-core/index.mjs \
BROWSER_EXECUTABLE=/path/to/chromium \
WEB_SIM_URL=http://127.0.0.1:8000 node tests/browser_web_sim.mjs
```

脚本实际绘制矩形、自由画墙、多边形和折线、摆放车辆与目标、保存加载、导入导出、验证两车型导航与急停，并在移动视口派发触摸事件验证屏幕摇杆。默认截图和导出用例保存在 `/tmp/pyrobo-web-browser`，可通过 `WEB_SIM_ARTIFACTS` 修改。Node 与 Playwright 仅用于开发验证，启动前端不需要它们。

`pyrobo/apps/sim_common/` 是与 Pygame 共享的配置、Session 和 Worker；`world.py` 增加多边形实体，`scene.py` 管理格式与观测，`storage.py` 管理目录和原子持久化，`server.py` 处理传输，`static/` 为静态前端，`replay.py` 为测试用例入口。算法与物理仍在 pyrobo 核心库中。工作线程独占仿真，ASGI 事件循环负责连接与消息；固定 dt，耗时规划使仿真变慢，不连续追赶补帧。

服务生命周期与 WebSocket 接口参考 FastAPI 官方 [lifespan 文档](https://fastapi.tiangolo.com/advanced/events/) 和 [WebSocket 文档](https://fastapi.tiangolo.com/advanced/websockets/)。

新增管理器专项为 `tests/browser_discovery_library.mjs`，使用相同环境变量。必须指向测试专用 `--scenes-dir`：它在「浏览器分类」中创建、覆盖和删除固定名称的测试资源。覆盖动态目录选项、模板/自定义、未知障碍转换、扫描绕行、重名、目录、场景改名覆盖、地形覆盖、导入导出和390px视口。
