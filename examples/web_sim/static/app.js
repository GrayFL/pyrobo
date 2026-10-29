import { Editor, clone, inside, boxes, segmentDistance } from "./editor.js";
const $ = (id) => document.getElementById(id),
  rad = (degrees) => (degrees * Math.PI) / 180,
  deg = (radians) => (radians * 180) / Math.PI;
let draft = null,
  applied = null,
  state = null,
  socket = null,
  dirty = false,
  serial = 0,
  generation = 0,
  busy = false,
  focused = true,
  neutral = false,
  advancedDirty = false;
let history = [],
  future = [],
  keys = new Set(),
  virtual = { active: false, throttle: 0, turn: 0 },
  pending = new Map();
const hints = {
  navigate: "点击目标立即导航；拖动指定到达朝向",
  robot: "点击放置初始车辆；按住拖动指定车头朝向",
  goal: "点击设置场景目标；拖动指定到达朝向",
  rect: "拖动绘制矩形障碍",
  brush: "按住连续画墙，支持触屏绘制",
  polyline: "逐点点击绘制折线墙；回车或「完成折线」结束；Esc 取消",
  polygon: "逐点点击；回车或「完成多边形」封闭；Esc 取消",
  erase: "点击删除一个障碍，自由画墙可分段删除",
  pan: "拖动平移地图；滚轮缩放",
};
function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}
function setDirty(value) {
  dirty = value;
  editor.dirty = value;
  $("edit-status").textContent = value
    ? "草稿未应用 · 将从初始位姿重新开始"
    : "已应用";
  $("edit-status").style.color = value ? "#b67b2b" : "";
  editor.draw();
}
function checkpoint() {
  history.push(clone(draft));
  if (history.length > 30) history.shift();
  future = [];
}
function change(mutator, { fit = false } = {}) {
  if (!draft || busy) return;
  checkpoint();
  mutator(draft);
  if (!dirty) action("paused").catch(() => {});
  setDirty(true);
  sync(fit);
}
function sync(fit = false) {
  if (!draft) return;
  const c = draft.config;
  $("scene-name").value = draft.name;
  $("scene-description").value = draft.description;
  $("scene-title").textContent = draft.name;
  $("planner").value = c.backend.planner;
  $("model").value = c.backend.model;
  $("body-length").value = c.footprint.length;
  $("body-width").value = c.footprint.width;
  for (const name of ["start", "goal"]) {
    const p = c.map[name] || [0, 0, 0];
    $("" + name + "-x").value = +p[0].toFixed(3);
    $(name + "-y").value = +p[1].toFixed(3);
    $(name + "-yaw").value = +deg(p[2]).toFixed(2);
  }
  ["xmin", "xmax", "ymin", "ymax"].forEach(
    (id, i) => ($(id).value = c.map.bounds[i]),
  );
  $("dt").value = c.app.dt;
  $("manual-speed").value = c.teleop.speed_limit;
  $("horizon").value = c.dwa.prediction_time;
  $("budget").value = c[c.backend.planner].time_budget;
  $("planner-clearance").value = c[c.backend.planner].clearance;
  $("adaptive-clearance").value = c.backend.adaptive_clearance
    ? "adaptive"
    : "fixed";
  $("goal-approach").value = c.navigation.goal_approach_distance ?? "";
  $("mass").value = c.dynamics.mass;
  $("force").value = c.dynamics.drive_force;
  $("policy").value = c.teleop.policy;
  if (!advancedDirty) $("advanced").value = JSON.stringify(c, null, 2);
  editor.setScene(draft, fit);
  $("undo").disabled = !history.length;
  $("redo").disabled = !future.length;
}
function ready() {
  return socket?.readyState === WebSocket.OPEN;
}
function request(type, data = {}) {
  if (!ready()) return Promise.reject(Error("连接已断开，请先重新连接"));
  const id = ++serial;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(Error("操作超时，请检查连接或场景复杂度"));
    }, 15000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ type, id, ...data }));
  });
}
function suppress() {
  neutral = true;
  virtual = { active: false, throttle: 0, turn: 0 };
  $("stick-knob").style.transform = "";
}
function sendControl() {
  if (!ready()) return;
  socket.send(JSON.stringify({ type: "control", input: input() }));
}
async function action(name, value) {
  suppress();
  sendControl();
  return request("action", { action: name, value });
}
async function apply() {
  if (busy) return false;
  if (advancedDirty)
    throw Error("完整参数 JSON 尚未更新，请先点击「更新参数草稿」");
  busy = true;
  suppress();
  sendControl();
  try {
    const result = await request("apply", { scene: draft });
    generation = result.generation;
    draft = result.scene;
    applied = clone(draft);
    state = null;
    editor.state = null;
    setDirty(false);
    sync();
    message("场景已应用，车辆回到初始位姿。点击自动导航或使用摇杆。");
    return true;
  } finally {
    busy = false;
  }
}
function connect() {
  if (
    socket &&
    [WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState)
  )
    return;
  $("connection").textContent = "正在连接";
  $("connection").className = "badge";
  socket = new WebSocket(
    `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`,
  );
  socket.onmessage = async (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "hello") {
      generation = data.generation;
      state = null;
      editor.state = null;
      if (!draft) {
        draft = data.scene;
        applied = clone(draft);
        setDirty(false);
        sync(true);
      } else {
        setDirty(true);
        sync();
        message("已重新连接；草稿已保留，应用后可重新开始。");
      }
      $("connection").textContent = "已连接";
      $("connection").className = "badge online";
      $("reconnect").hidden = true;
      sendControl();
    } else if (data.type === "reply") {
      const call = pending.get(data.id);
      if (call) {
        clearTimeout(call.timer);
        pending.delete(data.id);
        data.ok ? call.resolve(data) : call.reject(Error(data.error));
      } else if (!data.ok) message(data.error, true);
    } else if (data.type === "state" && data.generation === generation) {
      state = data;
      editor.state = data;
      editor.draw();
      $("mode").textContent = data.mode.toUpperCase();
      $("speed").textContent = data.speed.toFixed(2);
      $("turn").textContent =
        `${deg(data.steering).toFixed(1)}° / ${data.omega.toFixed(2)}`;
      $("timing").textContent =
        `${data.time.toFixed(1)} s / ${Math.round(data.elapsed_ms)} ms`;
      $("diagnostics").textContent =
        `${data.status} · ${data.planner_status}\n候选 ${data.feasible}/${data.candidates} · 间距 ${data.clearance === null ? "—" : data.clearance.toFixed(2) + " m"}\n${data.assessment}`;
      $("diagnostics").style.whiteSpace = "pre-line";
      const planningHint = {
        invalid_goal_approach:
          "目标直线入射段与障碍相交，可检查目标朝向或缩短「目标入射距离」。",
        invalid_start:
          "起点不满足全局规划间距，请检查「全局额外间距」与车体安全边距。",
        invalid_goal:
          "目标不满足全局规划间距，请检查目标位置、朝向与额外间距。",
        no_path: "路网未连通起终点，请检查通道间距和全局参数。",
        budget_exhausted: "规划预算耗尽，可增加规划预算或检查地图规模。",
      }[data.planner_status];
      if (planningHint && data.status === "no_path")
        $("diagnostics").textContent += `\n${planningHint}`;
      if (data.fault || data.collided)
        message(
          data.fault || "发生实体碰撞，实验已停止；重置后可重新开始。",
          true,
        );
    }
  };
  socket.onclose = () => {
    $("connection").textContent = "连接已断开";
    $("connection").className = "badge offline";
    $("reconnect").hidden = false;
    keys.clear();
    suppress();
    for (const p of pending.values()) {
      clearTimeout(p.timer);
      p.reject(Error("连接已断开"));
    }
    pending.clear();
    message("连接已断开，实验已停止；本地草稿保留，可导出或重新连接。", true);
  };
  socket.onerror = () =>
    message("无法连接仿真服务，请检查地址和服务状态。", true);
}
async function navigate(p) {
  if (advancedDirty) throw Error("请先更新完整参数 JSON 草稿");
  const pose = state?.pose || draft.config.map.start;
  p[2] ??= Math.atan2(p[1] - pose[1], p[0] - pose[0]);
  if (dirty) {
    change((s) => (s.config.map.goal = p));
    if (!(await apply())) return;
    await action("auto");
  } else {
    await action("goal", p);
    draft.config.map.goal = p;
    applied.config.map.goal = clone(p);
    sync();
  }
  message("导航目标已发送；WASD 或摇杆可随时接管。");
}
const editor = new Editor($("map"), {
  message,
  pose: (tool, p) => {
    if (tool === "navigate") {
      navigate(p).catch((e) => message(e.message, true));
      return;
    }
    const key = tool === "robot" ? "start" : "goal";
    p[2] ??= draft.config.map[key]?.[2] || 0;
    change((s) => (s.config.map[key] = p));
  },
  rect: (box) => change((s) => s.config.map.boxes.push(box)),
  polygons: (polygons) => {
    if (draft.polygons.length + polygons.length > 256) {
      message("最多 256 个多边形；可加大墙体宽度或简化绘图。", true);
      return;
    }
    change((s) => s.polygons.push(...clone(polygons)));
  },
  polyline: (line) => {
    if ((draft.polylines?.length || 0) >= 256) {
      message("最多 256 条折线墙", true);
      return;
    }
    change((s) => {
      s.polylines ??= [];
      s.polylines.push(clone(line));
    });
  },
  erase: (p) => {
    let line = (draft.polylines || []).findLastIndex((l) =>
      l.points
        .slice(1)
        .some(
          (q, j) => segmentDistance(p, l.points[j], q) <= l.width / 2 + 0.05,
        ),
    );
    if (line >= 0) {
      change((s) => s.polylines.splice(line, 1));
      return;
    }
    let i = draft.polygons.findLastIndex((poly) => inside(p, poly));
    if (i >= 0) {
      change((s) => s.polygons.splice(i, 1));
      return;
    }
    i = boxes(draft).findLastIndex((poly) => inside(p, poly));
    if (i >= 0) change((s) => s.config.map.boxes.splice(i, 1));
  },
});

// Browser inputs are refreshed independently of rendering and backend planning.
function input() {
  let pad = [0, 0, false],
    source = "keyboard";
  const cfg = draft?.config.joystick;
  if (cfg?.enabled && navigator.getGamepads) {
    try {
      const devices = [...navigator.getGamepads()].filter(Boolean),
        g = devices[cfg.device_index];
      if (g) {
        const axis = (index) => {
          const v = g.axes[index];
          if (v === undefined) return 0;
          return Math.abs(v) <= cfg.deadzone
            ? 0
            : (Math.sign(v) * (Math.abs(v) - cfg.deadzone)) /
                (1 - cfg.deadzone);
        };
        const t = axis(cfg.throttle_axis) * (cfg.invert_throttle ? -1 : 1),
          r = axis(cfg.turn_axis) * (cfg.invert_turn ? -1 : 1);
        const valid =
          Math.max(cfg.throttle_axis, cfg.turn_axis) < g.axes.length &&
          (cfg.deadman_button < 0 || cfg.deadman_button < g.buttons.length);
        const enabled =
          valid &&
          (cfg.deadman_button < 0
            ? !!(t || r)
            : g.buttons[cfg.deadman_button].pressed);
        pad = enabled ? [t, r, true] : pad;
        $("gamepad-status").textContent = valid
          ? `手柄：${g.id} · 按住按钮 ${cfg.deadman_button}`
          : "手柄轴或按钮映射无效";
      } else
        $("gamepad-status").textContent = window.isSecureContext
          ? "手柄：未连接，连接后按一下按钮"
          : "手柄：浏览器可能要求 HTTPS；触屏摇杆可直接使用";
    } catch {
      $("gamepad-status").textContent = "浏览器不允许读取手柄，请使用屏幕摇杆";
    }
  }
  let [throttle, turn, engaged] = pad;
  if (keys.size) {
    throttle =
      Number(keys.has("w") || keys.has("arrowup")) -
      Number(keys.has("s") || keys.has("arrowdown"));
    turn =
      Number(keys.has("a") || keys.has("arrowleft")) -
      Number(keys.has("d") || keys.has("arrowright"));
    engaged = true;
  } else if (virtual.active) {
    ({ throttle, turn } = virtual);
    engaged = true;
    source = "virtual";
  } else if (engaged) source = "gamepad";
  if (neutral && !engaged) neutral = false;
  if (
    neutral ||
    !focused ||
    document.hidden ||
    dirty ||
    busy ||
    advancedDirty
  ) {
    throttle = turn = 0;
    engaged = false;
  }
  return {
    throttle,
    turn,
    engaged,
    source,
    focused: focused && !document.hidden,
  };
}
setInterval(sendControl, 75);
function lostFocus() {
  focused = false;
  keys.clear();
  suppress();
  sendControl();
  action("paused").catch(() => {});
}
window.addEventListener("blur", lostFocus);
window.addEventListener("focus", () => {
  focused = true;
  suppress();
  sendControl();
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) lostFocus();
  else {
    focused = true;
    suppress();
    sendControl();
  }
});
window.addEventListener("gamepaddisconnected", () => {
  suppress();
  sendControl();
  action("paused").catch(() => {});
});
const editable = (target) =>
  ["INPUT", "SELECT", "TEXTAREA"].includes(target?.tagName) ||
  target?.isContentEditable;
document.addEventListener("focusin", (e) => {
  if (editable(e.target)) {
    keys.clear();
    sendControl();
  }
});
const driveKeys = [
  "w",
  "s",
  "a",
  "d",
  "arrowup",
  "arrowdown",
  "arrowleft",
  "arrowright",
];
window.addEventListener("keydown", (e) => {
  if (editable(e.target) || $("library").open) return;
  const k = e.key.toLowerCase();
  if (driveKeys.includes(k)) {
    e.preventDefault();
    keys.add(k);
    sendControl();
    return;
  }
  if (e.repeat) return;
  if (k === "enter" && ["polygon", "polyline"].includes(editor.tool)) {
    editor.finishPolygon();
    return;
  }
  if (k === "escape") {
    editor.vertices = [];
    editor.cancel();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && k === "z") {
    e.preventDefault();
    $(e.shiftKey ? "redo" : "undo").click();
    return;
  }
  const name = {
    g: "auto",
    m: "manual",
    " ": "paused",
    e: "estop",
    r: "release",
    home: "reset",
  }[k];
  if (name) {
    e.preventDefault();
    runAction(name);
  }
});
window.addEventListener("keyup", (e) => {
  keys.delete(e.key.toLowerCase());
  sendControl();
});
let stickPointer = null;
function moveStick(e) {
  const b = $("joystick").getBoundingClientRect(),
    x = (e.clientX - b.left - b.width / 2) / (b.width / 2 - 14),
    y = (e.clientY - b.top - b.height / 2) / (b.height / 2 - 14),
    n = Math.max(1, Math.hypot(x, y));
  virtual = { active: true, throttle: -y / n, turn: -x / n };
  $("stick-knob").style.transform =
    `translate(${(x / n) * 44}px,${(y / n) * 44}px)`;
  sendControl();
}
$("joystick").addEventListener("pointerdown", (e) => {
  if (stickPointer !== null) return;
  stickPointer = e.pointerId;
  $("joystick").setPointerCapture(e.pointerId);
  moveStick(e);
});
$("joystick").addEventListener("pointermove", (e) => {
  if (stickPointer === e.pointerId) moveStick(e);
});
for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
  $("joystick").addEventListener(name, () => {
    stickPointer = null;
    virtual = { active: false, throttle: 0, turn: 0 };
    $("stick-knob").style.transform = "";
    sendControl();
  });

async function runAction(name) {
  try {
    if (["auto", "manual"].includes(name) && dirty) {
      if (!(await apply())) return;
    }
    await action(name);
    if (name === "reset") {
      state = null;
      editor.state = null;
    }
    message(
      {
        auto: "自动导航已开启",
        manual: "手动模式：WASD / 摇杆驾驶",
        paused: "已请求暂停",
        estop: "急停已锁存；先解除，再选择模式",
        release: "已解除到暂停状态",
        reset: "已回到本次实验的初始状态",
      }[name],
    );
  } catch (e) {
    message(e.message, true);
  }
}
document
  .querySelectorAll("[data-action]")
  .forEach((b) => (b.onclick = () => runAction(b.dataset.action)));
document.querySelectorAll("[data-tool]").forEach(
  (b) =>
    (b.onclick = () => {
      editor.setTool(b.dataset.tool);
      document
        .querySelectorAll("[data-tool]")
        .forEach((x) => x.classList.toggle("selected", x === b));
      $("tool-hint").textContent = hints[b.dataset.tool];
      $("finish-polygon").hidden = !["polygon", "polyline"].includes(
        b.dataset.tool,
      );
      $("finish-polygon").textContent =
        b.dataset.tool === "polyline" ? "完成折线 ↵" : "完成多边形 ↵";
    }),
);
$("finish-polygon").onclick = () => editor.finishPolygon();
$("fit").onclick = () => editor.fit();
$("points-toggle").onclick = () => {
  editor.points = !editor.points;
  $("points-toggle").classList.toggle("selected", editor.points);
  editor.draw();
};
$("apply").onclick = () => apply().catch((e) => message(e.message, true));
$("reconnect").onclick = connect;
for (const [id, other] of [
  ["undo", "redo"],
  ["redo", "undo"],
])
  $(id).onclick = () => {
    const from = id === "undo" ? history : future,
      to = id === "undo" ? future : history;
    if (!from.length || busy) return;
    to.push(clone(draft));
    draft = from.pop();
    advancedDirty = false;
    setDirty(true);
    sync();
    action("paused").catch(() => {});
  };
$("clear-obstacles").onclick = () =>
  change((s) => {
    s.polygons = [];
    s.polylines = [];
    s.config.map.boxes = [];
  });
$("current-as-start").onclick = () => {
  if (state) change((s) => (s.config.map.start = clone(state.pose)));
};
$("scene-name").onchange = () =>
  change((s) => (s.name = $("scene-name").value.trim() || "未命名场景"));
$("scene-description").onchange = () =>
  change((s) => (s.description = $("scene-description").value));
function number(id) {
  const value = Number($(id).value);
  if (!$(id).value.trim() || !Number.isFinite(value))
    throw Error("请输入有限数值");
  return value;
}
const setters = {
  "body-length": (c, v) => (c.footprint.length = v),
  "body-width": (c, v) => (c.footprint.width = v),
  dt: (c, v) => (c.app.dt = v),
  "manual-speed": (c, v) => (c.teleop.speed_limit = v),
  horizon: (c, v) => (c.dwa.prediction_time = v),
  budget: (c, v) => (c[c.backend.planner].time_budget = v),
  "planner-clearance": (c, v) => (c[c.backend.planner].clearance = v),
  mass: (c, v) => (c.dynamics.mass = v),
  force: (c, v) => (c.dynamics.drive_force = v),
};
for (const p of ["start", "goal"])
  for (const [i, axis] of ["x", "y", "yaw"].entries())
    setters[`${p}-${axis}`] = (c, v) => {
      c.map[p] ??= [0, 0, 0];
      c.map[p][i] = i === 2 ? rad(v) : v;
    };
["xmin", "xmax", "ymin", "ymax"].forEach(
  (id, i) => (setters[id] = (c, v) => (c.map.bounds[i] = v)),
);
for (const [id, set] of Object.entries(setters))
  $(id).onchange = () => {
    try {
      const v = number(id);
      change((s) => set(s.config, v));
    } catch (e) {
      message(e.message, true);
    }
  };
$("model").onchange = () => {
  const model = $("model").value;
  change((s) => {
    const c = s.config;
    c.backend.model = model;
    const car = model === "ackermann";
    c.dwa.prediction_time = car ? 2.4 : 1.8;
    c.dwa.lookahead = c.navigation.lookahead = car ? 1.6 : 1.3;
    const clearance =
      0.12 +
      (car
        ? (0.4 * c.ackermann.wheelbase) / Math.tan(c.ackermann.max_steering)
        : 0);
    c.rrt_connect.clearance = c.voronoi.clearance = clearance;
  });
};
$("goal-approach").onchange = () => {
  try {
    const value = $("goal-approach").value.trim()
      ? number("goal-approach")
      : null;
    change((s) => (s.config.navigation.goal_approach_distance = value));
  } catch (e) {
    message(e.message, true);
  }
};
$("adaptive-clearance").onchange = () => {
  const enabled = $("adaptive-clearance").value === "adaptive";
  change((s) => (s.config.backend.adaptive_clearance = enabled));
};
$("planner").onchange = () => {
  const planner = $("planner").value;
  change((s) => (s.config.backend.planner = planner));
};
$("policy").onchange = async () => {
  const value = $("policy").value;
  if (dirty) {
    change((s) => (s.config.teleop.policy = value));
    return;
  }
  try {
    await action("policy", value);
    draft.config.teleop.policy = value;
    applied.config.teleop.policy = value;
    sync();
  } catch (e) {
    message(e.message, true);
  }
};
$("advanced").oninput = () => {
  advancedDirty = true;
  message("完整参数有未保存修改；点击「更新参数草稿」后再应用。");
};
$("advanced-update").onclick = async () => {
  try {
    const config = JSON.parse($("advanced").value);
    const result = await request("validate", {
      scene: { ...clone(draft), config },
    });
    advancedDirty = false;
    change((s) => (s.config = result.scene.config));
    message("参数已校验并更新草稿；应用后生效。");
  } catch (e) {
    message(e.message, true);
  }
};
function scenePayload() {
  if (advancedDirty) throw Error("请先更新完整参数 JSON 草稿");
  return clone(draft);
}
$("save").onclick = async () => {
  try {
    const response = await fetch("/api/scenes", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(scenePayload()),
      }),
      result = await response.json();
    if (!response.ok) throw Error(result.detail);
    message(`已保存「${result.name}」到共享场景库；初始位姿与参数均已保存。`);
  } catch (e) {
    message(e.message, true);
  }
};
$("export").onclick = () => {
  try {
    const blob = new Blob([JSON.stringify(scenePayload(), null, 2)], {
        type: "application/json",
      }),
      url = URL.createObjectURL(blob),
      link = document.createElement("a");
    link.href = url;
    link.download = `${draft.name.replace(/[/\\:*?"<>|]/g, "_") || "scene"}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    message("场景 JSON 已导出，可使用 replay 命令重放。");
  } catch (e) {
    message(e.message, true);
  }
};
async function loadScene(scene) {
  if (busy) return;
  checkpoint();
  draft = scene;
  advancedDirty = false;
  setDirty(true);
  sync(true);
  await apply();
  editor.fit();
}
$("import-button").onclick = () => $("import-file").click();
$("import-file").onchange = async (e) => {
  if (busy) return;
  busy = true;
  suppress();
  sendControl();
  try {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 2000000) throw Error("文件超过 2 MB");
    const parsed = JSON.parse(await file.text());
    const result = await request("apply", { scene: parsed });
    checkpoint();
    draft = result.scene;
    generation = result.generation;
    applied = clone(draft);
    advancedDirty = false;
    state = null;
    editor.state = null;
    setDirty(false);
    sync(true);
    message("场景已导入并重置。");
  } catch (error) {
    message(error.message, true);
  } finally {
    e.target.value = "";
    busy = false;
  }
};
$("new-scene").onclick = () => {
  change(
    (s) => {
      s.name = "新场景";
      s.description = "";
      s.polygons = [];
      s.polylines = [];
      s.config.map = {
        preset: "custom",
        bounds: [-5, 5, -4, 4],
        start: [-3, 0, 0],
        goal: [3, 0, 0],
        boxes: [],
      };
    },
    { fit: true },
  );
  message("空白场景已准备好；绘制地图或放置车辆后应用。");
};
$("library-open").onclick = async () => {
  try {
    const response = await fetch("/api/scenes");
    if (!response.ok) throw Error("场景库加载失败");
    const scenes = await response.json();
    $("scene-list").replaceChildren();
    if (!scenes.length) $("scene-list").textContent = "还没有保存的场景。";
    for (const s of scenes) {
      const row = document.createElement("div"),
        info = document.createElement("div"),
        title = document.createElement("strong"),
        detail = document.createElement("small"),
        button = document.createElement("button");
      row.className = "scene-item";
      title.textContent = s.name;
      detail.textContent = `${new Date(s.modified * 1000).toLocaleString()} · ${s.description || "无说明"}`;
      button.textContent = "加载";
      button.onclick = async () => {
        try {
          const r = await fetch(`/api/scenes/${s.id}`);
          if (!r.ok) throw Error("场景读取失败");
          await loadScene(await r.json());
          $("library").close();
          message("已从场景库加载，车辆回到保存的初始位姿。");
        } catch (e) {
          message(e.message, true);
        }
      };
      info.append(title, detail);
      row.append(info, button);
      $("scene-list").append(row);
    }
    $("library").showModal();
  } catch (e) {
    message(e.message, true);
  }
};
$("library-close").onclick = () => $("library").close();
connect();
