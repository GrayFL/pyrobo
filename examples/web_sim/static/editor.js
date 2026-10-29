export const clone = (value) => structuredClone(value);
export const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
export function segmentDistance(p, a, b) {
  const dx = b[0] - a[0],
    dy = b[1] - a[1],
    n = dx * dx + dy * dy;
  if (n === 0) return distance(p, a);
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n));
  return distance(p, [a[0] + t * dx, a[1] + t * dy]);
}
function wallOutlines(line) {
  const result = [];
  for (let i = 1; i < line.points.length; i++) {
    const a = line.points[i - 1],
      b = line.points[i],
      yaw = Math.atan2(b[1] - a[1], b[0] - a[0]),
      r = line.width / 2;
    result.push([
      ...Array.from({ length: 17 }, (_, j) => {
        const t = yaw + Math.PI / 2 + (j * Math.PI) / 16;
        return [a[0] + r * Math.cos(t), a[1] + r * Math.sin(t)];
      }),
      ...Array.from({ length: 17 }, (_, j) => {
        const t = yaw - Math.PI / 2 + (j * Math.PI) / 16;
        return [b[0] + r * Math.cos(t), b[1] + r * Math.sin(t)];
      }),
    ]);
  }
  return result;
}
export function inside(point, polygon) {
  let hit = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i],
      b = polygon[j];
    if (
      a[1] > point[1] !== b[1] > point[1] &&
      point[0] < ((b[0] - a[0]) * (point[1] - a[1])) / (b[1] - a[1]) + a[0]
    )
      hit = !hit;
  }
  return hit;
}
export function boxes(scene) {
  return scene.config.map.boxes.map(([a, b, c, d]) => [
    [a, c],
    [b, c],
    [b, d],
    [a, d],
  ]);
}
export function brushPolygons(points, width) {
  // Independent simple solids support loops and crossings in a hand-drawn wall.
  const selected = [points[0]];
  for (const p of points.slice(1))
    if (distance(p, selected.at(-1)) > Math.max(0.08, width * 0.6)) selected.push(p);
  if (distance(points.at(-1), selected.at(-1)) > 0.01) selected.push(points.at(-1));
  const result = [];
  for (let i = 1; i < selected.length; i++) {
    const a = selected[i - 1],
      b = selected[i],
      n = distance(a, b),
      x = ((-(b[1] - a[1]) / n) * width) / 2,
      y = (((b[0] - a[0]) / n) * width) / 2;
    result.push([
      [a[0] + x, a[1] + y],
      [b[0] + x, b[1] + y],
      [b[0] - x, b[1] - y],
      [a[0] - x, a[1] - y],
    ]);
  }
  for (const p of selected)
    result.push(
      Array.from({ length: 8 }, (_, i) => [
        p[0] + (width / 2) * Math.cos((i * Math.PI) / 4),
        p[1] + (width / 2) * Math.sin((i * Math.PI) / 4),
      ]),
    );
  return result;
}

export class Editor {
  constructor(canvas, hooks) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.hooks = hooks;
    this.scene = null;
    this.state = null;
    this.dirty = false;
    this.tool = 'navigate';
    this.points = false;
    this.center = [0, 0];
    this.zoom = 40;
    this.gesture = null;
    this.vertices = [];
    this.preview = null;
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    canvas.addEventListener('pointerdown', (e) => this.down(e));
    canvas.addEventListener('pointermove', (e) => this.move(e));
    canvas.addEventListener('pointerup', (e) => this.up(e));
    canvas.addEventListener('pointercancel', () => this.cancel());
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const before = this.world(e),
          factor = Math.exp(-e.deltaY * 0.001);
        this.zoom = Math.max(2, Math.min(900, this.zoom * factor));
        const after = this.world(e);
        this.center = [
          this.center[0] + before[0] - after[0],
          this.center[1] + before[1] - after[1],
        ];
        this.draw();
      },
      { passive: false },
    );
  }
  resize() {
    const r = this.canvas.getBoundingClientRect(),
      d = window.devicePixelRatio || 1;
    this.width = r.width;
    this.height = r.height;
    this.canvas.width = Math.round(r.width * d);
    this.canvas.height = Math.round(r.height * d);
    this.ctx.setTransform(d, 0, 0, d, 0, 0);
    this.draw();
  }
  setScene(scene, fit = false) {
    this.scene = scene;
    if (fit) this.fit();
    else this.draw();
  }
  fit() {
    if (!this.scene) return;
    const [a, b, c, d] = this.scene.config.map.bounds;
    this.center = [(a + b) / 2, (c + d) / 2];
    this.zoom = Math.min((this.width - 70) / (b - a), (this.height - 80) / (d - c));
    this.draw();
  }
  pixel(p) {
    return [
      (p[0] - this.center[0]) * this.zoom + this.width / 2,
      this.height / 2 - (p[1] - this.center[1]) * this.zoom,
    ];
  }
  world(e) {
    const r = this.canvas.getBoundingClientRect();
    return [
      (e.clientX - r.left - this.width / 2) / this.zoom + this.center[0],
      -(e.clientY - r.top - this.height / 2) / this.zoom + this.center[1],
    ];
  }
  within(p) {
    const [a, b, c, d] = this.scene.config.map.bounds;
    return p[0] >= a && p[0] <= b && p[1] >= c && p[1] <= d;
  }
  clamp(p) {
    const [a, b, c, d] = this.scene.config.map.bounds;
    let m = ['brush', 'polyline'].includes(this.tool)
      ? Number(document.querySelector('#brush-width').value) / 2
      : 0;
    return [Math.max(a + m, Math.min(b - m, p[0])), Math.max(c + m, Math.min(d - m, p[1]))];
  }
  cancel() {
    this.gesture = null;
    this.preview = null;
    this.draw();
  }
  setTool(tool) {
    this.tool = tool;
    this.vertices = [];
    this.cancel();
  }
  down(e) {
    if (!this.scene || this.gesture) return;
    if (e.button === 2) {
      this.vertices = [];
      this.cancel();
      return;
    }
    if (e.button !== 0 && e.button !== 1) return;
    const p = this.world(e),
      pan = e.button === 1 || this.tool === 'pan';
    if (!pan && !this.within(p)) return;
    this.canvas.setPointerCapture(e.pointerId);
    this.gesture = {
      id: e.pointerId,
      start: this.clamp(p),
      last: this.clamp(p),
      pixels: [e.clientX, e.clientY],
      center: [...this.center],
      pan,
      stroke: [this.clamp(p)],
    };
  }
  move(e) {
    const p = this.world(e);
    document.querySelector('#coordinates').textContent =
      `X ${p[0].toFixed(2)} / Y ${p[1].toFixed(2)} m`;
    const g = this.gesture;
    if (!g || g.id !== e.pointerId) return;
    if (g.pan) {
      this.center = [
        g.center[0] - (e.clientX - g.pixels[0]) / this.zoom,
        g.center[1] + (e.clientY - g.pixels[1]) / this.zoom,
      ];
      this.draw();
      return;
    }
    g.last = this.clamp(p);
    if (this.tool === 'brush' && distance(g.last, g.stroke.at(-1)) > 0.04) g.stroke.push(g.last);
    this.preview = g;
    this.draw();
  }
  up(e) {
    const g = this.gesture;
    if (!g || g.id !== e.pointerId) return;
    this.canvas.releasePointerCapture(e.pointerId);
    this.gesture = null;
    this.preview = null;
    if (g.pan) {
      this.draw();
      return;
    }
    const p = this.clamp(this.world(e));
    if (['polygon', 'polyline'].includes(this.tool)) {
      if (!this.vertices.length || distance(p, this.vertices.at(-1)) > 0.01) this.vertices.push(p);
      this.draw();
      return;
    }
    if (['robot', 'goal', 'navigate'].includes(this.tool)) {
      const yaw =
        distance(g.start, p) * this.zoom > 7
          ? Math.atan2(p[1] - g.start[1], p[0] - g.start[0])
          : null;
      this.hooks.pose(this.tool, [...g.start, yaw]);
    } else if (this.tool === 'rect') {
      if (
        distance(g.start, p) > 0.08 &&
        Math.abs(g.start[0] - p[0]) > 0.02 &&
        Math.abs(g.start[1] - p[1]) > 0.02
      )
        this.hooks.rect([
          Math.min(g.start[0], p[0]),
          Math.max(g.start[0], p[0]),
          Math.min(g.start[1], p[1]),
          Math.max(g.start[1], p[1]),
        ]);
    } else if (this.tool === 'brush') {
      g.stroke.push(p);
      const width = Number(document.querySelector('#brush-width').value);
      if (Number.isFinite(width) && width >= 0.05 && width <= 3)
        this.hooks.polygons(brushPolygons(g.stroke, width));
    } else if (this.tool === 'erase') this.hooks.erase(p);
    this.draw();
  }
  finishPolygon() {
    const line = this.tool === 'polyline';
    if (this.vertices.length < (line ? 2 : 3)) {
      this.hooks.message(line ? '折线至少需要两个顶点' : '多边形至少需要三个顶点', true);
      return;
    }
    if (line) {
      const width = Number(document.querySelector('#brush-width').value);
      if (!Number.isFinite(width) || width < 0.05 || width > 3) {
        this.hooks.message('线宽应为 0.05–3 m', true);
        return;
      }
      this.hooks.polyline({ points: clone(this.vertices), width });
    } else this.hooks.polygons([this.vertices]);
    this.vertices = [];
    this.draw();
  }
  path(points, color, width = 2, fill = null, close = false) {
    if (!points.length) return;
    const c = this.ctx;
    c.beginPath();
    points.forEach((p, i) => {
      const q = this.pixel(p);
      i ? c.lineTo(...q) : c.moveTo(...q);
    });
    if (close) c.closePath();
    if (fill) {
      c.fillStyle = fill;
      c.fill();
    }
    if (color) {
      c.strokeStyle = color;
      c.lineWidth = width;
      c.stroke();
    }
  }
  arrow(p, color, size = 0.65) {
    const e = [p[0] + size * Math.cos(p[2]), p[1] + size * Math.sin(p[2])];
    this.path([p.slice(0, 2), e], color, 2);
    for (const delta of [-0.5, 0.5])
      this.path(
        [e, [e[0] - 0.15 * Math.cos(p[2] + delta), e[1] - 0.15 * Math.sin(p[2] + delta)]],
        color,
        2,
      );
  }
  robot(p, ghost = false) {
    const f = this.scene.config.footprint,
      c = this.ctx,
      cs = Math.cos(p[2]),
      sn = Math.sin(p[2]);
    const map = ([x, y]) => [p[0] + x * cs - y * sn, p[1] + x * sn + y * cs];
    const corners = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ].map(([x, y]) => map([f.offset_x + (x * f.length) / 2, (y * f.width) / 2]));
    c.save();
    if (ghost) c.setLineDash([5, 4]);
    this.path(corners, ghost ? '#809e89' : '#236d96', 2, ghost ? '#d8e5dbaa' : '#bad8e8', true);
    c.setLineDash([]);
    this.arrow(p, ghost ? '#7e9b83' : '#225d78', Math.max(0.4, f.length * 0.65));
    if (!ghost)
      for (const x of [-1, 1])
        for (const y of [-1, 1]) {
          const center = map([f.offset_x + x * f.length * 0.32, y * f.width * 0.56]),
            a =
              p[2] +
              (x === 1 && this.scene.config.backend.model === 'ackermann'
                ? this.state?.steering || 0
                : 0);
          this.path(
            [
              [center[0] - 0.09 * Math.cos(a), center[1] - 0.09 * Math.sin(a)],
              [center[0] + 0.09 * Math.cos(a), center[1] + 0.09 * Math.sin(a)],
            ],
            '#264e61',
            4,
          );
        }
    c.restore();
  }
  draw() {
    if (!this.ctx || !this.scene) return;
    const c = this.ctx;
    c.clearRect(0, 0, this.width, this.height);
    c.fillStyle = '#e8ede4';
    c.fillRect(0, 0, this.width, this.height);
    const [a, b, d, e] = this.scene.config.map.bounds,
      border = [
        [a, d],
        [b, d],
        [b, e],
        [a, e],
      ],
      step = this.zoom < 12 ? 5 : this.zoom < 25 ? 2 : 1;
    this.path(border, '#adbdad', 1, '#f7f8f1', true);
    c.save();
    c.beginPath();
    border.forEach((p, i) => {
      const q = this.pixel(p);
      i ? c.lineTo(...q) : c.moveTo(...q);
    });
    c.closePath();
    c.clip();
    for (let x = Math.ceil(a / step) * step; x <= b; x += step)
      this.path(
        [
          [x, d],
          [x, e],
        ],
        '#dfe6da',
        0.7,
      );
    for (let y = Math.ceil(d / step) * step; y <= e; y += step)
      this.path(
        [
          [a, y],
          [b, y],
        ],
        '#dfe6da',
        0.7,
      );
    const solids = [...boxes(this.scene), ...this.scene.polygons];
    for (const polygon of solids) this.path(polygon, '#7f9287', 1, '#b0bdb2', true);
    const lines = this.scene.polylines || [];
    c.save();
    c.lineCap = 'round';
    c.lineJoin = 'round';
    for (const line of lines) this.path(line.points, '#95ac9b', line.width * this.zoom);
    c.restore();
    if (this.points) {
      c.fillStyle = '#4b6d59';
      const spacing = this.scene.config.app.point_spacing;
      for (const polygon of [border, ...solids, ...lines.flatMap(wallOutlines)])
        for (let i = 0; i < polygon.length; i++) {
          const p = polygon[i],
            q = polygon[(i + 1) % polygon.length],
            n = Math.ceil(distance(p, q) / spacing);
          for (let k = 0; k < n; k++) {
            const z = this.pixel([p[0] + ((q[0] - p[0]) * k) / n, p[1] + ((q[1] - p[1]) * k) / n]);
            c.fillRect(z[0] - 0.7, z[1] - 0.7, 1.4, 1.4);
          }
        }
    }
    if (this.state && !this.dirty) {
      this.path(this.state.trail, '#599fc9', 2);
      this.path(this.state.path, '#c69b3b', 2.5);
      this.path(this.state.forecast, '#168d78', 2);
      if (this.state.target) {
        const q = this.pixel(this.state.target);
        c.beginPath();
        c.arc(...q, 4, 0, Math.PI * 2);
        c.strokeStyle = '#168d78';
        c.stroke();
      }
    }
    const goal = this.dirty
      ? this.scene.config.map.goal
      : this.state?.goal || this.scene.config.map.goal;
    if (goal) {
      const q = this.pixel(goal);
      c.beginPath();
      c.arc(...q, 8, 0, Math.PI * 2);
      c.strokeStyle = '#c28a2d';
      c.lineWidth = 2;
      c.stroke();
      this.arrow(goal, '#c28a2d');
    }
    const initial = this.scene.config.map.start;
    if (!this.dirty && this.state) {
      this.robot(initial, true);
      this.robot(this.state.pose);
    } else this.robot(initial);
    if (this.vertices.length) {
      c.save();
      if (this.tool === 'polyline') {
        c.lineCap = 'round';
        c.lineJoin = 'round';
        this.path(
          this.vertices,
          '#87ae9277',
          Number(document.querySelector('#brush-width').value) * this.zoom,
        );
      }
      this.path(this.vertices, '#487f61', 2, this.tool === 'polygon' ? '#a5d5b650' : null);
      c.restore();
      for (const p of this.vertices) {
        c.beginPath();
        c.arc(...this.pixel(p), 4, 0, Math.PI * 2);
        c.fillStyle = '#487f61';
        c.fill();
      }
    }
    const g = this.preview;
    if (g) {
      if (this.tool === 'rect')
        this.path(
          [g.start, [g.last[0], g.start[1]], g.last, [g.start[0], g.last[1]]],
          '#577d62',
          1,
          '#97c6a777',
          true,
        );
      else if (this.tool === 'brush') {
        c.lineCap = 'round';
        c.lineJoin = 'round';
        this.path(
          g.stroke,
          '#87ae92',
          Number(document.querySelector('#brush-width').value) * this.zoom,
        );
        c.lineCap = 'butt';
      } else if (['robot', 'goal', 'navigate'].includes(this.tool))
        this.arrow(
          [...g.start, Math.atan2(g.last[1] - g.start[1], g.last[0] - g.start[0])],
          '#477c61',
        );
    }
    c.restore();
    const scale = document.querySelector('#scale');
    scale.textContent = `${step} m`;
    scale.style.width = `${this.zoom * step}px`;
  }
}
