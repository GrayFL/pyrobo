// Opt-in real-browser integration. See examples/web_sim/README.md for setup.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const base = process.env.WEB_SIM_URL || 'http://127.0.0.1:8090';
const artifacts = process.env.WEB_SIM_ARTIFACTS || '/tmp/pyrobo-web-browser';
await mkdir(artifacts, { recursive: true });
const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.BROWSER_EXECUTABLE || undefined,
  args: ['--no-sandbox'],
});
const errors = [];
const page = await browser.newPage({ viewport: { width: 1512, height: 1100 } });
page.on('pageerror', (error) => errors.push(error.message));
async function applied() {
  await page.waitForFunction(() => document.querySelector('#edit-status').textContent === '已应用');
}
async function pixel(x, y) {
  const r = await page.locator('#map').boundingBox(),
    scale = Math.min((r.width - 70) / 10, (r.height - 80) / 8);
  return [r.x + r.width / 2 + x * scale, r.y + r.height / 2 - y * scale];
}
async function stroke(points) {
  const start = await pixel(...points[0]);
  await page.mouse.move(...start);
  await page.mouse.down();
  for (const p of points.slice(1)) await page.mouse.move(...(await pixel(...p)), { steps: 5 });
  await page.mouse.up();
}
async function exportScene() {
  const download = page.waitForEvent('download');
  await page.locator('#export').click();
  const file = await download;
  const data = JSON.parse(await readFile(await file.path(), 'utf8'));
  return data;
}
try {
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#connection').textContent === '已连接');
  await page.locator('#new-scene').click();
  await page.locator('#adaptive-clearance').evaluate((el) => { el.closest('details').open = true; });
  assert.equal(await page.locator('#adaptive-clearance').inputValue(), 'adaptive');
  await page.locator('#adaptive-clearance').selectOption('fixed');
  assert.equal((await exportScene()).config.backend.adaptive_clearance, false);
  await page.locator('#adaptive-clearance').selectOption('adaptive');
  assert.equal((await exportScene()).config.backend.adaptive_clearance, true);
  await page.locator('[data-tool="rect"]').click();
  await stroke([
    [-0.6, 0.6],
    [0.5, 1.4],
  ]);
  await page.locator('[data-tool="brush"]').click();
  await stroke([
    [-2, 2],
    [-1, 2.5],
    [0.4, 2.5],
  ]);
  await page.locator('[data-tool="polygon"]').click();
  for (const p of [
    [1, 1],
    [2.4, 1],
    [2.4, 2],
    [1.7, 1.4],
  ])
    await page.mouse.click(...(await pixel(...p)));
  await page.locator('#finish-polygon').click();
  await page.locator('[data-tool="polyline"]').click();
  for (const p of [
    [-3.5, 0.5],
    [-3.5, 1.5],
    [-2.7, 1.5],
  ])
    await page.mouse.click(...(await pixel(...p)));
  await page.keyboard.press('Enter');
  await page.locator('[data-tool="robot"]').click();
  await stroke([
    [-3, -2],
    [-2.3, -2],
  ]);
  await page.locator('[data-tool="goal"]').click();
  await stroke([
    [3, -2],
    [3.6, -2],
  ]);
  await page.locator('#scene-name').fill('浏览器 · 地图编辑回归');
  await page.locator('#scene-name').press('Tab');
  let original = await exportScene();
  assert.equal(original.config.map.boxes.length, 1);
  assert.ok(original.polygons.length > 3);
  assert.equal(original.polylines.length, 1);
  assert.equal(original.polylines[0].points.length, 3);
  assert.equal(original.polylines[0].width, 0.3);
  assert.ok(Math.abs(original.config.map.start[0] + 3) < 0.03);
  assert.ok(Math.abs(original.config.map.start[1] + 2) < 0.03);
  await page.locator('#undo').click();
  await page.locator('#redo').click();
  await page.locator('#apply').click();
  await applied();
  await page.locator('#save').click();
  await page.waitForFunction(() =>
    document.querySelector('#message').textContent.startsWith('已保存'),
  );
  const saved = await exportScene();
  await writeFile(`${artifacts}/browser-case.json`, JSON.stringify(saved, null, 2));
  await page.screenshot({ path: `${artifacts}/editor-desktop.png`, fullPage: true });
  await page.locator('#new-scene').click();
  await page.locator('#library-open').click();
  await page
    .locator('.scene-item')
    .filter({ hasText: saved.name })
    .first()
    .getByRole('button', { name: '加载' })
    .click();
  await applied();
  assert.deepEqual(await exportScene(), saved);
  await page.locator('[data-action="auto"]').click();
  await page.waitForFunction(() => document.querySelector('#mode').textContent === 'AUTO');
  await page.waitForFunction(
    () => document.querySelector('#diagnostics').textContent.startsWith('arrived'),
    {},
    { timeout: 25000 },
  );
  assert.equal(await page.locator('#speed').innerText(), '0.00');
  await page.screenshot({ path: `${artifacts}/navigation-arrival.png`, fullPage: true });
  // Stop latch survives held drive keys and ordinary Auto commands.
  await page.keyboard.down('w');
  await page.waitForTimeout(300);
  await page.keyboard.press('e');
  await page.waitForFunction(() => document.querySelector('#mode').textContent === 'ESTOP');
  await page.keyboard.press('g');
  await page.waitForTimeout(250);
  assert.equal(await page.locator('#mode').innerText(), 'ESTOP');
  await page.keyboard.press('r');
  await page.waitForFunction(() => document.querySelector('#mode').textContent === 'PAUSED');
  await page.waitForTimeout(200);
  assert.equal(await page.locator('#mode').innerText(), 'PAUSED');
  await page.keyboard.up('w');
  // Import, model/planner changes, and JSON validation use the same canonical scene.
  await page
    .locator('#import-file')
    .setInputFiles({
      name: 'case.json',
      mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(saved)),
    });
  await page.waitForFunction(
    () => document.querySelector('#message').textContent === '场景已导入并重置。',
  );
  await page.locator('#model').selectOption('ackermann');
  await page.locator('#planner').selectOption('voronoi');
  await page.locator('#body-length').fill('1');
  await page.locator('#body-length').press('Tab');
  await page.locator('#apply').click();
  await applied();
  let car = await exportScene();
  assert.equal(car.config.dwa.prediction_time, 2.4);
  assert.equal(car.config.voronoi.time_budget, 2);
  await page.locator('[data-action="auto"]').click();
  await page.waitForFunction(
    () => document.querySelector('#diagnostics').textContent.startsWith('arrived'),
    {},
    { timeout: 25000 },
  );
  await page.screenshot({ path: `${artifacts}/ackermann-arrival.png`, fullPage: true });
  // Mobile layout, touch editing, and virtual stick via actual browser pointer events.
  const context = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
    deviceScaleFactor: 2,
  });
  const mobile = await context.newPage();
  mobile.on('pageerror', (e) => errors.push(e.message));
  await mobile.goto(base);
  await mobile.waitForFunction(
    () => document.querySelector('#connection').textContent === '已连接',
  );
  assert.ok(await mobile.evaluate(() => document.documentElement.scrollWidth <= 390));
  await mobile.locator('#new-scene').tap();
  await mobile.locator('#apply').tap();
  await mobile.waitForFunction(
    () => document.querySelector('#edit-status').textContent === '已应用',
  );
  await mobile.locator('#joystick').scrollIntoViewIfNeeded();
  const stick = await mobile.locator('#joystick').boundingBox();
  const touch = await context.newCDPSession(mobile);
  await touch.send('Input.dispatchTouchEvent', {
    type: 'touchStart',
    touchPoints: [{ x: stick.x + stick.width / 2, y: stick.y + 18 }],
  });
  await mobile.waitForFunction(() => document.querySelector('#mode').textContent === 'MANUAL');
  await mobile.waitForTimeout(900);
  assert.ok(Number(await mobile.locator('#speed').innerText()) > 0.05);
  await touch.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  await mobile.screenshot({ path: `${artifacts}/mobile-touch.png`, fullPage: true });
  await context.close();
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({
      passed: true,
      checks: [
        'drawing',
        'start-goal-placement',
        'undo-redo',
        'case-save-load-export-import',
        'two-model-navigation',
        'estop-latch',
        'mobile-layout',
        'touch-stick',
      ],
      artifacts,
    }),
  );
} finally {
  await browser.close();
}
