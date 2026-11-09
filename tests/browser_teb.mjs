// Opt-in real-browser TEB configuration, persistence, corner navigation and mobile layout.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const base = process.env.WEB_SIM_URL || 'http://127.0.0.1:8090';
const artifacts = process.env.WEB_SIM_ARTIFACTS || '/tmp/pyrobo-teb-browser';
await mkdir(artifacts, { recursive: true });
const browser = await chromium.launch({headless: true,
  executablePath: process.env.BROWSER_EXECUTABLE || undefined, args: ['--no-sandbox']});
const errors = [], states = [], corners = [];
let page;
try {
  page = await browser.newPage({viewport: {width: 1512, height: 1100}});
  page.on('pageerror', error => errors.push(error.message));
  page.on('websocket', ws => ws.on('framereceived', frame => {
    const data = JSON.parse(frame.payload.toString());
    if (data.type === 'state') states.push(data);
  }));
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#connection').textContent === '已连接');
  await page.locator('#new-scene').click();
  assert.equal(await page.locator('#local-optimizer').inputValue(), 'dwa');
  await page.locator('#local-optimizer').selectOption('teb');
  assert.equal(await page.locator('#teb-settings').isVisible(), true);
  await page.locator('#teb-nodes').fill('12');
  await page.locator('#teb-nodes').press('Tab');
  await page.locator('#teb-iterations').fill('10');
  await page.locator('#teb-iterations').press('Tab');
  await page.locator('#teb-budget').fill('');
  await page.locator('#teb-budget').press('Tab');
  await page.locator('#horizon').evaluate(el => {el.closest('details').open = true;});
  await page.locator('#horizon').fill('2.2');
  await page.locator('#horizon').press('Tab');
  await page.locator('#apply').click();
  await page.waitForFunction(() => document.querySelector('#edit-status').textContent === '已应用');
  const download = page.waitForEvent('download');
  await page.locator('#export').click();
  const exported = JSON.parse(await readFile(await (await download).path(), 'utf8'));
  assert.equal(exported.config.backend.local_optimizer, 'teb');
  assert.equal(exported.config.teb.max_samples, 12);
  assert.equal(exported.config.teb.max_iterations, 10);
  assert.equal(exported.config.teb.time_budget, null);
  assert.equal(exported.config.teb.prediction_time, 2.2);
  assert.equal(exported.config.dwa.prediction_time, 1.8);
  await writeFile(`${artifacts}/teb-scene.json`, JSON.stringify(exported, null, 2));
  await page.locator('#save').click();
  await page.waitForFunction(() => document.querySelector('#message').textContent.startsWith('已保存'));
  for (const model of ['differential', 'ackermann']) {
    await page.locator('#model').selectOption(model);
    await page.locator('#apply').click();
    await page.waitForFunction(() => document.querySelector('#edit-status').textContent === '已应用');
    await page.locator('[data-action="auto"]').click();
    await page.waitForFunction(() => document.querySelector('#diagnostics').textContent.includes('TEB converged') ||
      document.querySelector('#diagnostics').textContent.includes('TEB iteration_limit'));
    await page.screenshot({path: `${artifacts}/teb-${model}.png`, fullPage: true});
    await page.waitForFunction(() => document.querySelector('#diagnostics').textContent.startsWith('arrived'),
      {}, {timeout: 30000});
    assert.equal(await page.locator('#speed').innerText(), '0.00');
  }
  const narrow = await readFile(new URL('../pyrobo/apps/scenes/teb_narrow_road.json', import.meta.url));
  await page.locator('#import-file').setInputFiles({name: 'teb-narrow-road.json', mimeType: 'application/json',
    buffer: narrow});
  await page.waitForFunction(() => document.querySelector('#message').textContent === '场景已导入并重置。');
  for (const planner of ['rrt_connect', 'voronoi']) {
    await page.locator('#planner').selectOption(planner);
    await page.locator('#apply').click();
    await page.waitForFunction(() => document.querySelector('#edit-status').textContent === '已应用');
    await page.locator('[data-action="auto"]').click();
    await page.waitForFunction(() => document.querySelector('#diagnostics').textContent.startsWith('navigating'),
      {}, {timeout: 30000});
    await page.waitForFunction(() => document.querySelector('#diagnostics').textContent.startsWith('arrived'),
      {}, {timeout: 180000});
    const final = states.at(-1);
    assert.equal(final.optimizer, 'teb');
    assert.equal(final.status, 'arrived');
    assert.ok(final.safe && !final.collided && !final.fault);
    assert.ok(Math.max(...final.trail.map(p => p[1])) > 6);
    assert.ok(Math.abs(final.speed) < 1e-3 && Math.abs(final.omega) < 1e-3);
    corners.push({planner, ticks: final.ticks, simulation_seconds: final.time, pose: final.pose});
    await page.screenshot({path: `${artifacts}/teb-narrow-${planner}.png`, fullPage: true});
  }
  await page.locator('#import-file').setInputFiles({name: 'teb.json', mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(exported))});
  await page.waitForFunction(() => document.querySelector('#message').textContent === '场景已导入并重置。');
  assert.equal(await page.locator('#local-optimizer').inputValue(), 'teb');
  assert.equal(await page.locator('#teb-nodes').inputValue(), '12');
  await page.locator('#local-optimizer').selectOption('dwa');
  assert.equal(await page.locator('#horizon').inputValue(), '1.8');
  assert.equal(await page.locator('#teb-settings').isVisible(), false);
  await page.locator('#local-optimizer').selectOption('teb');
  assert.equal(await page.locator('#horizon').inputValue(), '2.2');
  const mobile = await browser.newPage({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true});
  mobile.on('pageerror', error => errors.push(error.message));
  await mobile.goto(base);
  await mobile.waitForFunction(() => document.querySelector('#connection').textContent === '已连接');
  await mobile.locator('#local-optimizer').selectOption('teb');
  assert.ok(await mobile.evaluate(() => document.documentElement.scrollWidth <= 390));
  await mobile.screenshot({path: `${artifacts}/teb-mobile.png`, fullPage: true});
  assert.deepEqual(errors, []);
  const automatic = states.filter(s => s.mode === 'auto' && s.candidates > 0);
  assert.ok(automatic.length > 10);
  assert.ok(automatic.every(s => s.safe && !s.collided && !s.fault));
  assert.ok(automatic.some(s => s.optimizer_diagnostics.teb_warm_started));
  console.log(JSON.stringify({passed: true, automatic_samples: automatic.length, corners, artifacts}));
} catch (error) {
  await writeFile(`${artifacts}/failed-states.json`, JSON.stringify(states, null, 2));
  if (page) await page.screenshot({path: `${artifacts}/failure.png`, fullPage: true});
  throw error;
} finally {
  await browser.close();
}
