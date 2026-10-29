// Real-browser regression for both global backends using the saved user geometry.
import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const base = process.env.WEB_SIM_URL || 'http://127.0.0.1:8091';
const out = process.env.WEB_SIM_ARTIFACTS || '/tmp/pyrobo-narrow-browser';
const scene = JSON.parse(await readFile(new URL('../examples/scenes/narrow_ackermann_adaptive.json', import.meta.url), 'utf8'));
await mkdir(out, { recursive: true });
const browser = await chromium.launch({ headless: true,
  executablePath: process.env.BROWSER_EXECUTABLE || undefined, args: ['--no-sandbox'] });
try {
  for (const planner of ['rrt_connect', 'voronoi']) {
    const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.waitForFunction(() => document.querySelector('#connection').textContent === '已连接');
    scene.config.backend.planner = planner;
    await page.locator('#import-file').setInputFiles({ name: 'narrow.json', mimeType: 'application/json',
      buffer: Buffer.from(JSON.stringify(scene)) });
    await page.waitForFunction(() => document.querySelector('#message').textContent === '场景已导入并重置。');
    assert.equal(await page.locator('#adaptive-clearance').inputValue(), 'adaptive');
    assert.equal(await page.locator('#goal-approach').inputValue(), '');
    await page.locator('[data-action="auto"]').click();
    await page.waitForFunction(() => document.querySelector('#diagnostics').textContent.startsWith('arrived'),
      {}, { timeout: 65000 });
    assert.equal(await page.locator('#speed').innerText(), '0.00');
    assert.deepEqual(errors, []);
    await page.screenshot({ path: `${out}/${planner}-arrival.png`, fullPage: true });
    console.log(JSON.stringify({ planner, status: await page.locator('#diagnostics').innerText(),
      body: scene.config.footprint, clearance: scene.config[planner].clearance }));
    await page.close();
  }
} finally { await browser.close(); }
