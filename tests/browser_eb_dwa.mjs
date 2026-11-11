import { saveThroughLibrary } from './browser_library_helpers.mjs';
// Opt-in real Edge/Chromium test of the same portable experiments as CLI replay.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const base = process.env.WEB_SIM_URL || 'http://127.0.0.1:8090';
const artifacts = process.env.WEB_SIM_ARTIFACTS || '/tmp/pyrobo-eb-browser';
await mkdir(artifacts, { recursive: true });
const browser = await chromium.launch({ headless: true,
  executablePath: process.env.BROWSER_EXECUTABLE || undefined, args: ['--no-sandbox'] });
const errors = [], states = [], runs = [];
let page;
try {
  page = await browser.newPage({ viewport: { width: 1512, height: 1100 } });
  page.on('pageerror', error => errors.push(error.message));
  page.on('websocket', ws => ws.on('framereceived', frame => {
    const data = JSON.parse(frame.payload.toString());
    if (data.type === 'state') states.push(data);
  }));
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#connection').textContent === '已连接');
  await page.locator('#library-open').click();
  await page.locator('#library-folder').selectOption('builtin');
  const row = page.locator('.scene-item').filter({hasText: 'EB-DWA · 长直道与直角'});
  await row.waitFor();
  assert.equal(await row.count(), 1);
  await row.getByRole('button', {name: '加载'}).click();
  await page.waitForFunction(() => document.querySelector('#local-optimizer').value === 'eb_dwa');
  assert.ok(await page.locator('#eb-settings').isVisible());
  assert.ok((await page.locator('#experiment').inputValue()).includes('reference_path'));
  assert.ok(await page.locator('#eb-speed-profile').isChecked());
  await page.locator('#eb-speed-profile').uncheck();
  await page.locator('#eb-speed-profile').check();
  await page.locator('#eb-lateral-fraction').fill('0.3');
  await page.locator('#eb-lateral-fraction').press('Tab');
  await page.locator('#eb-clearance').fill('0.18');
  await page.locator('#eb-clearance').press('Tab');
  await page.locator('#eb-budget').fill('');
  await page.locator('#eb-budget').press('Tab');
  await page.locator('#apply').click();
  await page.waitForFunction(() => document.querySelector('#edit-status').textContent === '已应用');
  const download = page.waitForEvent('download');
  await page.locator('#export').click();
  const exported = JSON.parse(await readFile(await (await download).path(), 'utf8'));
  assert.equal(exported.config.eb_dwa.preferred_clearance, .18);
  assert.equal(exported.config.eb_dwa.time_budget_s, null);
  assert.equal(exported.config.eb_dwa.speed_profile, true);
  assert.equal(exported.config.eb_dwa.speed_lateral_fraction, .3);
  assert.deepEqual(exported.experiment.reference_path, [[0, 0], [5, 0], [5, 5]]);
  await saveThroughLibrary(page);
  const cases = process.env.EB_BROWSER_EXTENDED
    ? ['corner', 'ackermann_reverse', 'ackermann_corner', 'narrow_road:rrt_connect', 'narrow_road:voronoi']
    : ['corner', 'ackermann_reverse'];
  for (const name of cases) {
    if (name !== 'corner') {
      const generation = states.at(-1).generation;
      const [scene, planner] = name.split(':');
      const data = JSON.parse(await readFile(new URL(`../pyrobo/apps/scenes/eb_dwa_${scene}.json`, import.meta.url)));
      data.config.eb_dwa = {time_budget_s: null};
      if (planner) data.config.backend.planner = planner;
      await page.locator('#import-file').setInputFiles({name: `${name}.json`, mimeType: 'application/json',
        buffer: Buffer.from(JSON.stringify(data))});
      await page.waitForFunction(() => document.querySelector('#message').textContent === '场景已导入并重置。');
      // Repeated imports share the same message text. Wait for the backend's
      // new episode before sending AUTO, otherwise the reset can erase it.
      const deadline = Date.now()+10000;
      while (states.at(-1)?.generation === generation && Date.now() < deadline)
        await page.waitForTimeout(50);
      assert.ok(states.at(-1).generation > generation);
    }
    await page.locator('[data-action="auto"]').click();
    await page.waitForFunction(() => document.querySelector('#diagnostics').textContent.includes('EB '));
    await page.screenshot({path: `${artifacts}/${name}-running.png`, fullPage: true});
    await page.waitForFunction(() => document.querySelector('#diagnostics').textContent.startsWith('arrived'),
      {}, {timeout: 240000});
    const final = states.at(-1);
    assert.equal(final.optimizer, 'eb_dwa');
    assert.ok(final.safe && !final.collided && !final.fault);
    assert.ok(Math.abs(final.speed) < 1e-3);
    runs.push({name, ticks: final.ticks, simulation_seconds: final.time, pose: final.pose});
    await page.screenshot({path: `${artifacts}/${name}-arrived.png`, fullPage: true});
  }
  await page.locator('#local-optimizer').selectOption('dwa');
  assert.equal(await page.locator('#eb-settings').isVisible(), false);
  await page.locator('#local-optimizer').selectOption('eb_dwa');
  assert.equal(await page.locator('#eb-settings').isVisible(), true);
  const mobile = await browser.newPage({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true});
  mobile.on('pageerror', error => errors.push(error.message));
  await mobile.goto(base);
  await mobile.waitForFunction(() => document.querySelector('#connection').textContent === '已连接');
  await mobile.locator('#local-optimizer').selectOption('eb_dwa');
  assert.equal(await mobile.locator('#eb-budget').inputValue(), '');
  assert.ok(await mobile.evaluate(() => document.documentElement.scrollWidth <= 390));
  await mobile.screenshot({path: `${artifacts}/mobile.png`, fullPage: true});
  const moving = states.filter(s => s.mode === 'auto' && s.candidates > 0);
  assert.ok(moving.length > 20 && moving.every(s => s.safe && !s.collided && !s.fault));
  assert.ok(moving.some(s => s.reference?.xy.length > 2 && s.optimizer_diagnostics.eb_cache_hit));
  assert.deepEqual(errors, []);
  await writeFile(`${artifacts}/states.json`, JSON.stringify(states));
  console.log(JSON.stringify({passed: true, runs, samples: moving.length, artifacts}));
} catch (error) {
  await writeFile(`${artifacts}/failed-states.json`, JSON.stringify(states));
  if (page) await page.screenshot({path: `${artifacts}/failure.png`, fullPage: true});
  throw error;
} finally {
  await browser.close();
}
