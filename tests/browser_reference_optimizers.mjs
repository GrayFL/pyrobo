import { saveThroughLibrary } from './browser_library_helpers.mjs';
// Opt-in real browser verification of the editable classic planner variants.
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const base = process.env.WEB_SIM_URL || 'http://127.0.0.1:8090';
const artifacts = process.env.WEB_SIM_ARTIFACTS || '/tmp/pyrobo-reference-browser';
await mkdir(artifacts, {recursive: true});
const browser = await chromium.launch({headless: true,
  executablePath: process.env.BROWSER_EXECUTABLE || undefined, args: ['--no-sandbox']});
const errors = [], states = [], results = [];
let page;
try {
  page = await browser.newPage({viewport: {width: 1512, height: 1100}});
  page.on('pageerror', e => errors.push(e.message));
  page.on('websocket', ws => ws.on('framereceived', frame => {
    const data = JSON.parse(frame.payload.toString());
    if (data.type === 'state') states.push(data);
  }));
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#connection').textContent === '已连接');
  await page.locator('#new-scene').click();
  // Vehicle configuration belongs to the scene, independently of the optimizer.
  for (const [id, value] of [['body-length', '0.79'], ['body-width', '0.54'], ['body-offset', '-0.295']]) {
    await page.locator(`#${id}`).fill(value);
    await page.locator(`#${id}`).press('Tab');
  }
  await page.locator('#model').selectOption('ackermann');
  for (const name of ['ref_dwa', 'ref_teb', 'towing_dwa', 'towing_teb']) {
    await page.locator('#local-optimizer').selectOption(name);
    assert.equal(await page.locator('#model').inputValue(), 'ackermann');
    assert.equal(await page.locator('#model').isDisabled(), false);
    assert.equal(await page.locator('#body-offset').inputValue(), '-0.295');
  }
  for (const name of ['ref_dwa', 'ref_teb', 'towing_dwa', 'towing_teb']) {
    await page.locator('#local-optimizer').selectOption(name);
    const towing = name.startsWith('towing'), teb = name.endsWith('teb');
    assert.equal(await page.locator('#body-length').inputValue(), '0.79');
    assert.equal(await page.locator('#body-width').inputValue(), '0.54');
    assert.equal(await page.locator('#body-offset').inputValue(), '-0.295');
    assert.equal(await page.locator('#model').isDisabled(), false);
    // Change model while a towing optimizer is selected, then run the actual car.
    const model = towing ? 'ackermann' : 'differential';
    await page.locator('#model').selectOption(model);
    assert.equal(await page.locator('#teb-settings').isVisible(), teb);
    if (teb) {
      assert.equal(await page.locator('#teb-budget').inputValue(), '');
      await page.locator('#teb-nodes').fill(name === 'ref_teb' ? '12' : '14');
      await page.locator('#teb-nodes').press('Tab');
    }
    await page.locator('#apply').click();
    await page.waitForFunction(() => document.querySelector('#edit-status').textContent === '已应用');
    const pending = page.waitForEvent('download');
    await page.locator('#export').click();
    const scene = JSON.parse(await readFile(await (await pending).path(), 'utf8'));
    assert.equal(scene.config.backend.local_optimizer, name);
    assert.equal(scene.config.backend.model, model);
    assert.deepEqual(scene.config.footprint, {length: .79, width: .54, offset_x: -.295, margin: .08});
    if (teb) {
      assert.equal(scene.config[name].max_samples, name === 'ref_teb' ? 12 : 14);
      assert.equal(scene.config.teb.max_samples, 16);
      assert.equal(scene.config[name].time_budget, null);
    }
    await writeFile(`${artifacts}/${name}.json`, JSON.stringify(scene, null, 2));
    await page.locator('[data-action="auto"]').click();
    await page.waitForFunction(() => document.querySelector('#diagnostics').textContent.startsWith('navigating'));
    await page.screenshot({path: `${artifacts}/${name}.png`, fullPage: true});
    await page.waitForFunction(() => document.querySelector('#diagnostics').textContent.startsWith('arrived'),
      {}, {timeout: 60000});
    const final = states.at(-1);
    assert.equal(final.optimizer, name);
    assert.ok(final.safe && !final.collided && !final.fault);
    assert.ok(Math.abs(final.speed) < 1e-3 && Math.abs(final.omega) < 1e-3);
    results.push({name, model, ticks: final.ticks, pose: final.pose});
  }
  await page.locator('#local-optimizer').selectOption('ref_teb');
  assert.equal(await page.locator('#teb-nodes').inputValue(), '12');
  assert.equal(await page.locator('#body-offset').inputValue(), '-0.295');
  assert.equal(await page.locator('#model').inputValue(), 'ackermann');
  await page.locator('#local-optimizer').selectOption('towing_teb');
  assert.equal(await page.locator('#teb-nodes').inputValue(), '14');
  await saveThroughLibrary(page);
  const exported = await readFile(`${artifacts}/towing_teb.json`);
  await page.locator('#import-file').setInputFiles({name: 'towing.json', mimeType: 'application/json', buffer: exported});
  await page.waitForFunction(() => document.querySelector('#message').textContent === '场景已导入并重置。');
  assert.equal(await page.locator('#local-optimizer').inputValue(), 'towing_teb');
  assert.equal(await page.locator('#body-offset').inputValue(), '-0.295');
  assert.equal(await page.locator('#model').inputValue(), 'ackermann');
  assert.equal(await page.locator('#model').isDisabled(), false);
  const mobile = await browser.newPage({viewport: {width: 390, height: 844}, isMobile: true, hasTouch: true});
  mobile.on('pageerror', e => errors.push(e.message));
  await mobile.goto(base);
  await mobile.waitForFunction(() => document.querySelector('#connection').textContent === '已连接');
  await mobile.locator('#local-optimizer').selectOption('towing_dwa');
  assert.ok(await mobile.evaluate(() => document.documentElement.scrollWidth <= 390));
  await mobile.screenshot({path: `${artifacts}/mobile.png`, fullPage: true});
  assert.deepEqual(errors, []);
  const automatic = states.filter(s => s.mode === 'auto' && s.candidates > 0);
  assert.ok(automatic.length > 20);
  assert.ok(automatic.every(s => s.safe && !s.collided && !s.fault));
  const summary = {passed: true, results, automatic_samples: automatic.length, artifacts};
  await writeFile(`${artifacts}/result.json`, JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary));
} catch (error) {
  await writeFile(`${artifacts}/failed-states.json`, JSON.stringify(states, null, 2));
  if (page) await page.screenshot({path: `${artifacts}/failure.png`, fullPage: true});
  throw error;
} finally {
  await browser.close();
}
