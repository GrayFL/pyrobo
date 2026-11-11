// Real A/D input and actual canvas vertices: verify the drive axle is the pivot.
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const base = process.env.WEB_SIM_URL || 'http://127.0.0.1:8000';
const artifacts = process.env.WEB_SIM_ARTIFACTS || '/tmp/pyrobo-rotation-center';
await mkdir(artifacts, {recursive: true});
const browser = await chromium.launch({headless: true,
  executablePath: process.env.BROWSER_EXECUTABLE || undefined, args: ['--no-sandbox']});
const errors = [], results = [];
const page = await browser.newPage({viewport: {width: 1512, height: 1100}});
page.on('pageerror', error => errors.push(error.message));
try {
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#connection').textContent === '已连接');
  await page.locator('#new-scene').click();
  await page.evaluate(async () => {
    const {Editor} = await import('/static/editor.js');
    const drawRobot = Editor.prototype.robot;
    window.rotationFrames = [];
    Editor.prototype.robot = function(pose, ghost) {
      let corners = null;
      const drawPath = this.path;
      this.path = function(points, color, width, fill, close) {
        if (!ghost && !corners && fill && close) corners = points.map(p => [...p]);
        return drawPath.call(this, points, color, width, fill, close);
      };
      try { drawRobot.call(this, pose, ghost); } finally { this.path = drawPath; }
      if (!ghost) window.rotationFrames.push({pose: [...pose], corners,
        speed: this.state?.speed, omega: this.state?.omega, mode: this.state?.mode,
        footprint: {...this.scene.config.footprint}});
    };
  });
  for (const name of ['dwa', 'towing_dwa', 'towing_teb']) {
    await page.locator('#local-optimizer').selectOption(name);
    for (const [id, value] of [['body-length', '.79'], ['body-width', '.54'], ['body-offset', '-.295']]) {
      await page.locator(`#${id}`).fill(value);
      await page.locator(`#${id}`).press('Tab');
    }
    await page.locator('#apply').click();
    await page.waitForFunction(() => document.querySelector('#edit-status').textContent === '已应用');
    await page.locator('[data-action="manual"]').click();
    await page.evaluate(() => {window.rotationFrames = [];});
    await page.keyboard.down('a');
    await page.waitForFunction(() => window.rotationFrames.some(f => f.pose[2] > 1.4));
    await page.screenshot({path: `${artifacts}/${name}-left.png`, fullPage: true});
    await page.keyboard.up('a');
    await page.waitForFunction(() => document.querySelector('#turn').textContent.endsWith('/ 0.00'));
    const heading = await page.evaluate(() => window.rotationFrames.at(-1).pose[2]);
    await page.keyboard.down('d');
    await page.waitForFunction(angle => window.rotationFrames.at(-1).pose[2] < angle-1.4, heading);
    await page.keyboard.up('d');
    await page.locator('[data-action="paused"]').click();
    await page.waitForFunction(() => document.querySelector('#mode').textContent === 'PAUSED' &&
      document.querySelector('#turn').textContent.endsWith('/ 0.00'));
    const frames = await page.evaluate(() => window.rotationFrames.filter(f => Math.abs(f.omega) > .1));
    assert.ok(frames.length >= 15);
    assert.ok(frames.some(f => f.omega > .1) && frames.some(f => f.omega < -.1));
    let maxAxleDisplacement = 0, maxRadiusError = 0;
    for (const f of frames) {
      const center = f.corners.reduce((a, p) => [a[0]+p[0]/4, a[1]+p[1]/4], [0, 0]);
      const pivotShift = Math.hypot(f.pose[0]+3, f.pose[1]);
      const radiusError = Math.abs(Math.hypot(center[0]+3, center[1])-.295);
      assert.ok(pivotShift < 1e-10, 'drive axle moved during a pure spin');
      assert.ok(radiusError < 1e-10, 'drawn body center did not orbit the drive axle');
      assert.ok(Math.abs(f.speed) < 1e-10);
      maxAxleDisplacement = Math.max(maxAxleDisplacement, pivotShift);
      maxRadiusError = Math.max(maxRadiusError, radiusError);
    }
    // The geometric center must actually move, rather than just stay offset.
    const centers = frames.map(f => f.corners.reduce((a, p) => [a[0]+p[0]/4, a[1]+p[1]/4], [0, 0]));
    assert.ok(centers.some(c => Math.hypot(c[0]-centers[0][0], c[1]-centers[0][1]) > .25));
    results.push({name, frames: frames.length, maxAxleDisplacement, maxRadiusError});
    await writeFile(`${artifacts}/${name}-frames.json`, JSON.stringify(frames, null, 2));
  }
  assert.ok((await page.locator('.legend').innerText()).includes('驱动轴心'));
  await page.setViewportSize({width: 1000, height: 800});
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= 1000));
  await page.setViewportSize({width: 390, height: 844});
  await page.locator('#fit').click();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= 390));
  await page.screenshot({path: `${artifacts}/mobile.png`, fullPage: true});
  assert.deepEqual(errors, []);
  const summary = {passed: true, base, results};
  await writeFile(`${artifacts}/result.json`, JSON.stringify(summary, null, 2));
  console.log(JSON.stringify(summary));
} finally {
  await browser.close();
}
