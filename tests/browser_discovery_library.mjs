// Opt-in browser test. Run against an isolated --scenes-dir; it creates/deletes named resources.
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
const {chromium} = await import(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const base = process.env.WEB_SIM_URL || 'http://127.0.0.1:8092';
const artifacts = process.env.WEB_SIM_ARTIFACTS || '/tmp/pyrobo-discovery-browser';
await mkdir(artifacts,{recursive:true});
const browser = await chromium.launch({headless:true,executablePath:process.env.BROWSER_EXECUTABLE,args:['--no-sandbox']});
const errors=[], states=[];
const page = await browser.newPage({viewport:{width:1512,height:1100}});
page.on('pageerror',e=>errors.push(e.message));
page.on('dialog',d=>d.accept());
page.on('websocket', ws=>ws.on('framereceived', f=>{const d=JSON.parse(f.payload.toString());if(d.type==='state') states.push(d);}));
const fill = async (id,value) => {await page.locator('#'+id).fill(String(value));await page.locator('#'+id).press('Tab');};
const modal = async () => {await page.locator('#library-open').click();await page.locator('#library').waitFor({state:'visible'});await page.waitForFunction(()=>!document.querySelector('#resource-save').disabled);};
const response = (method,path,click) => Promise.all([page.waitForResponse(r=>r.request().method()===method && new URL(r.url()).pathname===path),click()]).then(([r])=>r);
try {
  // Only remove resources authored by this opt-in test in its dedicated category.
  const registered = await (await page.request.get(base+'/api/library')).json();
  if (registered.some(f=>f.id==='浏览器分类')) {
    for (const name of ['浏览器发现测试','浏览器重命名']) await page.request.delete(base+'/api/library/浏览器分类/scenes/'+name);
    await page.request.delete(base+'/api/library/浏览器分类/maps/共用空场地');
    await page.request.delete(base+'/api/library/浏览器分类');
  }
  await page.goto(base);
  await page.waitForFunction(()=>document.querySelector('#connection').textContent==='已连接');
  const catalog = await (await page.request.get(base+'/api/catalog')).json();
  for (const [id,key] of [['planner','planners'],['model','models'],['local-optimizer','optimizers']]) {
    assert.deepEqual(await page.locator(`#${id} option`).evaluateAll(xs=>xs.map(x=>x.value)),catalog[key].map(x=>x.id));
  }
  await page.locator('#new-scene').click();
  await page.locator('#vehicle-template').selectOption('towing');
  assert.equal(await page.locator('#body-length').inputValue(),'0.79');
  assert.equal(await page.locator('#body-width').inputValue(),'0.54');
  assert.equal(await page.locator('#body-offset').inputValue(),'-0.295');
  for (const opt of catalog.optimizers) {
    await page.locator('#local-optimizer').selectOption(opt.id);
    assert.equal(await page.locator('#body-offset').inputValue(),'-0.295');
  }
  await fill('body-width',.57);
  assert.equal(await page.locator('#vehicle-template').inputValue(),'custom');
  await page.locator('#vehicle-template').selectOption('compact');
  await page.locator('#local-optimizer').selectOption(catalog.optimizers.some(x=>x.id==='browser_local')?'browser_local':'dwa');
  await page.locator('#model').selectOption(catalog.models.some(x=>x.id==='browser_model')?'browser_model':'differential');
  await page.locator('#planner').selectOption(catalog.planners.some(x=>x.id==='browser_global')?'browser_global':'rrt_connect');
  // Draw an unknown rectangle, convert it to known terrain, then back again.
  await page.locator('#obstacle-layer').selectOption('unknown');
  await page.locator('[data-tool="rect"]').click();
  await page.locator('#fit').click();
  await page.evaluate(()=>scrollTo(0,0));
  const box=await page.locator('#map').boundingBox();
  const scale=Math.min((box.width-70)/10,(box.height-80)/8);
  const at=(x,y)=>[box.x+box.width/2+x*scale,box.y+box.height/2-y*scale];
  await page.mouse.move(...at(-.1,-.5)); await page.mouse.down();await page.mouse.move(...at(.2,.5),{steps:8});await page.mouse.up();
  await page.locator('[data-tool="classify"]').click();await page.mouse.click(...at(.05,0));await page.mouse.click(...at(.05,0));
  await fill('lidar-range',2.5);
  await fill('scene-name','浏览器发现测试');
  await page.locator('#apply').click();
  await page.waitForFunction(()=>document.querySelector('#edit-status').textContent==='已应用');
  await modal();
  await page.getByText('管理分类目录',{exact:true}).click();
  await fill('folder-name','浏览器分类');
  const add=await response('POST','/api/library',()=>page.locator('#folder-add').click());assert.equal(add.status(),201);
  await page.waitForFunction(()=>document.querySelector('#library-folder').value==='浏览器分类');
  await fill('resource-name','浏览器发现测试');await fill('terrain-name','共用空场地');
  const path='/api/library/浏览器分类/scenes';
  const encoded='/api/library/'+encodeURIComponent('浏览器分类')+'/scenes';
  const saved=await response('POST',encoded,()=>page.locator('#resource-save').click());assert.equal(saved.status(),201);
  await page.waitForFunction(()=>document.querySelector('#library-message').textContent.includes('已保存'));
  const duplicate=await response('POST',encoded,()=>page.locator('#resource-save').click());assert.equal(duplicate.status(),409);
  await page.waitForFunction(()=>document.querySelector('#library-message').textContent.includes('已存在'));
  await page.screenshot({path:artifacts+'/duplicate-desktop.png'});
  const persisted=await (await page.request.get(base+path+'/浏览器发现测试')).json();
  assert.equal(persisted.unknown_obstacles.boxes.length,1);
  assert.equal(persisted.config.map.boxes.length,0);
  // Rename/overwrite the loaded resource and export a self-contained v2 scene.
  await fill('resource-name','浏览器重命名');
  const overwritten=await response('PUT',encoded+'/'+encodeURIComponent('浏览器发现测试'),()=>page.locator('#resource-overwrite').click());assert.equal(overwritten.status(),200);
  await page.waitForFunction(()=>document.querySelector('#library-message').textContent.includes('已保存「浏览器重命名」'));
  const downloadPromise=page.waitForEvent('download');await page.locator('#resource-export').click();
  const download=await downloadPromise;await download.saveAs(artifacts+'/export.json');
  const exported=JSON.parse(await readFile(artifacts+'/export.json','utf8'));
  assert.equal(exported.schema_version,2);assert.equal(exported.terrain.kind,'terrain');
  assert.deepEqual(Object.keys(exported.config.map).sort(),['goal','start']);
  await page.locator('#library-close').click();
  await page.locator('#import-file').setInputFiles(artifacts+'/export.json');
  await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('场景已导入'));
  await page.locator('[data-action="auto"]').click();
  await page.waitForFunction(()=>document.querySelector('#diagnostics').textContent.startsWith('arrived'),null,{timeout:90000});
  const driven=states.filter(s=>s.mode==='auto'&&s.ticks>0);
  assert.ok(driven.length>20);assert.ok(driven.every(s=>s.safe===true&&!s.collided));
  assert.ok(driven.some(s=>(s.experiment.scan_points||[]).some(p=>Math.abs(p[0]+.1)<.03&&Math.abs(p[1])<.5)));
  await page.screenshot({path:artifacts+'/lidar-arrived.png'});
  // Terrain import preserves vehicle and task settings; it does not inject an authored pose.
  await writeFile(artifacts+'/terrain.json',JSON.stringify(exported.terrain));
  await page.locator('#import-file').setInputFiles(artifacts+'/terrain.json');
  await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('地形已载入'));
  assert.equal(await page.locator('#body-length').inputValue(),'.5'.replace(/^\./,'0.'));
  await modal();
  await page.locator('#resource-kind').selectOption('maps');
  await page.waitForFunction(()=>document.querySelector('#scene-list').textContent.includes('共用空场地'));
  const mapRow=page.locator('.scene-item').filter({hasText:'共用空场地'});
  await mapRow.getByRole('button',{name:'加载 / 编辑'}).click();await page.locator('#library').waitFor({state:'hidden'});
  await modal();
  await fill('resource-name','共用空场地');
  const mapUpdate=await response('PUT','/api/library/'+encodeURIComponent('浏览器分类')+'/maps/'+encodeURIComponent('共用空场地'),()=>page.locator('#resource-overwrite').click());
  assert.equal(mapUpdate.status(),200);
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:artifacts+'/manager-mobile.png'});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await page.locator('#resource-kind').selectOption('scenes');
  await page.waitForFunction(()=>document.querySelector('#scene-list').textContent.includes('浏览器重命名'));
  const deleted=await response('DELETE','/api/library/'+encodeURIComponent('浏览器分类')+'/scenes/'+encodeURIComponent('浏览器重命名'),()=>page.locator('.scene-item').filter({hasText:'浏览器重命名'}).getByRole('button',{name:'删除',exact:true}).click());
  assert.equal(deleted.status(),200);
  await page.waitForFunction(()=>!document.querySelector('#scene-list').textContent.includes('浏览器重命名'));
  await page.locator('#resource-kind').selectOption('maps');
  await page.waitForFunction(()=>document.querySelector('#scene-list').textContent.includes('共用空场地'));
  const removed=await response('DELETE','/api/library/'+encodeURIComponent('浏览器分类')+'/maps/'+encodeURIComponent('共用空场地'),()=>page.locator('.scene-item').filter({hasText:'共用空场地'}).getByRole('button',{name:'删除',exact:true}).click());
  assert.equal(removed.status(),200);
  assert.deepEqual(errors,[]);
  await writeFile(artifacts+'/result.json',JSON.stringify({passed:true,catalog,automaticSnapshots:driven.length,errors},null,2));
  console.log('Browser passed:',driven.length,'automatic snapshots');
} catch(e) {
  await page.screenshot({path:artifacts+'/failure.png'});console.error(await page.locator('#library-message').textContent());throw e;
} finally {await browser.close();}
