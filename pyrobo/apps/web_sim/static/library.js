// One manager for named scene documents and reusable terrain resources.
const $ = id => document.getElementById(id);
const enc = encodeURIComponent;
async function api(url, method = 'GET', body) {
  const response = await fetch(url, {method, headers: {'Content-Type': 'application/json'},
    ...(body === undefined ? {} : {body: JSON.stringify(body)})});
  const data = await response.json();
  if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
function download(data, name) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
  const a = document.createElement('a'); a.href = url;
  a.download = `${name.replace(/[/\\:*?"<>|]/g, '_')}.json`; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function installLibrary({scenePayload, loadScene, loadTerrain, setName, message}) {
  let folders = [], loaded = {}, terrainName = '', working = false;
  const kind = () => $('resource-kind').value;
  const folder = () => $('library-folder').value;
  const url = (id) => `/api/library/${enc(folder())}/${kind()}${id === undefined ? '' : '/'+enc(id)}`;
  const tell = (text, error = false) => {
    $('library-message').textContent = text; $('library-message').classList.toggle('error', error);
  };
  async function guard(fn) {
    if (working) return;
    working = true;
    try {await fn();} catch(e) {tell(e.message, true);} finally {working = false; controls();}
  }
  function controls() {
    const entry = folders.find(f => f.id === folder());
    const current = loaded[kind()];
    $('resource-save').disabled = working || !entry || entry.readonly;
    $('resource-overwrite').disabled = working || !current || current.folder !== folder() || entry?.readonly;
    $('folder-remove').disabled = working || ['user','builtin'].includes(folder());
    $('folder-path').textContent = entry ? `${entry.path}${entry.readonly ? ' · 原厂只读' : ''}` : '';
    $('terrain-name-row').hidden = kind() !== 'scenes';
  }
  async function refreshFolders(selected = folder() || 'user') {
    folders = await api('/api/library');
    $('library-folder').replaceChildren(...folders.map(f => new Option(f.name, f.id)));
    $('library-folder').value = folders.some(f => f.id === selected) ? selected : 'user';
  }
  async function refresh() {
    controls();
    const items = await api(url());
    const selectedFolder = folder(), selectedKind = kind();
    $('scene-list').replaceChildren();
    if (!items.length) $('scene-list').textContent = '此目录暂无资源。';
    for (const item of items) {
      const row = document.createElement('div'), info = document.createElement('div');
      row.className = 'scene-item';
      const title = document.createElement('strong'), detail = document.createElement('small');
      title.textContent = item.name;
      detail.textContent = `${item.readonly ? '原厂' : new Date(item.modified*1000).toLocaleString()} · ${item.description || item.terrain_ref || item.id+'.json'}`;
      info.append(title, detail);
      const load = document.createElement('button'); load.textContent = '加载 / 编辑';
      load.onclick = () => guard(async () => {
        const data = await api(`/api/library/${enc(selectedFolder)}/${selectedKind}/${enc(item.id)}`);
        if (selectedKind === 'scenes') {
          await loadScene(data);
          terrainName = item.terrain_ref ? item.terrain_ref.slice(5,-5) : data.name;
          loaded.maps = item.terrain_ref ? {folder: selectedFolder, id: terrainName} : undefined;
        } else {loadTerrain(data); terrainName = data.name;}
        loaded[selectedKind] = {folder: selectedFolder, id: item.id};
        $('resource-name').value = data.name;
        $('terrain-name').value = terrainName;
        $('library').close();
        if (selectedKind === 'scenes') message('场景已加载；编辑后可在管理器覆盖保存或另存。');
      });
      const remove = document.createElement('button'); remove.textContent = '删除'; remove.disabled = item.readonly;
      remove.onclick = () => guard(async () => {
        if (!confirm(`删除「${item.name}」？磁盘文件将被移除。`)) return;
        await api(`/api/library/${enc(selectedFolder)}/${selectedKind}/${enc(item.id)}`, 'DELETE');
        if (loaded[selectedKind]?.id === item.id && loaded[selectedKind]?.folder === selectedFolder) delete loaded[selectedKind];
        await refresh(); tell(`已删除「${item.name}」`);
      });
      const actions = document.createElement('div'); actions.className = 'paired'; actions.append(load,remove);
      row.append(info, actions); $('scene-list').append(row);
    }
  }
  function terrain(name) {
    const s = scenePayload();
    return {schema_version:1, kind:'terrain', name, bounds:s.config.map.bounds,
      boxes:s.config.map.boxes, polygons:s.polygons, polylines:s.polylines};
  }
  async function save(overwrite) {
    const name = $('resource-name').value.trim();
    if (!name) throw Error('请填写保存名称');
    const current = loaded[kind()];
    if (overwrite && (!current || current.folder !== folder())) throw Error('请先加载要覆盖的资源');
    if (overwrite && !confirm(`覆盖「${current.id}」${name !== current.id ? `并重命名为「${name}」` : ''}？`)) return;
    let payload;
    if (kind() === 'scenes') {
      const mapName = $('terrain-name').value.trim();
      if (!mapName) throw Error('请填写地形名称');
      payload = {scene:{...scenePayload(), name}, map_name:mapName};
    } else payload = terrain(name);
    const result = await api(url(overwrite ? current.id : undefined), overwrite ? 'PUT' : 'POST', payload);
    loaded[kind()] = {folder: folder(), id: result.id};
    if (kind() === 'scenes') {
      setName(result.name); terrainName = $('terrain-name').value.trim();
      loaded.maps = {folder: folder(), id: terrainName};
    } else {terrainName = name; $('terrain-name').value = name;}
    await refresh(); tell(`已保存「${result.name}」${kind() === 'scenes' ? '，地形单独存储' : ''}。`);
  }
  async function open() {
    await guard(async () => {
      await refreshFolders();
      const s = scenePayload();
      $('resource-name').value = kind() === 'scenes' ? s.name : (terrainName || s.name);
      $('terrain-name').value = terrainName || s.name;
      tell(''); await refresh();
      if (!$('library').open) $('library').showModal();
    });
  }
  $('library-open').onclick = open;
  $('save').onclick = () => {$('resource-kind').value = 'scenes'; open();};
  $('library-close').onclick = () => $('library').close();
  $('resource-kind').onchange = () => guard(async () => {
    $('resource-name').value = kind() === 'scenes' ? scenePayload().name : ($('terrain-name').value || scenePayload().name);
    tell(''); await refresh();
  });
  $('library-folder').onchange = () => guard(async () => {tell(''); await refresh();});
  $('resource-save').onclick = () => guard(() => save(false));
  $('resource-overwrite').onclick = () => guard(() => save(true));
  $('resource-export').onclick = () => guard(async () => {
    const name = $('resource-name').value.trim() || scenePayload().name;
    const data = kind() === 'maps' ? terrain(name) : await api('/api/export','POST',{...scenePayload(),name});
    download(data, name); tell('已导出；场景导出包含地形，可独立导入或重放。');
  });
  $('folder-add').onclick = () => guard(async () => {
    const name = $('folder-name').value.trim(), path = $('folder-location').value.trim();
    await api('/api/library', 'POST', {name, ...(path ? {path} : {})});
    await refreshFolders(name); await refresh(); tell('目录已登记，下次启动仍可使用。');
  });
  $('folder-remove').onclick = () => guard(async () => {
    if (!confirm('移除此目录登记？磁盘文件保留。')) return;
    await api(`/api/library/${enc(folder())}`, 'DELETE');
    await refreshFolders('user'); await refresh(); tell('目录登记已移除，文件保留。');
  });
  return {reset() {loaded = {}; terrainName = '';},
    importTerrain(terrain) {delete loaded.maps; terrainName = terrain.name; loadTerrain(terrain);}};
}
