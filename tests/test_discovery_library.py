"""Registration, real disk lifecycle and sensor visibility contracts (fast)."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import pytest
pytest.importorskip('fastapi')
from fastapi.testclient import TestClient

from pyrobo import DifferentialDrive, DWAConfig, DynamicWindow, pose2d
from pyrobo.planning import RRTConnect, RRTConnectConfig
from pyrobo.registry import (LOCAL_OPTIMIZERS, GLOBAL_PLANNERS, KINEMATIC_MODELS,
                             register_local_optimizer, register_global_planner, register_kinematic_model)
from pyrobo.apps.sim_common.catalog import catalog
from pyrobo.apps.sim_common.session import Action
from pyrobo.apps.web_sim.scene import Scene, SceneSession, Terrain, SceneStore
from pyrobo.apps.web_sim.server import create_app
from pyrobo.apps.web_sim.storage import Library, load_scene_file


def case(**kwargs):
    return Scene(name='仓库实验', config={'map': {'preset': 'custom', 'bounds': [-5, 5, -4, 4],
                        'start': [-3, 0, 0], 'goal': [3, 0, 0], 'boxes': []}}, **kwargs)


def test_import_registration_drives_config_catalog_and_actual_assembly():
    # Register after apps/config was imported: no frozen list of defaults.
    @dataclass(frozen=True)
    class CustomConfig(DWAConfig):
        pass

    @register_local_optimizer('test_local', config=CustomConfig, label='测试局部', family='dwa')
    class CustomLocal(DynamicWindow):
        pass

    @dataclass(frozen=True)
    class GlobalConfig(RRTConnectConfig):
        pass

    @register_global_planner('test_global', config=GlobalConfig, label='测试全局')
    class CustomGlobal(RRTConnect):
        pass

    @register_kinematic_model('test_model', label='测试车型', family='differential')
    @dataclass(frozen=True)
    class CustomModel(DifferentialDrive):
        pass

    try:
        payload = case().model_dump()
        payload['config']['backend'].update(local_optimizer='test_local', planner='test_global',
                                             model='test_model', adaptive_clearance=False)
        scene = Scene.model_validate(payload)
        run = SceneSession(scene)
        assert isinstance(run.model, CustomModel)
        assert isinstance(run.nav.optimizer, CustomLocal)
        assert isinstance(run.nav.planner, CustomGlobal)
        assert scene.config['test_local']['prediction_time'] == 1.8
        assert any(item['id'] == 'test_local' for item in catalog()['optimizers'])
        with pytest.raises(ValueError, match='duplicate'):
            register_local_optimizer('test_local', config=CustomConfig)(DynamicWindow)
        assert run.tick().motion.safe
    finally:
        LOCAL_OPTIMIZERS.pop('test_local')
        GLOBAL_PLANNERS.pop('test_global')
        KINEMATIC_MODELS.pop('test_model')


def test_scene_files_reference_reusable_terrain_and_portable_export(tmp_path):
    store = SceneStore(tmp_path)
    first = case()
    second = first.model_copy(update={'name': '第二场景'})
    store.save(first, map_name='仓库地形')
    store.save(second, map_name='仓库地形')
    saved = json.loads((tmp_path/'仓库实验.json').read_text())
    assert saved['terrain_ref'] == 'maps/仓库地形.json'
    assert set(saved['config']['map']) == {'start', 'goal'}
    assert 'polygons' not in saved and 'terrain' not in saved
    terrain = json.loads((tmp_path/'maps/仓库地形.json').read_text())
    assert 'config' not in terrain and 'start' not in terrain and 'unknown_obstacles' not in terrain
    assert load_scene_file(tmp_path/'仓库实验.json').model_dump() == first.model_dump()
    assert Scene.model_validate(first.document()).model_dump() == first.model_dump()
    with pytest.raises(FileExistsError, match='引用'):
        store.delete('仓库地形', 'maps')
    store.delete(first.name)
    store.delete(second.name)
    store.delete('仓库地形', 'maps')


def test_duplicate_races_rename_and_explicit_overwrite(tmp_path):
    store = SceneStore(tmp_path)
    def save():
        try:
            return store.save(case())
        except FileExistsError:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: save(), range(2))) == ['conflict', '仓库实验']
    updated = case().model_copy(update={'name': '重命名', 'description': '修改内容'})
    assert store.save(updated, identifier='仓库实验', map_name='仓库实验') == '重命名'
    assert not (tmp_path/'仓库实验.json').exists()
    assert store.load('重命名').description == '修改内容'
    # Equivalent Unicode and case spellings cannot quietly create duplicates.
    a = case().model_copy(update={'name':'Demo'})
    store.save(a)
    with pytest.raises(FileExistsError):
        store.save(a.model_copy(update={'name':'demo'}))
    for name in ('../escape', 'a/b', 'a\\b', 'NUL', '.hidden', 'bad\x00name'):
        with pytest.raises(ValueError):
            store.save(a.model_copy(update={'name': name}))


def test_legacy_uuid_library_can_be_read_and_renamed(tmp_path):
    identifier = 'a'*32
    legacy = case().model_dump()
    legacy['name'] = '旧名称 / 含路径分隔符'
    (tmp_path/f'{identifier}.json').write_text(json.dumps(legacy))
    store = SceneStore(tmp_path)
    assert store.list()[0]['name'] == legacy['name']
    old = store.load(identifier)
    old.name = '已迁移'
    store.save(old, identifier=identifier)
    assert not (tmp_path/f'{identifier}.json').exists()
    assert store.load('已迁移').name == '已迁移'


def test_folder_registration_persists_and_http_lifecycle(tmp_path):
    root, external = tmp_path/'library', tmp_path/'external'
    app = create_app(scenes_dir=root, initial_scene=case())
    with TestClient(app) as c:
        assert c.post('/api/library', json={'name':'厂房','path':str(external)}).status_code == 201
        base = '/api/library/厂房/scenes'
        payload = {'scene':case().model_dump(), 'map_name':'共享地形'}
        assert c.post(base,json=payload).status_code == 201
        assert c.post(base,json=payload).status_code == 409
        payload['scene']['description'] = '覆盖修改'
        assert c.put(base+'/仓库实验',json=payload).status_code == 200
        assert c.get(base+'/仓库实验').json()['description'] == '覆盖修改'
        assert c.post('/api/library/builtin/scenes',json=payload).status_code == 403
        assert c.delete('/api/library/厂房/maps/共享地形').status_code == 409
    with TestClient(create_app(scenes_dir=root)) as c:
        assert any(f['id']=='厂房' for f in c.get('/api/library').json())
        assert c.delete('/api/library/厂房/scenes/仓库实验').status_code == 200
        assert c.delete('/api/library/厂房/maps/共享地形').status_code == 200
        assert c.delete('/api/library/厂房').status_code == 200
        assert external.is_dir()


def test_unknown_obstacles_only_enter_cloud_as_visible_lidar_returns():
    scene = case(unknown_obstacles={'boxes': [[0,.2,-.5,.5]]}, lidar={'max_range':1.5, 'beams':360},
                 experiment={'events':[{'kind':'freeze_cloud','time_s':2.}, {'kind':'resume_cloud','time_s':3.}]})
    run = SceneSession(scene)
    known = run.known_cloud.points.copy()
    assert not np.any((run.cloud.points[:,0] >= 0) & (run.cloud.points[:,0] <= .2) & (np.abs(run.cloud.points[:,1]) < .5))
    # Still a real solid even before sensing it.
    assert run.world.collides(pose2d(.1,0),run.footprint)
    assert not run.known_world.collides(pose2d(.1,0),run.footprint)
    run.sim.state = replace(run.sim.state, pose=pose2d(-1,0), timestamp=1.)
    run.observation()
    scan = np.asarray(run.scan_points)
    assert np.any(np.isclose(scan[:,0],0) & (np.abs(scan[:,1]) < .5))
    assert run.cloud.timestamp == 1.
    np.testing.assert_array_equal(run.cloud.points[:len(known)], known)
    run.sim.state = replace(run.sim.state, timestamp=2.)
    run.observation()
    assert run.cloud.timestamp == 1.  # freezing cannot rejuvenate an older scan
    run.sim.state = replace(run.sim.state, timestamp=3.)
    run.observation()
    assert run.cloud.timestamp == 3.
    run.apply(Action('reset'))
    assert run.sim.state.timestamp == 0.
    assert len(run.cloud.points) < len(known)+100


def test_lidar_respects_occlusion_and_does_not_reveal_back_surfaces():
    payload = case(unknown_obstacles={'boxes': [[1,1.2,-.4,.4]]}, lidar={'max_range':8}).model_dump()
    payload['config']['map']['boxes'] = [[0,.2,-1,1]]
    run = SceneSession(Scene.model_validate(payload))
    scan = np.asarray(run.scan_points)
    assert np.any(np.isclose(scan[:,0],0))
    assert not np.any((scan[:,0] >= 1) & (scan[:,0] <= 1.2) & (np.abs(scan[:,1]) <= .4))


def test_map_reference_cannot_escape_or_follow_symlink(tmp_path):
    store = SceneStore(tmp_path/'library')
    store.save(case())
    path = store.path('仓库实验')
    data = json.loads(path.read_text())
    data['terrain_ref'] = 'maps/../../outside.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        store.load('仓库实验')
    external = tmp_path/'outside.json'; external.write_text('{}')
    store.path('link').symlink_to(external)
    with pytest.raises(ValueError):
        store.delete('link')
    assert external.exists()


def test_manifest_and_malformed_files_do_not_block_default_directory(tmp_path):
    library = Library(tmp_path/'user', tmp_path/'builtin')
    library.add_folder('分类')
    store = library.store('user')
    (store.directory/'broken.json').write_text('{broken')
    (store.directory/'array.json').write_text('[]')
    assert store.save(case()) == case().name
    assert [item['name'] for item in store.list()] == [case().name]
    with pytest.raises(FileExistsError):
        store.save(case().model_copy(update={'name':'broken'}))
    assert store.load(case().name).name == case().name


def test_optimizer_with_its_own_config_does_not_need_dwa_parameter_names():
    @dataclass(frozen=True)
    class OwnConfig:
        speed: float = .4

    @register_local_optimizer('own_config', config=OwnConfig, label='独立参数')
    class OwnOptimizer(DynamicWindow):
        def __init__(self, config, *args, **kwargs):
            super().__init__(DWAConfig(max_speed=config.speed), *args, **kwargs)

    try:
        payload = case().model_dump()
        payload['config']['backend']['local_optimizer'] = 'own_config'
        scene = Scene.model_validate(payload)
        assert scene.config['own_config'] == {'speed': .4}
        assert SceneSession(scene).tick().motion.safe
    finally:
        LOCAL_OPTIMIZERS.pop('own_config')
