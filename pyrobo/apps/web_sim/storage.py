"""Named terrain/scene files and a persistent catalog of registered directories."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
import unicodedata

MAX_BYTES = 2_000_000
# One application process, including concurrent FastAPI worker threads.
_LOCK = RLock()


def name_key(name):
    if not isinstance(name, str) or not name or name != name.strip() or len(name) > 80:
        raise ValueError('名称须为 1–80 个字符，首尾不能有空格')
    if (name in ('.', '..') or name.startswith('.') or name.endswith('.') or
            re.search(r'[/\\:*?"<>|\x00-\x1f\x7f]', name) or
            re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', name)):
        raise ValueError('名称不能包含路径、控制字符或系统保留名称')
    return unicodedata.normalize('NFC', name).casefold()


def read_json(path):
    if path.is_symlink():
        raise ValueError('资源文件不能是符号链接')
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('resource exceeds 2 MB')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('resource must be a JSON object')
    return data


def atomic_json(path, data, *, overwrite=False):
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    if len(payload) > MAX_BYTES:
        raise ValueError('resource exceeds 2 MB')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('资源文件不能是符号链接')
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            if overwrite:
                os.replace(temporary, path)
            else:
                # Atomic no-clobber creation, including simultaneous requests.
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class SceneStore:
    def __init__(self, directory, *, readonly=False):
        self.directory = Path(directory).expanduser().resolve()
        self.readonly = readonly
        if not readonly:
            self.directory.mkdir(parents=True, exist_ok=True)

    def folder(self, kind='scenes'):
        if kind not in ('scenes', 'maps'):
            raise ValueError('resource kind must be scenes or maps')
        folder = self.directory if kind == 'scenes' else self.directory/'maps'
        if folder.is_symlink():
            raise ValueError('maps directory cannot be a symbolic link')
        return folder

    def path(self, identifier, kind='scenes'):
        name_key(identifier)
        return self.folder(kind)/f'{identifier}.json'

    def writable(self):
        if self.readonly:
            raise PermissionError('原厂目录只读；请另存到用户目录')

    def check_name(self, name, kind, *, existing=None):
        key = name_key(name)
        for path in self.folder(kind).glob('*.json'):
            if path.stem == existing or path.name.startswith('.'):
                continue
            conflict = unicodedata.normalize('NFC', path.stem).casefold() == key
            try:
                data = read_json(path)
                conflict |= unicodedata.normalize('NFC', str(data.get('name', path.stem))).casefold() == key
            except (OSError, ValueError):
                pass  # Corrupt/unrelated resources cannot block other names.
            if conflict:
                raise FileExistsError(f'「{name}」已存在，请选择覆盖或更换名称')

    def save_map(self, terrain, *, identifier=None):
        from .scene import Terrain
        terrain = Terrain.model_validate(terrain) if isinstance(terrain, dict) else terrain
        self.writable()
        with _LOCK:
            if identifier is not None:
                path = self.path(identifier, 'maps')
                read_json(path)
                if terrain.name != identifier:
                    raise ValueError('地图重命名请另存；原有场景仍引用原地图')
            self.check_name(terrain.name, 'maps', existing=identifier)
            atomic_json(self.path(terrain.name, 'maps'), terrain.model_dump(mode='json'), overwrite=identifier is not None)
        return terrain.name

    def save(self, scene, *, identifier=None, map_name=None):
        self.writable()
        map_name = map_name or scene.name
        with _LOCK:
            if identifier is not None:
                read_json(self.path(identifier))
            self.check_name(scene.name, 'scenes', existing=identifier)
            document = scene.document(map_name)
            terrain = document.pop('terrain')
            map_path = self.path(map_name, 'maps')
            if map_path.exists():
                from .scene import Terrain
                if Terrain.model_validate(read_json(map_path)).model_dump(mode='json') != terrain:
                    raise FileExistsError(f'地形「{map_name}」已有不同内容；请先覆盖保存地形，或使用新的地形名')
            else:
                self.save_map(scene.terrain(map_name))
            document['terrain_ref'] = f'maps/{map_name}.json'
            target = self.path(scene.name)
            atomic_json(target, document, overwrite=identifier == scene.name)
            if identifier is not None and identifier != scene.name:
                self.path(identifier).unlink()
        return scene.name

    def load_map(self, identifier):
        from .scene import Terrain
        return Terrain.model_validate(read_json(self.path(identifier, 'maps')))

    def load(self, identifier):
        return load_scene_file(self.path(identifier))

    def list(self, kind='scenes'):
        result = []
        for path in self.folder(kind).glob('*.json'):
            try:
                name_key(path.stem)
                data = read_json(path)
                if kind == 'scenes' and ('config' not in data or data.get('kind') == 'terrain'):
                    continue
                if kind == 'maps' and data.get('kind') != 'terrain':
                    continue
                result.append({'id': path.stem, 'name': data['name'], 'description': data.get('description', ''),
                               'modified': path.stat().st_mtime, 'readonly': self.readonly,
                               'terrain_ref': data.get('terrain_ref')})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(result, key=lambda item: item['name'].casefold())

    def delete(self, identifier, kind='scenes'):
        self.writable()
        with _LOCK:
            path = self.path(identifier, kind)
            data = read_json(path)
            if (kind == 'maps' and data.get('kind') != 'terrain') or (kind == 'scenes' and ('config' not in data or data.get('kind') == 'terrain')):
                raise ValueError('file is not a managed resource of this kind')
            if kind == 'maps':
                ref = f'maps/{identifier}.json'
                if any(item.get('terrain_ref') == ref for item in self.list()):
                    raise FileExistsError('该地形仍被场景引用，请先删除或改存相关场景')
            path.unlink()


def load_scene_file(path):
    from .scene import Scene, Terrain
    path = Path(path)
    data = read_json(path)
    if 'terrain_ref' in data:
        data = deepcopy(data)
        ref = data.pop('terrain_ref')
        # Files are portable within a library, with a deliberately narrow reference grammar.
        if not isinstance(ref, str) or not ref.startswith('maps/') or not ref.endswith('.json'):
            raise ValueError('invalid terrain reference')
        name = ref[5:-5]
        store = SceneStore(path.parent, readonly=True)
        data['terrain'] = Terrain.model_validate(read_json(store.path(name, 'maps'))).model_dump(mode='json')
    return Scene.model_validate(data)


class Library:
    def __init__(self, directory, builtin):
        self.root = Path(directory).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root/'.library.json'
        self.builtin = Path(builtin).resolve()
        self.folders = read_json(self.manifest) if self.manifest.exists() else {}
        if not isinstance(self.folders, dict):
            raise ValueError('invalid library manifest')

    def list_folders(self):
        return [dict(id='builtin', name='原厂场景', path=str(self.builtin), readonly=True),
                dict(id='user', name='用户场景', path=str(self.root), readonly=False),
                *(dict(id=name, name=name, path=path, readonly=False) for name, path in self.folders.items())]

    def store(self, folder):
        entry = next((f for f in self.list_folders() if f['id'] == folder), None)
        if entry is None:
            raise FileNotFoundError('folder not registered')
        return SceneStore(entry['path'], readonly=entry['readonly'])

    def add_folder(self, name, path=None):
        with _LOCK:
            key = name_key(name)
            if any(key in (name_key(f['id']), name_key(f['name'])) for f in self.list_folders()):
                raise FileExistsError('目录名称已存在')
            target = (Path(path).expanduser() if path else self.root/'categories'/name).resolve()
            if any(target == Path(f['path']) for f in self.list_folders()) or target == self.builtin or self.builtin in target.parents:
                raise FileExistsError('目录路径已登记，或位于原厂目录中')
            target.mkdir(parents=True, exist_ok=True)
            updated = {**self.folders, name: str(target)}
            atomic_json(self.manifest, updated, overwrite=True)
            self.folders = updated

    def remove_folder(self, name):
        with _LOCK:
            if name not in self.folders:
                raise ValueError('只能移除自定义目录登记')
            updated = {k: v for k, v in self.folders.items() if k != name}
            atomic_json(self.manifest, updated, overwrite=True)
            self.folders = updated
