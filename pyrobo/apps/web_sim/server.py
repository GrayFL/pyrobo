"""FastAPI transport. Each WebSocket owns one isolated simulation worker."""
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pyrobo.apps.sim_common.session import Action, ControlInput, Worker
from pyrobo.core import pose2d
from pyrobo.core.geometry import planar_state
from pyrobo.geometry.cache import SnapshotIndex
from .scene import MAX_SCENE_BYTES, Scene, SceneSession, SceneStore

STATIC = Path(__file__).parent/'static'


class Drive(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    throttle: float = Field(default=0., ge=-1, le=1)
    turn: float = Field(default=0., ge=-1, le=1)
    engaged: bool = False
    source: Literal['keyboard', 'virtual', 'gamepad'] = 'keyboard'
    focused: bool = True


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def snapshot_data(snapshot, generation):
    s, m = snapshot.state, snapshot.motion
    return clean({
        'type': 'state', 'generation': generation, 'ticks': snapshot.ticks, 'time': s.timestamp,
        'pose': planar_state(s.pose), 'speed': s.velocity.linear, 'omega': s.velocity.angular,
        'steering': s.steering, 'mode': str(snapshot.mode), 'policy': str(snapshot.policy),
        'goal': planar_state(snapshot.goal) if snapshot.goal is not None else None,
        'path': snapshot.path, 'trail': snapshot.trail,
        'forecast': m.trajectory.poses.position[:, :2] if m and m.trajectory else [],
        'target': snapshot.progress.target.translation[:2] if snapshot.progress else None,
        'status': snapshot.navigation_status, 'planner_status': snapshot.planner_status,
        'safe': bool(m.safe) if m else None, 'assessment': str(m.safety_assessment) if m else '',
        'clearance': m.diagnostics.clearance_m if m else None,
        'candidates': m.diagnostics.candidate_count if m else 0,
        'feasible': m.diagnostics.feasible_count if m else 0,
        'elapsed_ms': snapshot.elapsed_ms, 'collided': snapshot.collided,
        'notice': snapshot.notice, 'fault': snapshot.fault,
    })


class Room:
    def __init__(self):
        self.worker = None
        self.generation = 0

    def replace(self, scene):
        session = SceneSession(scene)  # validate/build before replacing the current experiment
        index = SnapshotIndex().get(session.cloud)
        if self.worker is not None:
            self.worker.close()
        self.scene, self.world, self.goal_index = scene, session.world, index
        self.footprint = session.footprint
        self.worker = Worker(session)
        self.worker.publish(ControlInput(focused=False))
        self.generation += 1
        self.worker.start()

    def enqueue(self, action):
        if not self.worker.enqueue(action):
            raise ValueError('command queue is busy; retry after the next state update')

    def action(self, data):
        kind = data.get('action')
        if kind in ('auto', 'manual', 'paused'):
            self.enqueue(Action('mode', kind))
        elif kind in ('estop', 'release', 'reset'):
            self.enqueue(Action(kind))
        elif kind == 'policy':
            if data.get('value') not in ('teleop_stop', 'teleop_avoid'):
                raise ValueError('invalid manual policy')
            self.enqueue(Action('policy', data['value']))
        elif kind == 'goal':
            values = np.asarray(data.get('value'), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError('goal must be finite [x, y, yaw]')
            pose = pose2d(*values, body='goal')
            if (self.world.collides(pose, self.footprint, padding=self.footprint.margin) or
                    self.goal_index.clearances(values[None], self.footprint, self.world.bounds)[0] <= 0):
                raise ValueError('目标与障碍或边界相交，请为矩形车体留出空间')
            self.enqueue(Action('goal', values.tolist()))
        else:
            raise ValueError('unknown action')

    def close(self):
        if self.worker is not None:
            self.worker.publish(ControlInput(focused=False))
            self.worker.close()


def create_app(*, scenes_dir=None, initial_scene=None, max_clients=8):
    initial_scene = initial_scene or Scene(name='交错墙 · 绕行测试')
    rooms = set()

    @asynccontextmanager
    async def lifespan(app):
        app.state.store = SceneStore(scenes_dir or Path(os.environ.get('XDG_DATA_HOME', Path.home()/'.local/share'))/'pyrobo'/'scenes')
        yield
        await asyncio.gather(*(asyncio.to_thread(room.close) for room in tuple(rooms)))

    app = FastAPI(title='Pyrobo Motion Lab', lifespan=lifespan)
    app.mount('/static', StaticFiles(directory=STATIC), name='static')

    @app.get('/')
    def index():
        return FileResponse(STATIC/'index.html', headers={'Cache-Control': 'no-cache'})

    @app.get('/api/bootstrap')
    def bootstrap():
        return {'scene': initial_scene.model_dump(), 'max_clients': max_clients}

    @app.get('/api/scenes')
    def list_scenes():
        return app.state.store.list()

    @app.get('/api/scenes/{identifier}')
    def load_scene(identifier: str):
        try:
            return app.state.store.load(identifier).model_dump()
        except FileNotFoundError:
            raise HTTPException(404, 'scene not found')
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(422, str(error))

    @app.post('/api/scenes', status_code=201)
    async def save_scene(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_SCENE_BYTES:
                raise HTTPException(413, 'scene exceeds 2 MB')
        try:
            scene = await asyncio.to_thread(Scene.model_validate_json, body)
            identifier = await asyncio.to_thread(app.state.store.save, scene)
            return {'id': identifier, 'name': scene.name}
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(422, str(error))

    @app.websocket('/ws')
    async def websocket(ws: WebSocket):
        origin = ws.headers.get('origin')
        if origin and urlsplit(origin).netloc != ws.headers.get('host'):
            await ws.close(code=1008, reason='same-origin WebSocket required')
            return
        if len(rooms) >= max_clients:
            await ws.close(code=1013, reason='simulation capacity reached')
            return
        await ws.accept()
        room, send_lock = Room(), asyncio.Lock()
        rooms.add(room)

        async def send(data):
            async with send_lock:
                await asyncio.wait_for(ws.send_json(data), timeout=3.)

        async def stream():
            try:
                while True:
                    await send(snapshot_data(room.worker.snapshot(), room.generation))
                    await asyncio.sleep(.1)
            except (WebSocketDisconnect, RuntimeError, asyncio.TimeoutError):
                await ws.close()

        sender = None
        try:
            await asyncio.to_thread(room.replace, initial_scene.model_copy(deep=True))
            await send({'type': 'hello', 'scene': room.scene.model_dump(), 'generation': room.generation})
            sender = asyncio.create_task(stream())
            while True:
                raw = await ws.receive_text()
                if len(raw) > MAX_SCENE_BYTES:
                    await ws.close(code=1009)
                    break
                identifier = None
                try:
                    message = json.loads(raw)
                    if not isinstance(message, dict):
                        raise ValueError('message must be an object')
                    identifier, kind = message.get('id'), message.get('type')
                    if kind == 'control':
                        drive = Drive.model_validate(message.get('input', {}))
                        room.worker.publish(ControlInput(**drive.model_dump()))
                        continue
                    if kind == 'apply':
                        # Suspend the old experiment while validation/construction runs.
                        room.worker.publish(ControlInput(focused=False))
                        scene = await asyncio.to_thread(Scene.model_validate, message.get('scene'))
                        await asyncio.to_thread(room.replace, scene)
                        await send({'type': 'reply', 'id': identifier, 'ok': True,
                                    'scene': scene.model_dump(), 'generation': room.generation})
                    elif kind == 'validate':
                        scene = await asyncio.to_thread(Scene.model_validate, message.get('scene'))
                        await send({'type': 'reply', 'id': identifier, 'ok': True, 'scene': scene.model_dump()})
                    elif kind == 'action':
                        room.action(message)
                        await send({'type': 'reply', 'id': identifier, 'ok': True})
                    else:
                        raise ValueError('unknown message type')
                except (ValueError, TypeError, KeyError, ValidationError) as error:
                    await send({'type': 'reply', 'id': identifier, 'ok': False, 'error': str(error)[:1500]})
        except (WebSocketDisconnect, RuntimeError, asyncio.TimeoutError):
            pass
        finally:
            if sender:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            await asyncio.to_thread(room.close)
            rooms.discard(room)

    return app
