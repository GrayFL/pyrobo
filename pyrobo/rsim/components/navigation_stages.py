"""Compute acknowledgements tied to exact navigation publications.

Reader-local sequence cursors are deliberately absent from revisions. SampleId
survives transport, while a receiving Signal may assign a different cursor.
"""
from dataclasses import asdict
import json
import os

from rsim import StageEndpoint

from ..packets import latest
from ..scheduling import NavigationSnapshot, source_context


def publication(frame):
    if frame is None or frame.sample_id is None:
        raise ValueError('staged navigation requires an identified publication')
    return {'sample_id': asdict(frame.sample_id), 'clock': frame.clock, 'stamp_ns': frame.stamp_ns}


def snapshot_revision(snapshot):
    return {'state': publication(snapshot.state), 'map': publication(snapshot.map),
            'mission': publication(snapshot.mission), 'version': list(snapshot.input_revision)}


def encode_revision(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


async def wait_publication(signal, reference):
    if (not isinstance(reference, dict) or set(reference) != {'sample_id', 'clock', 'stamp_ns'}
            or not isinstance(reference['sample_id'], dict)):
        raise ValueError('invalid publication revision')
    cursor = 0
    while True:
        frames = signal.frames
        for frame in reversed(frames):
            if publication(frame) == reference:
                return frame
        if frames:
            cursor = frames[-1].sequence
            wanted, newest = reference['sample_id'], frames[-1].sample_id
            if (wanted.get('producer_instance_id') == newest.producer_instance_id
                    and wanted.get('canonical_port_id') == newest.canonical_port_id
                    and newest.publication_sequence >= wanted.get('publication_sequence', 0)):
                raise ValueError('required publication is missing or its revision differs')
        await signal.get(after=cursor)


class _NavigationStage(StageEndpoint):
    def __init__(self, task_id, algorithm, states, maps, missions, *, timeout=5.):
        super().__init__(task_id, algorithm, inputs=(states, maps, missions), timeout=timeout)
        self.algorithm = algorithm
        self.states, self.maps, self.missions = states, maps, missions

    async def open(self):
        if (self.algorithm._runtime is not self._runtime or self.algorithm._binding is not None
                or self.algorithm._canonical is not None):
            raise ValueError('a navigation stage and its algorithm must be co-located in one Runtime')
        await super().open()

    async def capture(self, request, revision):
        spec = revision['snapshot']
        # Keep each wait within the stage transaction. A rejected revision must
        # not leave sibling frame waiters alive across reset or close.
        frames = [await wait_publication(signal, spec[key]) for key, signal in
                  (('state', self.states), ('map', self.maps), ('mission', self.missions))]
        snapshot = NavigationSnapshot(*frames)
        snapshot.require_context(request.context)
        if list(snapshot.input_revision) != spec['version']:
            raise ValueError('actual map/mission revision differs from the stage request')
        self.check_current(revision)
        return snapshot

    def check_current(self, revision):
        for key, signal in (('state', self.states), ('map', self.maps), ('mission', self.missions)):
            if publication(latest(signal)) != revision['snapshot'][key]:
                raise ValueError('navigation snapshot changed during stage computation')

    def result(self, **values):
        return dict(pid=os.getpid(), **values)


class LocalNavigationStage(_NavigationStage):
    def __init__(self, algorithm, *, timeout=5.):
        super().__init__('local', algorithm, algorithm.states, algorithm.maps, algorithm.missions, timeout=timeout)

    async def compute(self, request):
        revision = json.loads(request.input_revision)
        snapshot = await self.capture(request, revision)
        if revision['route'] is not None:
            route = await wait_publication(self.algorithm.routes, revision['route'])
            previous = source_context(route)
            if (previous.session, previous.episode) != (request.context.session, request.context.episode):
                raise ValueError('route belongs to a different source generation')
            if publication(latest(self.algorithm.routes)) != revision['route']:
                raise ValueError('route changed before local computation')
        previous_request = latest(self.algorithm.requests)
        result = await self.algorithm.step_once(request.context, snapshot)
        self.check_current(revision)
        if result is None:
            raise ValueError('local stage did not compute the requested source step')
        proposal, planning = latest(self.algorithm.proposals), latest(self.algorithm.requests)
        if source_context(proposal) != request.context:
            raise ValueError('local stage did not publish the requested proposal')
        new_request = planning is not None and (previous_request is None
                                                or planning.sample_id != previous_request.sample_id)
        if new_request and source_context(planning) != request.context:
            raise ValueError('local stage published a planning request for another context')
        return self.result(proposal=publication(proposal),
                           request=publication(planning) if new_request else None,
                           request_id=planning.data['id'] if new_request else None,
                           status=str(self.algorithm.navigator.status), computations=self.algorithm.sequence)


class GlobalPlanningStage(_NavigationStage):
    def __init__(self, algorithm, states, missions, *, timeout=5.):
        super().__init__('global', algorithm, states, algorithm.maps, missions, timeout=timeout)

    async def compute(self, request):
        revision = json.loads(request.input_revision)
        await self.capture(request, revision)
        if revision['request'] is None:
            return self.result(route=None, request_id=None)
        planning = await wait_publication(self.algorithm.requests, revision['request'])
        if source_context(planning) != request.context or planning.data['id'] != revision['request_id']:
            raise ValueError('global stage planning request identity differs')
        result = await self.algorithm.step_once(request.context, planning)
        self.check_current(revision)
        route = latest(self.algorithm.routes)
        if result is None or route is None or route.data['id'] != revision['request_id']:
            raise ValueError('global stage did not publish the requested route')
        if source_context(route) != request.context:
            raise ValueError('global stage route context differs')
        return self.result(route=publication(route), request_id=revision['request_id'])


class ExecutionStage(_NavigationStage):
    def __init__(self, algorithm, *, actuator_horizon, timeout=5.):
        super().__init__('execution', algorithm, algorithm.states, algorithm.maps, algorithm.missions, timeout=timeout)
        self.actuator_horizon = actuator_horizon

    async def compute(self, request):
        revision = json.loads(request.input_revision)
        snapshot = await self.capture(request, revision)
        proposal = await wait_publication(self.algorithm.proposals, revision['proposal'])
        if source_context(proposal) != request.context:
            raise ValueError('execution proposal belongs to another source context')
        if publication(latest(self.algorithm.proposals)) != revision['proposal']:
            raise ValueError('proposal changed before execution')
        result = await self.algorithm.step_once(request.context, snapshot,
                                                actuator_horizon=self.actuator_horizon)
        self.check_current(revision)
        command = latest(self.algorithm.commands)
        if result is None or command is None or command.data is not result:
            raise ValueError('execution stage did not publish its computed command')
        return self.result(command=publication(command), computations=self.algorithm.sequence)
