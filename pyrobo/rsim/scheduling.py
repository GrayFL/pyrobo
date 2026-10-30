"""Source snapshots and progress accounting, independent of task wakeups."""
from dataclasses import dataclass
import math

from rsim import Component, FrameTickPacer, SignalPacer, StepContext

from .packets import episode, latest, source_session


def source_context(frame):
    metadata = frame.metadata
    step = metadata.get('step_index', frame.sequence)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError('source step_index must be a nonnegative integer')
    generation = episode(frame)
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError('source episode must be a nonnegative integer')
    return StepContext(source_session(frame), generation, step)


def source_rate(frame):
    """Configured pacing rate used to shorten host leases conservatively.

    Soft real-time scheduling is not a strict bound on each adjacent wall
    interval. SourceCommandBound separately limits the proved physical end.
    """
    rate = frame.metadata.get('source_rate', 1.)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('source_rate must be finite and positive')
    return float(rate)


def source_pacer(owner, signal, config, *, hz=None, tick_source=None):
    """Bind an independent local subscription, or latest-only source events.

    The bounded every queue is owned by TickSource, never Signal history.
    History must retain all source samples spanned by a consumer's queue.
    """
    if config.environment_policy == 'latest':
        return SignalPacer(signal, hz=hz, deduplicate=True)
    if not isinstance(tick_source, Component) or not callable(getattr(tick_source, 'subscribe', None)):
        raise ValueError('every scheduling requires an explicitly injected local tick_source Component')
    if (owner._runtime is None or tick_source._runtime is not owner._runtime
            or tick_source._binding is not None or tick_source._canonical is not None
            or signal._resolved() is not signal or signal.producer._runtime is not owner._runtime):
        raise ValueError('every scheduling requires one local Runtime without placement or shared-port proxies')
    period_ns = getattr(tick_source, 'tick_period_ns', None)
    if not isinstance(period_ns, int) or isinstance(period_ns, bool) or period_ns <= 0:
        raise ValueError('every tick_source must declare its fixed tick_period_ns')
    stride = 1 if hz is None else max(1, math.ceil(1e9/(hz*period_ns)))
    required = (config.tick_capacity+1)*stride+1
    if signal.history_size < required:
        raise ValueError(f'every consumer requires history >= {required}, got {signal.history_size}')
    subscription = tick_source.subscribe(hz=hz, policy='every', capacity=config.tick_capacity)
    return FrameTickPacer(signal, subscription)


@dataclass(frozen=True)
class NavigationSnapshot:
    state: object
    map: object
    mission: object

    @property
    def context(self):
        return source_context(self.state)

    @property
    def input_revision(self):
        return (self.mission.data['epoch'], self.map.data['revision'],
                source_session(self.map), episode(self.map))

    @property
    def version(self):
        context = self.context
        return (*self.input_revision, context.session, context.episode)

    def require_context(self, context):
        if context is not None and context != self.context:
            raise ValueError('navigation snapshot does not match the requested source context')


def snapshot_from(states, maps, missions, *, state=None):
    state = latest(states) if state is None else state
    map_frame, mission_frame = latest(maps), latest(missions)
    if any(frame is None for frame in (state, map_frame, mission_frame)):
        return None
    return NavigationSnapshot(state, map_frame, mission_frame)


class SourceProgress:
    """Account for consumed samples; duplicates never become a new control dt.

    dt is the actual source interval between consumed samples. The first sample
    of each generation uses an explicitly supplied initial period. Skipped
    counts include deliberate source-frequency decimation, not only overload.
    """
    def __init__(self, initial_dt):
        if not math.isfinite(initial_dt) or initial_dt <= 0:
            raise ValueError('initial source period must be finite and positive')
        self.initial_dt = initial_dt
        self.context = None
        self.stamp_ns = None
        self.dt = None
        self.previous_stamp_ns = None
        self.initial = True
        self.received = self.duplicates = self.skipped = self.resets = 0
        self._retired_sessions = set()

    def consume(self, frame):
        context, stamp = source_context(frame), frame.stamp_ns
        previous = self.context
        same = previous is not None and (context.session, context.episode) == (previous.session, previous.episode)
        if same:
            if context.step_index < previous.step_index or stamp < self.stamp_ns:
                raise ValueError('source time/step moved backwards within one generation')
            if context.step_index == previous.step_index or stamp == self.stamp_ns:
                if context.step_index != previous.step_index or stamp != self.stamp_ns:
                    raise ValueError('source stamp and step must advance together')
                self.duplicates += 1
                return None
            dt = (stamp-self.stamp_ns)/1e9
            self.skipped += context.step_index-previous.step_index-1
        else:
            if context.session in self._retired_sessions or (previous is not None
                    and context.session == previous.session and context.episode < previous.episode):
                raise ValueError('source sample belongs to a retired generation')
            if previous is not None:
                self.resets += 1
                if context.session != previous.session:
                    self._retired_sessions.add(previous.session)
            dt = self.initial_dt
        self.previous_stamp_ns = self.stamp_ns if same else None
        self.context, self.stamp_ns, self.dt = context, stamp, dt
        self.initial = not same
        self.received += 1
        return dt

    def diagnostics(self):
        return {'source_dt_s': self.dt, 'previous_source_stamp_ns': self.previous_stamp_ns,
                'initial_dt': self.initial,
                'source_steps_skipped': self.skipped, 'source_samples': self.received,
                'source_duplicates': self.duplicates, 'source_resets': self.resets}
