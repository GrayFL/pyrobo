# Optional RSim navigation and devices

`pyrobo.rsim` contains runtime wrappers; pure `import pyrobo`, geometry, planning,
control and `pyrobo.simulation` do not import RSim. Use the real local RSim and
graphmap repositories. The optional `rsim` extra requires Python compatible with
RSim; pyrobo's pure library still supports Python >=3.11. No engine is needed in
the navigation consumer.

## Common chassis navigation

```python
from rsim import Runtime
from rsim.devices import ChassisPortsDevice
from pyrobo.rsim import (
    MemoryChassisFactory, NavigationInputs, NavigationCommandOutput,
    NavigationStack, StackConfig,
)

factory = MemoryChassisFactory(clock='world:demo', motion_enabled=True)
source = factory.build()                      # No running resources yet.
device = ChassisPortsDevice.from_component(source)
config = StackConfig(clock='world:demo')
# maps and missions are explicitly supplied Signals, using the same clock.
inputs = NavigationInputs(device, maps, missions, config)
stack = NavigationStack(inputs.states, inputs.maps, inputs.missions, config,
    local_options={'offload': True}, global_options={'offload': True})
output = NavigationCommandOutput(stack.commands, device, config)
async with Runtime(output.drive, stack.diagnostics):
    diagnostic = await stack.diagnostics.get(timeout=15)
```

`from_component` is a local connection view. A separate consumer uses
`ChassisPortsDevice(key=..., clock=...)` and never calls Factory.build/shared.
A provider process explicitly runs `factory.shared(key=...)`. Factory construction
and build do not start a Runtime; open creates the existing finite-force
`Simulator2D`, and close stops its commands. Each factory build creates a distinct
world. The full memory port specification lives only in
`pyrobo.rsim.drivers.memory.MemoryChassisFactory.port_specs`.

Both memory and MujocoSim implement `chassis-ports-v1`. Replacing the provider
requires no backend branch inside NavigationInputs, NavigationStack or
NavigationCommandOutput. Select matching robot geometry and motion limits in the
application. `MemoryChassisDevice` adds the memory provider's full odom/imu views;
those are connection-only capabilities.

The packaged consumer demonstrates the same graph against an existing provider:

```bash
python -m pyrobo.apps.navigation_provider --key demo --clock simulation \
  --execution thread --duration 20 --output navigation.json
```

`--execution local`, `thread` and `process` select direct algorithms, thread
offload or three ProcessPlacement workers. Process/shared modes require RSim DDS
and shared mapping dependencies. The default scene is a fixed prebuilt empty 6 m
room, a 0.6 m square footprint, start (0,0,0), and goal (0.8,0,0). Localization is
explicitly projected onto the plane; `--strict-planar-pose` rejects 3D poses
instead. Reports identify the localization's `pose_kind`; ground truth is not a
SLAM or localization accuracy result. `--map` supplies a different fixed map.

The example uses `command_speed_limit=.45` m/s and `command_yaw_limit=.75` rad/s
for both providers. These optional StackConfig limits restrict candidate command
references while preserving the model's larger measured-state envelope. They
participate in configuration fingerprints and process serialization. Physical
overshoot is measured unchanged; no input velocity is clipped to an operating
limit. A state outside the configured model envelope is an explicit error and
the consumer stops. Choose limits from the actual robot's operating envelope.

## Data and command boundaries

The public pose and state must have the same source stamp, clock, session,
episode and step index. The pose is a graphmap Pose in metres, with radians at
conversion boundaries. State must supply `linear_x`, `angular_z`, `acceleration`,
`yaw_acceleration`, `frame`, `episode`, `step_index` and `pose_kind`. Missing
measurements fail explicitly. Input conversion uses the existing StateCodec; it
never fabricates zero velocity or accesses an engine's live truth object.
When ports arrive separately, conversion selects the newest complete pair from
bounded histories and preserves that pair's original source age. Command output
checks both source session and episode, including restarts that reuse episode 0.

| Signal | Data |
| --- | --- |
| states | `{'state': plant.codec.pack(chassis), 'frame': 'world', 'episode': int}` |
| maps | `{'points': Nx2/Nx3, 'frame': 'world', 'bounds': [xmin,xmax,ymin,ymax], 'revision': int, 'static': bool}` |
| missions | `{'epoch': str, 'goal': [x,y,yaw], 'mode': 'auto' or 'paused'}` |
| commands | CommandEnvelope of native command, mode, task epoch and environment episode |

Maps are caller-owned. Change revision when geometry/frame/bounds change; dynamic
maps must match the state episode. Static maps may be reused across resets.
Dynamic map snapshots retain their source acquisition time, not a new timestamp
on each read. Goal/mission changes use a new task epoch. Only AUTO/PAUSED are
supported by this transport; manual authority and latched emergency control
remain explicit Supervisor/arbiter responsibilities.

The common velocity capability currently supports differential drive. Native
wheel commands convert to linear m/s and yaw rad/s, preserving controller ID,
controller epoch, sequence, the exact host deadline and any `SourceCommandBound`.
`coast` is rejected.
`soft_stop` and `estop` explicitly degrade to a zero-velocity request; this does
not claim equivalent physical braking force. Ackermann commands cannot silently
enter this adapter.

## Environment time and safety

`StackConfig.clock` defaults to `host:monotonic` for compatibility. Set the
provider's environment clock explicitly for simulation. Observation time,
proposal horizons and reset versions use source time; host freshness and command
safety use independent monotonic deadlines. Provider metadata carries
`host_monotonic_ns` across the same-host transport. Older sources use a
conservative conversion of receipt age. An input converter cannot renew a source
by repeatedly receiving or reading the same frame.

Local navigation processes each state once. Global replies and proposals are
associated with mission, map, source session and episode; stale results are not
adopted after reset. Execution keeps a short host lease and independently checks
source freshness. Pausing source time cannot keep a previous command alive.
The default host clock retains the existing executor behavior: each short
actuator lease is a new decision checked against current state. For environment
clocks, a motion proposal also needs enough host lifetime for the full actuator
interval. Otherwise execution requests a newly checked stop before expiry.
Forwarded motion never extends its proposal deadline, and the executor's sent
history records the actual emitted interval.
Memory reset stops commands, retires old controller epochs, isolates histories
without reusing sequence cursors, resets the physical time to zero, and publishes
a new episode. `source.reset()` is a local management operation, not remote RPC.

Cross-host monotonic deadlines are not supported by this example.
Python/asyncio/DDS are soft realtime.

## Explicit source scheduling

Clock domains and scheduling are separate. `StackConfig(scheduling='realtime')`
retains the existing host-frequency tasks. `scheduling='environment'` uses source
publications: input conversion consumes a complete pose/state pair, local
navigation uses `1/control_period` source Hz, execution uses `1/execution_period`
source Hz, and command/planning events are forwarded on publication. The same
`step_once` methods contain the algorithm work in both modes; no second DWA or
planner implementation is introduced.

`SourceProgress` records the exact consumed source interval, duplicates, initial
samples and skipped physical steps. Local control uses this actual interval,
including noninteger frequency ratios. The first sample in a new session/episode
uses the configured control period explicitly. Skipped physical steps include
intentional decimation, so they are not all overload drops. Published source
stamps remain integer nanoseconds throughout conversion and execution.

`RobotObservation.dt` at the independent executor remains the upcoming actuator
interval. It is not replaced by a measurement interval. In environment mode,
`source_rate` metadata conservatively shortens the host lease; the fixed
`SourceCommandBound` separately limits the checked physical endpoint. The final
provider validates the original session/episode/clock and the complete next
physics step before integration. Delivery delay, replay and controller reuse
cannot move that endpoint. Host TTL still expires while source time is stopped.

```python
source = MemoryChassisFactory(
    clock='world:demo', scheduling='accelerated', speedup=2.,
    motion_enabled=True,
).build()
config = StackConfig(clock='world:demo', scheduling='environment')
```

Memory `accelerated` mode uses a fixed physics dt and adjustable host cadence.
The default `realtime` source retains its old cadence. While a Runtime owns the
source, `await source.pause()` immediately retires commands and stops ticks;
`await source.resume()` preserves pose/time/step and increments episode. An
effective `await source.set_rate(rate)` changes episode while running; while
paused it takes effect at the next resume. Reset alone resets physical time and
step to zero. These are local management operations, serialized and completed
before cancellation is propagated. Episode identifies a control generation,
not exclusively a world-time reset.

The default `environment_policy='latest'` works locally and through process or
shared-port transport. Each task coalesces pending input and reports the actual
consumed dt. `environment_policy='every'` requires explicit `tick_source=source`
in both `NavigationInputs` and `NavigationStack`, and one local Runtime. Each
consumer binds an independent bounded TickSubscription in `open()`;
FrameTickPacer matches its tick to the exact completed upstream frame. A
`tick_capacity` queue backpressures the source; Signal history is not a queue or
backpressure mechanism. The source declares `tick_period_ns`, and history must
cover `(tick_capacity + 1) * ceil(consumer_period / source_period) + 1` source
samples. `environment_history` configures the converted-state history (default
128 in every mode). Cross-placement/shared-port every delivery and insufficient
history fail explicitly. Tick delivery is not a compute or physics ACK.

`scheduling='lockstep'` disables autonomous navigation/IO tasks and exposes
`NavigationStack.step_once(context, NavigationSnapshot(...))`. It computes one
navigation stage and returns a command envelope; command acceptance, transaction
retry rules and physics advancement must be coordinated separately. Calling this
method alone does not establish a complete lockstep simulation.

The consumer CLI accepts `--scheduling realtime|environment`. Reports include
the chosen mode, exact source contexts, measured RTF, actual control dt, compute
counts and command drops. `rtf` divides the first-to-last observed sample's source
delta by `observed_wall_s`; `elapsed_host_s` also includes Runtime startup/close.
`source_delta_per_total_host_s` divides that same source delta by the total time.
These denominators are distinct; a steady observation rate is not total job
throughput. Configured speedup is not a throughput guarantee.
The same safety rules may stop an overloaded graph; default DWA settings and
dynamics remain unchanged. Timing and workload parameters must be reported with
performance results, including unsuccessful configurations.

## Compatibility

The 0.1 transition retains `pyrobo.integrations.rsim.*` as re-exports of the same
classes. These aliases retire in 0.2. Stop and recreate providers when upgrading;
cross-version live recipe/pickle migration is not promised. The memory provider
uses pyrobo's finite-force model and explicit motion permission, so it is not a
numerically identical substitute for RSim's previous ideal/noisy SimulatedChassis.

Detailed development evidence and limitations live in the repository's local
PROJECT.md; reproducible tests are in `tests/test_rsim_*`. Navigation benchmarks
live in `benchmarks/navigation`, separate from the installed runtime package.
