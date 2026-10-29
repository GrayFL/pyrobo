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
controller epoch, sequence and the exact host deadline. `coast` is rejected.
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
Memory reset stops commands, retires old controller epochs, isolates histories
without reusing sequence cursors, resets the physical time to zero, and publishes
a new episode. `source.reset()` is a local management operation, not remote RPC.

Default hz scheduling is unchanged. `speedup` changes memory source progression;
source-driven/lockstep navigation requires a separately configured scheduler and
is not proved by the realtime example. Cross-host monotonic deadlines are not
supported by this example. Python/asyncio/DDS are soft realtime.

## Compatibility

The 0.1 transition retains `pyrobo.integrations.rsim.*` as re-exports of the same
classes. These aliases retire in 0.2. Stop and recreate providers when upgrading;
cross-version live recipe/pickle migration is not promised. The memory provider
uses pyrobo's finite-force model and explicit motion permission, so it is not a
numerically identical substitute for RSim's previous ideal/noisy SimulatedChassis.

Detailed development evidence and limitations live in the repository's local
PROJECT.md; reproducible tests are in `tests/test_rsim_*`. Navigation benchmarks
live in `benchmarks/navigation`, separate from the installed runtime package.
