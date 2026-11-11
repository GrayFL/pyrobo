"""Import-time registration without importing algorithms, frontends or plugins.

Decorate an implementation in its own module. Importing that module makes it
available to every consumer. Discovery/import policy belongs to applications.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ComponentSpec:
    config: type
    factory: type
    label: str
    family: str = ''
    defaults: tuple = ()
    teleop_config: bool = False


LOCAL_OPTIMIZERS: dict[str, ComponentSpec] = {}
GLOBAL_PLANNERS: dict[str, ComponentSpec] = {}
KINEMATIC_MODELS: dict[str, ComponentSpec] = {}
VEHICLE_TEMPLATES: dict[str, ComponentSpec] = {}


def _register(registry, name, *, config=None, label=None, family='', defaults=(), teleop_config=False):
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise ValueError('component name must be a lowercase identifier')

    def decorate(factory):
        spec = ComponentSpec(config or factory, factory, label or name, family, tuple(defaults), teleop_config)
        if name in registry and registry[name] != spec:
            raise ValueError(f'duplicate component registration: {name}')
        registry[name] = spec
        return factory
    return decorate


def register_local_optimizer(name, **metadata):
    return _register(LOCAL_OPTIMIZERS, name, **metadata)


def register_global_planner(name, **metadata):
    return _register(GLOBAL_PLANNERS, name, **metadata)


def register_kinematic_model(name, **metadata):
    return _register(KINEMATIC_MODELS, name, **metadata)


def register_vehicle_template(name, **metadata):
    return _register(VEHICLE_TEMPLATES, name, **metadata)


def spec_for_config(registry, config, default):
    if config is None:
        return registry[default]
    # Exact types first, so independently registered subclasses are not swallowed.
    for spec in registry.values():
        if type(config) is spec.config:
            return spec
    for spec in registry.values():
        if isinstance(config, spec.config):
            return spec
    raise TypeError(f'unregistered configuration: {type(config).__name__}')
