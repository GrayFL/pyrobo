"""Application discovery policy and JSON metadata for dynamic selectors."""
from dataclasses import asdict
from importlib import import_module
import os
from pkgutil import iter_modules
from pyrobo.registry import GLOBAL_PLANNERS, LOCAL_OPTIMIZERS, KINEMATIC_MODELS, VEHICLE_TEMPLATES
from . import vehicles  # register bundled templates


def discover():
    # New packages/modules in these namespaces need no central list edits.
    for namespace in ('pyrobo.planning', 'pyrobo.control', 'pyrobo.models.kinematics'):
        package = import_module(namespace)
        for module in sorted(iter_modules(package.__path__), key=lambda m: m.name):
            if not module.name.startswith('_'):
                import_module(f'{namespace}.{module.name}')
    # Only server configuration may import external code, never a client request.
    for module in os.environ.get('PYROBO_PLUGINS', '').split(','):
        if module.strip():
            import_module(module.strip())


def catalog():
    from .config import default_config, controller_parameters, build_parameters
    result = {}
    for group, registry in (('planners', GLOBAL_PLANNERS), ('optimizers', LOCAL_OPTIMIZERS),
                            ('models', KINEMATIC_MODELS), ('vehicles', VEHICLE_TEMPLATES)):
        result[group] = [dict(id=name, label=s.label, family=s.family, defaults=asdict(s.config()))
                         for name, s in registry.items()]
    for model in result['models']:
        config = default_config()
        config['backend']['model'] = model['id']
        model['planner_defaults'] = {}
        for name in GLOBAL_PLANNERS:
            config['backend']['planner'] = name
            planner = build_parameters(config)[4]
            model['planner_defaults'][name] = {k: getattr(planner, k) for k in ('clearance',) if hasattr(planner, k)}
        model['controller_defaults'] = {name: {k: v for k, v in asdict(controller_parameters(config, name)).items()
                                              if k in ('prediction_time', 'lookahead', 'track_lookahead_m')}
                                        for name in LOCAL_OPTIMIZERS}
    return result
