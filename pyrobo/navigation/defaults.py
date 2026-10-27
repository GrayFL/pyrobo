"""Compatibility composition root; concrete algorithms are selected only here."""
from .config import NavigationConfig


def components(*, footprint, planner, controller, model, dynamics, global_planner, local_optimizer,
               command_space, settings, index, max_cloud_age):
    from ..control import DynamicWindow
    from ..safety import SafetyConfig
    from pyrobo.models.dynamics.chassis import ChassisDynamics
    from pyrobo.models.kinematics.differential import DifferentialDrive
    from ..models.kinematics import command_space_for
    from ..planning import RRTConnect, RRTConnectConfig, VoronoiConfig, VoronoiPlanner
    from pyrobo.core.footprint import Footprint

    footprint = footprint or getattr(local_optimizer, 'footprint', None) or Footprint()
    model = model or getattr(local_optimizer, 'model', None) or getattr(dynamics, 'model', None)
    if model is None:
        model = DifferentialDrive(**{key: getattr(controller, key) for key in ('max_speed', 'max_yaw_rate')
                                     if hasattr(controller, key)})
    if (getattr(local_optimizer, 'model', model) != model or
            getattr(local_optimizer, 'footprint', footprint) != footprint):
        raise ValueError('injected optimizer model and footprint must match navigation')
    space = command_space or getattr(local_optimizer, 'command_space', None) or command_space_for(model)
    if local_optimizer is None:
        local_optimizer = DynamicWindow(controller, footprint, model=model, dynamics=dynamics,
                                        command_space=space, index=index,
                                        input_limits=SafetyConfig(max_cloud_age=max_cloud_age))
    plant = (dynamics if hasattr(dynamics, 'integrate') else
             getattr(local_optimizer, 'plant', None) or ChassisDynamics(model, dynamics))
    cfg = controller or getattr(local_optimizer, 'config', None)
    settings = settings or NavigationConfig(**{key: getattr(cfg, key) for key in
                        ('lookahead', 'goal_tolerance', 'yaw_tolerance') if hasattr(cfg, key)})
    if global_planner is None:
        if planner is None and not space.can_spin:
            # Clearance allowance for the default car's finite turning radius.
            radius = space.goal_approach_distance(0.) / 1.5
            planner = RRTConnectConfig(clearance=.12 + .4*radius, time_budget=.2)
        if isinstance(planner, VoronoiConfig):
            global_planner = VoronoiPlanner(planner, footprint)
        elif planner is None or isinstance(planner, RRTConnectConfig):
            global_planner = RRTConnect(planner, footprint)
        else:
            raise TypeError('planner must be RRTConnectConfig or VoronoiConfig; use global_planner for an implementation')
    return footprint, model, space, plant, global_planner, local_optimizer, settings
