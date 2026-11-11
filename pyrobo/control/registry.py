"""Compatibility exports for the import-time component registry."""
from pyrobo.registry import (LOCAL_OPTIMIZERS, ComponentSpec as OptimizerSpec,
                             register_local_optimizer, spec_for_config)


def optimizer_spec(config):
    return spec_for_config(LOCAL_OPTIMIZERS, config, 'dwa')
