"""Geometry presets, independent of navigation algorithms and chassis models."""
from dataclasses import dataclass
from pyrobo.registry import register_vehicle_template


@register_vehicle_template('compact', label='默认小车 · 宽 0.50 × 长 0.50 m')
@dataclass(frozen=True)
class CompactVehicle:
    length: float = .5
    width: float = .5
    margin: float = .08
    offset_x: float = 0.


@register_vehicle_template('towing', label='牵引车 · 宽 0.54 × 长 0.79 m')
@dataclass(frozen=True)
class TowingVehicle:
    length: float = .79
    width: float = .54
    margin: float = .08
    offset_x: float = -.295


@register_vehicle_template('long', label='长车体 · 宽 0.50 × 长 1.00 m')
@dataclass(frozen=True)
class LongVehicle:
    length: float = 1.
    width: float = .5
    margin: float = .08
    offset_x: float = 0.
