"""Shared result and polyline output for continuous-space planners."""
from dataclasses import dataclass, field

import numpy as np
from graphmap.pose import PoseArray


@dataclass(frozen=True)
class PlanResult:
    path: PoseArray
    status: str
    extensions: int
    elapsed: float
    diagnostics: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "success"


def densify_path(xy: np.ndarray, spacing: float) -> np.ndarray:
    """Retain every corner; interpolating across corners can cut through obstacles."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    parts = []
    for a, b in zip(xy[:-1], xy[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        parts.append(np.linspace(a, b, count, endpoint=False))
    return np.concatenate([*parts, xy[-1:]])
