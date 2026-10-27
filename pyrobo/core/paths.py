"""A geometric route, without promises about timing or dynamic reachability."""
from dataclasses import dataclass

import numpy as np
from graphmap.pose import PoseArray


@dataclass(frozen=True)
class Path:
    poses: PoseArray

    def __post_init__(self):
        if not isinstance(self.poses, PoseArray) or len(self.poses.shape) != 1:
            raise ValueError('Path requires a one-dimensional graphmap PoseArray')
        if not self.poses.wrd_frame or not np.isfinite(self.poses.position).all():
            raise ValueError('path requires finite positions and an explicit world frame')
        if (not np.allclose(self.poses.scale, 1., atol=1e-9, rtol=0) or
                not np.allclose(self.poses.position[:, 2], 0., atol=1e-8, rtol=0) or
                not np.allclose(self.poses.rot_mat[..., :, 2], [0, 0, 1], atol=1e-8, rtol=0)):
            raise ValueError('path requires rigid planar poses')

    @property
    def frame(self):
        return self.poses.wrd_frame

    @property
    def xy(self):
        return self.poses.position[:, :2]

    def __len__(self):
        return len(self.poses.position)
