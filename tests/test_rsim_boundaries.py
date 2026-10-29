"""Optional runtime boundaries and transitional class identity."""
import subprocess
import sys

import pytest


def test_pure_modules_do_not_import_runtime_or_engines(tmp_path):
    program = '''
import importlib.abc, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'rsim', 'mujoco', 'mujocosim', 'pygame', 'fastapi'}:
            raise ImportError('forbidden runtime import: '+fullname)
sys.meta_path.insert(0, Deny())
from pyrobo import ChassisDynamics, DifferentialDrive, ChassisState, pose2d
from pyrobo.simulation import Simulator2D
model = DifferentialDrive()
result = ChassisDynamics(model).step(ChassisState(pose2d(0, 0)), model.command(.2), .02)
assert result.state.timestamp == .02
'''
    subprocess.run([sys.executable, '-c', program], cwd=tmp_path, check=True, capture_output=True, text=True)


def test_compatibility_paths_reexport_identical_classes():
    pytest.importorskip('rsim')
    from pyrobo.rsim import NavigationStack, StackConfig, LocalNavigation
    from pyrobo.integrations.rsim import NavigationStack as OldStack, StackConfig as OldConfig
    from pyrobo.integrations.rsim.components import LocalNavigation as OldLocal
    from pyrobo.integrations.rsim.packets import ObservationReader as OldReader
    from pyrobo.rsim.packets import ObservationReader
    assert (OldStack, OldConfig, OldLocal, OldReader) == (NavigationStack, StackConfig, LocalNavigation, ObservationReader)
