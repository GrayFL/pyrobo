"""Packaged tools must work without the repository's examples namespace."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module', ['pyrobo.apps.pygame_sim', 'pyrobo.apps.web_sim',
                                   'pyrobo.apps.web_sim.replay', 'pyrobo.apps.navigation_provider'])
def test_packaged_cli_help_from_outside_repository(module, tmp_path):
    result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout


def test_web_and_config_resources_load_without_examples(tmp_path):
    pytest.importorskip('fastapi')
    program = '''
import importlib.abc, sys, tempfile
from importlib.resources import files
class DenyExamples(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'examples':
            raise ImportError('application depends on root examples: '+fullname)
sys.meta_path.insert(0, DenyExamples())
from fastapi.testclient import TestClient
from pyrobo.apps.web_sim.server import create_app
from pyrobo.apps.sim_common.config import load_config
assert load_config(files('pyrobo.apps').joinpath('configs/pygame.toml'))
assert files('pyrobo.apps').joinpath('scenes/narrow_ackermann_001.json').is_file()
with tempfile.TemporaryDirectory() as folder:
    with TestClient(create_app(scenes_dir=folder)) as client:
        assert client.get('/').status_code == 200
        assert client.get('/static/app.js').status_code == 200
        assert client.get('/static/style.css').status_code == 200
'''
    subprocess.run([sys.executable, '-c', program], cwd=tmp_path, check=True,
                   capture_output=True, text=True, timeout=20)
