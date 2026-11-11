"""Scoped regression runner. Plain pytest and `full` always retain all tests."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
QUICK = [
    'tests/test_discovery_library.py', 'tests/test_geometry.py',
    'tests/test_web_sim.py::test_http_case_storage_and_validation',
    'tests/test_pygame_session.py::test_config_precedence_nested_drag_and_inactive_model_tables',
    'tests/test_public_contracts.py',
]
AREAS = {
    'web-ui': ['tests/test_web_sim.py::test_http_case_storage_and_validation',
               'tests/test_packaged_apps.py::test_web_and_config_resources_load_without_examples'],
    'web': ['tests/test_web_sim.py', 'tests/test_discovery_library.py', 'tests/test_packaged_apps.py'],
    'registry': ['tests/test_discovery_library.py', 'tests/test_public_contracts.py', 'tests/test_pygame_session.py',
                 'tests/test_reference_optimizers.py'],
    'pygame': ['tests/test_pygame_session.py', 'tests/test_pygame_ui.py', 'tests/test_packaged_apps.py'],
    'geometry': ['tests/test_geometry.py', 'tests/test_planning_control.py', 'tests/test_voronoi.py',
                 'tests/test_adaptive_planning.py', 'tests/test_model_navigation.py', 'tests/test_web_sim.py'],
    'planning': ['tests/test_planning_control.py', 'tests/test_voronoi.py', 'tests/test_adaptive_planning.py',
                 'tests/test_logical_planning.py', 'tests/test_navigation.py', 'tests/test_model_navigation.py'],
    'navigation': ['tests/test_navigation*.py', 'tests/test_model_navigation.py', 'tests/test_path_tracking.py',
                   'tests/test_runtime.py', 'tests/test_pygame_session.py'],
    'control': ['tests/test_dwa*.py', 'tests/test_teb.py', 'tests/test_eb*.py', 'tests/test_reference_optimizers.py',
                'tests/test_planning_control.py', 'tests/test_model_navigation.py', 'tests/test_command_pipeline.py'],
    'dynamics': ['tests/test_dynamics*.py', 'tests/test_extensible_batches.py', 'tests/test_command_pipeline.py',
                 'tests/test_model_navigation.py', 'tests/test_teb.py', 'tests/test_reference_optimizers.py'],
    'runtime': ['tests/test_runtime.py', 'tests/test_command*.py', 'tests/test_rsim_execution_leases.py',
                'tests/test_rsim_navigation*.py'],
    'rsim': ['tests/test_rsim*.py', 'tests/test_command_leases.py'],
}
PREFIXES = [
    ('pyrobo/apps/web_sim/', {'web'}), ('pyrobo/apps/scenes/', {'web'}),
    ('pyrobo/apps/sim_common/', {'web', 'pygame', 'registry'}),
    ('pyrobo/apps/pygame_sim/', {'pygame'}), ('pyrobo/apps/configs/', {'pygame'}),
    ('pyrobo/registry.py', {'registry'}), ('pyrobo/control/registry.py', {'registry'}),
    ('pyrobo/geometry/', {'geometry'}), ('pyrobo/planning/', {'planning'}),
    ('pyrobo/navigation/', {'navigation', 'registry'}), ('pyrobo/control/', {'control'}),
    ('pyrobo/models/', {'dynamics', 'control'}), ('pyrobo/runtime/', {'runtime'}),
    ('pyrobo/safety/', {'runtime', 'control'}), ('pyrobo/teleop/', {'runtime'}),
    ('pyrobo/rsim/', {'rsim'}), ('pyrobo/integrations/rsim/', {'rsim'}),
]


def changed_paths(base=None):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT).decode().split('\0')
    # Include staged, unstaged and untracked files. Optional base also includes commits.
    return sorted(set(filter(None, [*git('diff', '--name-only', '-z', base or 'HEAD'),
                                    *git('ls-files', '--others', '--exclude-standard', '-z')])))


def related(paths):
    areas, direct = set(), set()
    for path in paths:
        if path.endswith('.md') or path.startswith(('assets/', 'docs/', 'reports/')):
            continue
        if path.startswith('tests/test_') and path.endswith('.py'):
            direct.add(path); continue
        if path.startswith('tests/browser_'):
            print(f'浏览器专项需另行运行：{path}'); continue
        if path.startswith('pyrobo/apps/web_sim/static/'):
            areas.add('web-ui'); continue
        for prefix, mapped in PREFIXES:
            if path.startswith(prefix):
                areas.update(mapped); break
        else:
            # Unknown/core/package-wide changes are deliberately conservative.
            print(f'无法确定影响范围，选择完整测试：{path}')
            return ['tests']
    return expand(areas) + sorted(p for p in direct if (ROOT/p).is_file())


def expand(areas):
    return sorted({str(p.relative_to(ROOT)) + ('::'+node if node else '')
                   for area in areas for pattern in AREAS[area]
                   for path, _, node in [pattern.partition('::')] for p in ROOT.glob(path)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tier', nargs='?', choices=('quick', 'related', 'full'), default='quick')
    parser.add_argument('--area', action='append', choices=tuple(AREAS), help='explicit scope for this change; repeatable')
    parser.add_argument('--base', help='git base for automatic related scope (default: HEAD + working tree)')
    parser.add_argument('--list', action='store_true', help='print selection without running pytest')
    args, pytest_args = parser.parse_known_args()
    if args.tier != 'related' and (args.area or args.base):
        parser.error('--area/--base only apply to related')
    selected = QUICK if args.tier == 'quick' else ['tests'] if args.tier == 'full' else (
        expand(args.area) if args.area else related(changed_paths(args.base)))
    selected = list(dict.fromkeys(selected))
    if not selected:
        print('仅文档或无需 pytest 的变更；未启动测试。')
        return 0
    print(f'测试档位 {args.tier}：\n'+'\n'.join(selected), flush=True)
    if args.list:
        return 0
    return subprocess.call([sys.executable, '-m', 'pytest', '-q', *selected, *pytest_args], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
