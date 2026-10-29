"""Render saved benchmark evidence without running planners or controllers."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from pyrobo.core.geometry import planar_state
from .scenarios import scenarios, from_scene


def draw_map(ax, case):
    ax.add_collection(LineCollection(case.world.segments(), color='#74887b', linewidth=1.2))
    b = case.world.bounds
    ax.set(xlim=(b.xmin, b.xmax), ylim=(b.ymin, b.ymax), aspect='equal', xlabel='x / m', ylabel='y / m')
    for pose, color, label in [(case.start, '#247ea0', 'start'), (case.goal, '#b56f19', 'goal')]:
        x, y, yaw = planar_state(pose)
        ax.plot(x, y, 'o', color=color, label=label, markersize=4)
        ax.arrow(x, y, .6*np.cos(yaw), .6*np.sin(yaw), width=.025, color=color, length_includes_head=True)
    ax.grid(alpha=.15)


def render(report, output):
    cases = scenarios()
    if report['kind'] == 'global_planning':
        names = list(dict.fromkeys(row['scenario'] for row in report['runs']))
        fig, axes = plt.subplots(int(np.ceil(len(names)/4)), 4, figsize=(16, 4*np.ceil(len(names)/4)), squeeze=False)
        for ax, name in zip(axes.flat, names):
            draw_map(ax, cases[name])
            title = [name]
            for algorithm, color in [('rrt_connect', '#247ea0'), ('voronoi', '#bc671f')]:
                rows = [r for r in report['runs'] if r['scenario'] == name and r['algorithm'] == algorithm]
                if not rows:
                    continue
                solved = [r for r in rows if r['found'] and r['solid_path_valid']]
                if solved:
                    p = np.array(solved[0]['path_xy'])
                    ax.plot(p[:, 0], p[:, 1], color=color, lw=1.5, label=algorithm)
                title.append(f'{algorithm}: {len(solved)}/{len(rows)}')
            ax.set_title('\n'.join(title), fontsize=9)
            ax.legend(fontsize=6, loc='upper right')
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        fig.suptitle('Global planning — equal budgets, fixed maps, independent solid checks')
    elif report['kind'] == 'synchronous_closed_loop':
        names = list(dict.fromkeys(row['scenario'] for row in report['runs']))
        fig, axes = plt.subplots(int(np.ceil(len(names)/4)), 4,
                                 figsize=(16, 4*np.ceil(len(names)/4)), squeeze=False)
        for ax, name in zip(axes.flat, names):
            draw_map(ax, cases[name])
            rows = [r for r in report['runs'] if r['scenario'] == name]
            for row in rows:
                p = np.asarray(row['poses'])
                color = '#247ea0' if row['algorithm'] == 'rrt_connect' else '#bc671f'
                style = '-' if row['model'] == 'ackermann' else '--'
                label = f"{row['algorithm']} / {row['model']}"
                ax.plot(p[:, 0], p[:, 1], style, color=color, lw=1.3, label=label)
                if not row['arrived']:
                    ax.plot(*p[-1, :2], 'x', color='#c52c2c', markersize=9)
            ax.set_title(f"{name}: {sum(r['arrived'] for r in rows)}/{len(rows)} arrived", fontsize=10)
            ax.legend(fontsize=5, loc='upper right')
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        fig.suptitle('Synchronous closed loop — solid collision checks; red cross = unfinished')
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        ax = axes[0, 0]
        case = from_scene(report['scene']) if report.get('scene') else cases[report['scenario']]
        draw_map(ax, case)
        state = np.array([s['state'] for s in report['samples']])
        times = np.array([s['t'] for s in report['samples']])
        if report['plans']:
            p = np.array(report['plans'][0]['path_xy'])
            if len(p):
                ax.plot(p[:, 0], p[:, 1], '--', color='#bc671f', label='global route')
        ax.plot(state[:, 0], state[:, 1], color='#247ea0', label='actual vehicle')
        ax.legend(fontsize=8)
        ax.set_title(f"{report['scenario']} | arrived={report['arrived']} | collision={report['collided']}")
        ax = axes[0, 1]
        ax.plot(times, state[:, 3], label='speed', color='#247ea0')
        ax.axhline(report['config']['model']['max_speed'], ls='--', color='#bc671f', label='model speed limit')
        ax.set(xlabel='wall-clock simulation time / s', ylabel='m/s', title='Actual speed')
        ax.legend()
        ax = axes[1, 0]
        ax.plot(times, np.rad2deg(state[:, 5]), label='steering / deg')
        ax.plot(times, state[:, 4], label='yaw rate / rad/s', alpha=.7)
        ax.set(xlabel='wall-clock simulation time / s', title='Actual steering and rotation')
        ax.legend()
        ax = axes[1, 1]
        for key, label in [('controls', 'control'), ('executions', 'execution')]:
            values = np.sort([r['compute_ms'] for r in report[key]])
            if len(values):
                ax.plot(values, np.arange(1, len(values)+1)/len(values), label=label)
        ax.axvline(100, ls='--', color='gray', label='100 ms')
        ax.set(xlabel='computation time / ms', ylabel='fraction <= x', title='Empirical latency CDF')
        ax.legend()
        fig.suptitle('RSim navigation — physics advances independently of planning')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.report.read_text()), args.output)


if __name__ == '__main__':
    main()
