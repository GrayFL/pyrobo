"""Headless smoke demo: python -m pyrobo.simulation."""
import json

from . import demo_world, run_navigation


def main():
    world, start, goal = demo_world()
    run = run_navigation(world, start, goal)
    print(json.dumps(run.summary(), indent=2))
    if not run.arrived or run.collided:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

