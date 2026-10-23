"""Headless smoke demo: python -m pyrobo.simulation."""
import json
import argparse

from .. import Navigator, RRTConnectConfig, VoronoiConfig
from . import demo_world, run_navigation


def main():
    parser = argparse.ArgumentParser(description="Point-cloud global planning and DWA closed-loop demo")
    parser.add_argument("--planner", choices=("rrt_connect", "voronoi"), default="rrt_connect")
    args = parser.parse_args()
    config = VoronoiConfig() if args.planner == "voronoi" else RRTConnectConfig()
    world, start, goal = demo_world()
    run = run_navigation(world, start, goal, navigator=Navigator(planner=config))
    print(json.dumps({"planner": args.planner, **run.summary()}, indent=2))
    if not run.arrived or run.collided:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

