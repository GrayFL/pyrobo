"""Compatibility entry; use pyrobo.apps.web_sim.replay (removed in 0.2)."""
from pyrobo.apps.web_sim.replay import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
