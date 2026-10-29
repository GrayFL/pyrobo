"""Compatibility entry; use pyrobo.apps.web_sim.__main__ (removed in 0.2)."""
from pyrobo.apps.web_sim.__main__ import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
