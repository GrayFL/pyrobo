"""Compatibility entry; use benchmarks.navigation.planning (removed in 0.2)."""
from benchmarks.navigation.planning import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
