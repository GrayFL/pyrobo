"""Compatibility entry; use benchmarks.navigation.control (removed in 0.2)."""
from benchmarks.navigation.control import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
