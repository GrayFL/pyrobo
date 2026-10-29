"""Compatibility entry; use benchmarks.navigation.plot (removed in 0.2)."""
from benchmarks.navigation.plot import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
