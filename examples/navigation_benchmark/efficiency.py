"""Compatibility entry; use benchmarks.navigation.efficiency (removed in 0.2)."""
from benchmarks.navigation.efficiency import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
