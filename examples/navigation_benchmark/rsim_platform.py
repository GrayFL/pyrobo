"""Compatibility entry; use benchmarks.navigation.rsim_platform (removed in 0.2)."""
from benchmarks.navigation.rsim_platform import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
