"""Monitor entry point (wiki/Distributed.md): done-event handoff, reconcile,
alerts and metrics. One instance; restart-safe (state rebuilds from the done
stream, the job records and S3).
"""
from __future__ import annotations

import asyncio
import signal
import sys
from types import FrameType
from typing import Optional

from loguru import logger

from DataCurator.distributed.monitor import monitor_from_env


def _exit_on_sigterm(signum: int, frame: Optional[FrameType]) -> None:
    """Exit cleanly: as a container's PID 1, Python would ignore SIGTERM by default."""
    raise SystemExit(0)


def main() -> None:
    """Run the monitor until SIGTERM or SIGINT."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    asyncio.run(monitor_from_env().run())


if __name__ == "__main__":
    main()
