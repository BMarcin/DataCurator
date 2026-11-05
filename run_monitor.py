"""Monitor entry point (wiki/Distributed.md): done-event handoff, reconcile,
alerts and metrics. One instance; restart-safe (state rebuilds from the done
stream, the job records and S3).
"""
from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv
from loguru import logger

# load_dotenv()


from DataCurator.distributed.monitor import monitor_from_env  # noqa: E402  (env first)


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(monitor_from_env().run())


if __name__ == "__main__":
    main()
