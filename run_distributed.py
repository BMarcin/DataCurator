"""Distributed worker entry point (wiki/Distributed.md).

One long-lived process per VM. Reads connection settings from the process
environment and endpoint bindings from ``DC_BINDINGS_FILE``; behaviour
arrives in the work messages. Drain with SIGTERM or an ``s`` line on stdin.
"""
from __future__ import annotations

import asyncio
import sys

from loguru import logger

from DataCurator.distributed.observability import setup_worker_logging
from DataCurator.distributed.worker import worker_from_env


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    worker = worker_from_env()
    setup_worker_logging(worker.worker_id)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
