"""Distributed worker entry point (wiki/Distributed.md).

One long-lived process per VM. Reads connection settings and endpoint
bindings from the environment (``.env`` + ``DC_BINDINGS_FILE``); behaviour
arrives in the work messages. Drain with SIGTERM or an ``s`` line on stdin.
"""
from __future__ import annotations

import asyncio
import sys

from dotenv import load_dotenv
from loguru import logger

# load_dotenv()

from DataCurator.distributed.observability import setup_worker_logging  # noqa: E402  (env first)
from DataCurator.distributed.worker import worker_from_env  # noqa: E402


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    worker = worker_from_env()
    setup_worker_logging(worker.worker_id)
    asyncio.run(worker.run())


if __name__ == "__main__":
    main()
