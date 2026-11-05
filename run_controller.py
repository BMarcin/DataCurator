"""Admin CLI for the distributed pipeline (wiki/Distributed.md).

    run_controller.py seed   <experiment> [--max-chunks N] [overrides...]
    run_controller.py status <job_id>
    run_controller.py rerun  <job_id> [--chunks 000042,000043] [--force]
    run_controller.py cancel <job_id>
    run_controller.py gc     <job_id>

Overrides after ``seed <experiment>`` are Hydra-style (``dataset.input=...``).
Connection settings come from ``.env`` (NATS_URL, JOB_S3_*/DC_S3_*).
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from DataCurator.distributed import controller  # noqa: E402  (env first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed", help="prepare and enqueue a job")
    p_seed.add_argument("experiment")
    p_seed.add_argument("--max-chunks", type=int, default=None, help="canary: seed only the first N chunks")
    p_seed.add_argument("--chunk-size", type=int, default=None, help="records per chunk (default 20)")
    p_seed.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. dataset.input=data/x.jsonl")

    p_status = sub.add_parser("status", help="per-stage done/missing under current sigs")
    p_status.add_argument("job_id")

    p_rerun = sub.add_parser("rerun", help="enqueue only what is missing under the plan's sigs")
    p_rerun.add_argument("job_id")
    p_rerun.add_argument("--chunks", default=None, help="comma-separated chunk ids, e.g. 000042,000043")
    p_rerun.add_argument("--force", action="store_true", help="delete the named chunks' outputs and recompute them")

    p_cancel = sub.add_parser("cancel", help="purge the work queue and mark the job cancelled")
    p_cancel.add_argument("job_id")

    p_gc = sub.add_parser("gc", help="delete dead-sig prefixes")
    p_gc.add_argument("job_id")

    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if args.command == "seed":
        kwargs = {"max_chunks": args.max_chunks, "overrides": args.overrides}
        if args.chunk_size is not None:
            kwargs["chunk_size"] = args.chunk_size
        asyncio.run(controller.seed(args.experiment, **kwargs))
    elif args.command == "status":
        asyncio.run(controller.status(args.job_id))
    elif args.command == "rerun":
        chunks = args.chunks.split(",") if args.chunks else None
        asyncio.run(controller.rerun(args.job_id, chunks=chunks, force=args.force))
    elif args.command == "cancel":
        asyncio.run(controller.cancel(args.job_id))
    elif args.command == "gc":
        asyncio.run(controller.gc(args.job_id))


if __name__ == "__main__":
    main()
