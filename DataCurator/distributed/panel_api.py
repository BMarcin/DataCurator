"""The monitor's read-only HTTP surface (wiki/Distributed.md "Monitor HTTP API").

Clients read the monitor directly, exactly as Prometheus already does — no
job metadata re-enters S3, no manifest, no artifacts_rev. The
artifact view is an allowlist: only live-sig keys the plan addresses are ever
listed, so a re-seeded job's artifacts flip atomically to the new plan.

Auth: a static bearer token (``DC_PANEL_TOKEN``). ``/metrics`` stays unauthenticated for Prometheus.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict

from prometheus_client import make_asgi_app
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Mount, Route

from . import wire


def _authorized(request) -> bool:
    token = os.environ.get("DC_PANEL_TOKEN")
    if not token:
        return False  # no token configured -> the API is closed, not open
    return request.headers.get("authorization") == f"Bearer {token}"


def _job_summary(state) -> Dict[str, Any]:
    record = state.record
    last = len(record.stages) - 1
    return {
        "job_id": record.job_id,
        "state": record.state,
        "version": record.version,
        "chunk_count": record.chunk_count,
        "stages": [
            {"index": s.index, "stage_id": s.stage_id, "sig": s.sig, "requires": s.requires,
             "done": len(state.done.get(s.index, ()))}
            for s in record.stages
        ],
        "complete_chunks": len(state.done.get(last, ())),
        "dead_lettered": sorted(state.dead_lettered),
    }


def build_app(monitor) -> Starlette:
    def guard(handler):
        async def wrapped(request):
            if not _authorized(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return await handler(request)

        return wrapped

    async def jobs(request):
        return JSONResponse([_job_summary(s) for s in monitor.jobs.values()])

    async def job_detail(request):
        state = monitor.jobs.get(request.path_params["job_id"])
        if state is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        view = _job_summary(state)
        view["queues"] = state.queues
        view["flagged"] = state.flagged
        view["chunks_done"] = {
            s.stage_id: sorted(state.done.get(s.index, ())) for s in state.record.stages
        }
        view["fleet"] = [
            {
                "worker_id": worker_id,
                "capabilities": hb.get("capabilities", []),
                "in_flight": hb.get("in_flight", 0),
                "version": hb.get("version"),
                "metrics": hb.get("metrics", {}),
            }
            for worker_id, hb in monitor._alive_workers().items()
        ]
        return JSONResponse(view)

    async def artifacts(request):
        """Live-sig result keys, straight from the plan — the allowlist of
        downloadable results."""
        state = monitor.jobs.get(request.path_params["job_id"])
        if state is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        record = state.record
        last = record.stages[-1]
        keys = [
            wire.stage_key(monitor.prefix, record.job_id, last.index, last.stage_id, last.sig, chunk_id)
            for chunk_id in sorted(state.done.get(last.index, ()))
        ]
        return JSONResponse({"job_id": record.job_id, "stage_id": last.stage_id, "keys": keys})

    async def stream(request):
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        monitor.listeners.add(queue)

        async def gen():
            try:
                yield b"retry: 5000\n\n"
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=25)
                        yield f"data: {json.dumps(item)}\n\n".encode()
                    except asyncio.TimeoutError:
                        yield b": keepalive\n\n"
            finally:
                monitor.listeners.discard(queue)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return Starlette(
        routes=[
            Route("/api/jobs", guard(jobs)),
            Route("/api/jobs/{job_id}", guard(job_detail)),
            Route("/api/jobs/{job_id}/artifacts", guard(artifacts)),
            Route("/api/stream", guard(stream)),
            Mount("/metrics", app=make_asgi_app()),
        ]
    )
