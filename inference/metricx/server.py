"""The MetricX inference service: a small async HTTP surface over the model.

Follows the repo's house style for HTTP services (``DataCurator/distributed/
panel_api.py`` + ``monitor.py``): Starlette + ``uvicorn.Server(...).serve()``,
loguru logging, ``os.environ.get`` config, a ``make_asgi_app()`` Prometheus
mount at ``/metrics``, and an optional static bearer-token guard (open by
default, like the LanguageTool service — enforced only when ``METRICX_TOKEN``
is set).

Endpoints:
  POST /score   {"pairs": [{"source", "hypothesis"[, "reference"]}, ...]}
                -> {"scores": [...]}  ("reference" only on metricx-24-hybrid)
  GET  /health  -> {"status", "model_loaded", "accepts_reference"}
                (200 only once the model loads)
  GET  /metrics -> Prometheus
  GET  /docs    -> Swagger UI (spec at /openapi.json)

The module itself imports no torch: ``build_app`` takes either a ready
``infer_fn`` (tests) or a blocking ``model_loader`` that is run on a background
thread at startup (production). This keeps the HTTP/batching logic unit-testable
without the heavy MetricX stack.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from typing import Callable, List, Optional

from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Mount, Route

from batcher import DynamicBatcher

M_REQUESTS = Counter("metricx_requests_total", "Scoring requests", ["outcome"])
M_INFLIGHT = Gauge("metricx_inflight_requests", "In-flight scoring requests")
M_PAIRS = Counter("metricx_pairs_total", "Pairs scored")
M_BATCH = Histogram(
    "metricx_batch_size",
    "Items per inference batch",
    buckets=(1, 2, 4, 8, 16, 32, 48, 64, 96, 128),
)
M_QUEUE = Gauge("metricx_queue_depth", "Pending items in the batcher queue")
M_LATENCY = Histogram(
    "metricx_request_latency_seconds",
    "End-to-end /score latency",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)


def _authorized(request: Request, token: Optional[str]) -> bool:
    if not token:
        return True  # no token configured -> the service is open (internal LAN)
    return request.headers.get("authorization") == f"Bearer {token}"


# Hand-written spec (Starlette has no auto-schema); served at /openapi.json and
# rendered by the Swagger UI shell at /docs.
OPENAPI = {
    "openapi": "3.0.3",
    "info": {
        "title": "MetricX inference service",
        "version": "1.0",
        "description": (
            "Batched MetricX translation scoring. Scores are 0-25, **lower is "
            "better**, order-aligned with the submitted pairs. `reference` "
            "(ground truth) is only accepted on `metricx-24-hybrid` deployments "
            "— check `accepts_reference` in `/health`; pairs without one are "
            "scored QE-style. Auth is optional: a bearer token is required only "
            "when the service is configured with `METRICX_TOKEN`."
        ),
    },
    "paths": {
        "/score": {
            "post": {
                "summary": "Score translation pairs",
                "security": [{}, {"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["pairs"],
                                "properties": {
                                    "pairs": {
                                        "type": "array",
                                        "items": {"$ref": "#/components/schemas/Pair"},
                                    }
                                },
                            },
                            "example": {
                                "pairs": [
                                    {
                                        "source": "The cat sat.",
                                        "hypothesis": "Kot usiadł.",
                                        "reference": "Kot siedział.",
                                    }
                                ]
                            },
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Scores, one per pair, in input order.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "scores": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                        }
                                    },
                                }
                            }
                        },
                    },
                    "400": {
                        "description": (
                            "Malformed body, or a 'reference' sent to a "
                            "reference-free (QE) deployment."
                        )
                    },
                    "401": {"description": "Missing/wrong bearer token."},
                    "500": {"description": "Inference failed."},
                    "503": {"description": "Model still loading."},
                },
            }
        },
        "/health": {
            "get": {
                "summary": "Readiness + deployment capabilities",
                "responses": {
                    "200": {
                        "description": "Model loaded.",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "ok",
                                    "model_loaded": True,
                                    "accepts_reference": True,
                                }
                            }
                        },
                    },
                    "503": {"description": "Model still loading."},
                },
            }
        },
        "/metrics": {
            "get": {
                "summary": "Prometheus metrics",
                "responses": {"200": {"description": "Prometheus text format."}},
            }
        },
    },
    "components": {
        "schemas": {
            "Pair": {
                "type": "object",
                "required": ["source", "hypothesis"],
                "properties": {
                    "source": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "reference": {
                        "type": "string",
                        "description": (
                            "Ground truth; metricx-24-hybrid only. Absent/empty "
                            "-> the pair is scored QE-style."
                        ),
                    },
                },
            }
        },
        "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}},
    },
}

DOCS_HTML = """<!doctype html>
<html>
<head>
  <title>MetricX inference service — API docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({url: "/openapi.json", dom_id: "#swagger-ui"});</script>
</body>
</html>"""


def build_app(
    *,
    infer_fn: Optional[Callable[[list], list]] = None,
    model_loader: Optional[Callable[[], object]] = None,
    max_batch: int = 64,
    max_wait_ms: float = 8.0,
    token: Optional[str] = None,
    accepts_reference: bool = False,
    load_retry_s: float = 60.0,
) -> Starlette:
    """Build the ASGI app.

    Exactly one of ``infer_fn`` (already loaded — used by tests) or
    ``model_loader`` (a blocking callable returning an object with
    ``.score_batch(pairs)`` — used in production) should be given. With
    ``model_loader``, ``accepts_reference`` is taken from the loaded scorer.
    """
    # Mutable startup state, flipped once the model is ready.
    state = {
        "infer_fn": infer_fn,
        "ready": infer_fn is not None,
        "accepts_reference": accepts_reference,
        "last_error": None,
    }

    def _infer(pairs: list) -> list:
        fn = state["infer_fn"]
        if fn is None:  # guarded by the readiness check, should not happen
            raise RuntimeError("model not loaded")
        return fn(pairs)

    batcher = DynamicBatcher(
        _infer,
        max_batch=max_batch,
        max_wait_ms=max_wait_ms,
        on_batch=M_BATCH.observe,
    )

    @contextlib.asynccontextmanager
    async def lifespan(app):
        tasks = [asyncio.create_task(batcher.run(), name="batcher")]

        async def _sample_queue():
            while True:
                M_QUEUE.set(batcher.pending())
                await asyncio.sleep(0.5)

        tasks.append(asyncio.create_task(_sample_queue(), name="queue-sampler"))

        if model_loader is not None:
            async def _load():
                # An uncaught exception here would die inside a never-awaited
                # task and leave /health at "loading" forever — log and retry.
                while True:
                    logger.info("loading MetricX model...")
                    t0 = time.monotonic()
                    try:
                        scorer = await asyncio.to_thread(model_loader)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "model load failed; retrying in {}s", load_retry_s
                        )
                        state["last_error"] = f"{type(exc).__name__}: {exc}"
                        await asyncio.sleep(load_retry_s)
                        continue
                    state["infer_fn"] = scorer.score_batch
                    state["accepts_reference"] = getattr(scorer, "accepts_reference", False)
                    state["ready"] = True
                    state["last_error"] = None
                    logger.info("MetricX model ready in {:.1f}s", time.monotonic() - t0)
                    return

            tasks.append(asyncio.create_task(_load(), name="model-loader"))

        try:
            yield
        finally:
            for t in tasks:
                t.cancel()

    async def health(request: Request) -> JSONResponse:
        ready = state["ready"]
        return JSONResponse(
            {
                "status": "ok" if ready else "loading",
                "model_loaded": ready,
                "accepts_reference": state["accepts_reference"],
                "last_error": state["last_error"],
            },
            status_code=200 if ready else 503,
        )

    async def score(request: Request) -> JSONResponse:
        if not _authorized(request, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if not state["ready"]:
            M_REQUESTS.labels("unready").inc()
            return JSONResponse({"error": "model still loading"}, status_code=503)

        try:
            body = await request.json()
            pairs = body["pairs"]
            assert isinstance(pairs, list)
            for p in pairs:
                if not isinstance(p.get("source"), str) or not isinstance(
                    p.get("hypothesis"), str
                ):
                    raise ValueError("each pair needs string 'source' and 'hypothesis'")
                # Reject bad references up front — an encode error inside the
                # batcher would 500 every request sharing the coalesced batch.
                if "reference" in p and not isinstance(p["reference"], str):
                    raise ValueError("'reference' must be a string when present")
                if "reference" in p and not state["accepts_reference"]:
                    raise ValueError(
                        "this deployment is reference-free (QE); 'reference' not accepted"
                    )
        except (KeyError, ValueError, AssertionError, TypeError) as exc:
            M_REQUESTS.labels("bad_request").inc()
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)

        if not pairs:
            M_REQUESTS.labels("ok").inc()
            return JSONResponse({"scores": []})

        M_INFLIGHT.inc()
        start = time.monotonic()
        try:
            scores: List[float] = await asyncio.gather(
                *(batcher.submit(p) for p in pairs)
            )
        except Exception as exc:  # noqa: BLE001 — inference failure -> 5xx (fail-open client)
            logger.exception("scoring failed")
            M_REQUESTS.labels("error").inc()
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        finally:
            M_INFLIGHT.dec()
            M_LATENCY.observe(time.monotonic() - start)

        M_REQUESTS.labels("ok").inc()
        M_PAIRS.inc(len(pairs))
        return JSONResponse({"scores": scores})

    async def docs(request: Request) -> HTMLResponse:
        return HTMLResponse(DOCS_HTML)

    async def openapi(request: Request) -> JSONResponse:
        return JSONResponse(OPENAPI)

    return Starlette(
        lifespan=lifespan,
        routes=[
            Route("/score", score, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            Route("/docs", docs, methods=["GET"]),
            Route("/openapi.json", openapi, methods=["GET"]),
            Mount("/metrics", app=make_asgi_app()),
        ],
    )


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv

    from metricx_model import load_scorer_from_env

    load_dotenv()
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("METRICX_LOG_LEVEL", "INFO"))

    app = build_app(
        model_loader=load_scorer_from_env,
        max_batch=int(os.environ.get("MAX_BATCH_SIZE", "64")),
        max_wait_ms=float(os.environ.get("MAX_BATCH_WAIT_MS", "8")),
        token=os.environ.get("METRICX_TOKEN"),
    )
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("METRICX_PORT", "8000")),
        log_level="warning",
    )
    asyncio.run(uvicorn.Server(config).serve())


if __name__ == "__main__":
    main()
