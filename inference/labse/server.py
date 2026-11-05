"""The LaBSE embedding service: a small async HTTP surface over the model.

Follows the repo's house style for HTTP services (``inference/metricx/server.py``,
``DataCurator/distributed/panel_api.py`` + ``monitor.py``): Starlette +
``uvicorn.Server(...).serve()``, loguru logging, ``os.environ.get`` config, a
``make_asgi_app()`` Prometheus mount at ``/metrics``, and an optional static
bearer-token guard (open by default, enforced only when ``LABSE_TOKEN`` is set).

Endpoints:
  POST /embed   {"texts": ["...", ...]}
                -> {"embeddings": [[...768...], ...], "dim": 768}
  GET  /health  -> {"status", "model_loaded", "dim", "last_error"}
                (200 only once the model loads)
  GET  /metrics -> Prometheus
  GET  /docs    -> Swagger UI (spec at /openapi.json)

The module itself imports no torch: ``build_app`` takes either a ready
``infer_fn`` (tests) or a blocking ``model_loader`` that is run on a background
thread at startup (production). This keeps the HTTP/batching logic unit-testable
without the heavy sentence-transformers stack.
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

L_REQUESTS = Counter("labse_requests_total", "Embedding requests", ["outcome"])
L_INFLIGHT = Gauge("labse_inflight_requests", "In-flight embedding requests")
L_TEXTS = Counter("labse_texts_total", "Texts embedded")
L_BATCH = Histogram(
    "labse_batch_size",
    "Items per inference batch",
    buckets=(1, 2, 4, 8, 16, 32, 48, 64, 96, 128),
)
L_QUEUE = Gauge("labse_queue_depth", "Pending items in the batcher queue")
L_LATENCY = Histogram(
    "labse_request_latency_seconds",
    "End-to-end /embed latency",
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
        "title": "LaBSE embedding service",
        "version": "1.0",
        "description": (
            "Batched LaBSE sentence embeddings for margin-based bilingual pair "
            "filtering. Returns one **L2-normalized** 768-dim vector per input "
            "text, order-aligned with the request, so a downstream inner "
            "product is cosine similarity. Auth is optional: a bearer token is "
            "required only when the service is configured with `LABSE_TOKEN`."
        ),
    },
    "paths": {
        "/embed": {
            "post": {
                "summary": "Embed texts",
                "security": [{}, {"bearerAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["texts"],
                                "properties": {
                                    "texts": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    }
                                },
                            },
                            "example": {"texts": ["The cat sat.", "Kot usiadł."]},
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Unit-norm vectors, one per text, in input order.",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "embeddings": {
                                            "type": "array",
                                            "items": {
                                                "type": "array",
                                                "items": {"type": "number"},
                                            },
                                        },
                                        "dim": {"type": "integer"},
                                    },
                                }
                            }
                        },
                    },
                    "400": {"description": "Malformed body (missing/!string 'texts')."},
                    "401": {"description": "Missing/wrong bearer token."},
                    "500": {"description": "Inference failed."},
                    "503": {"description": "Model still loading."},
                },
            }
        },
        "/health": {
            "get": {
                "summary": "Readiness + embedding dimension",
                "responses": {
                    "200": {
                        "description": "Model loaded.",
                        "content": {
                            "application/json": {
                                "example": {
                                    "status": "ok",
                                    "model_loaded": True,
                                    "dim": 768,
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
        "securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer"}},
    },
}

DOCS_HTML = """<!doctype html>
<html>
<head>
  <title>LaBSE embedding service — API docs</title>
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
    dim: Optional[int] = None,
    load_retry_s: float = 60.0,
) -> Starlette:
    """Build the ASGI app.

    Exactly one of ``infer_fn`` (already loaded — used by tests; maps a list of
    texts to a list of vectors) or ``model_loader`` (a blocking callable
    returning an object with ``.embed_batch(texts)`` and ``.dim`` — production)
    should be given. With ``model_loader``, ``dim`` is taken from the loaded
    embedder.
    """
    # Mutable startup state, flipped once the model is ready.
    state = {
        "infer_fn": infer_fn,
        "ready": infer_fn is not None,
        "dim": dim,
        "last_error": None,
    }

    def _infer(texts: list) -> list:
        fn = state["infer_fn"]
        if fn is None:  # guarded by the readiness check, should not happen
            raise RuntimeError("model not loaded")
        return fn(texts)

    batcher = DynamicBatcher(
        _infer,
        max_batch=max_batch,
        max_wait_ms=max_wait_ms,
        on_batch=L_BATCH.observe,
    )

    @contextlib.asynccontextmanager
    async def lifespan(app):
        tasks = [asyncio.create_task(batcher.run(), name="batcher")]

        async def _sample_queue():
            while True:
                L_QUEUE.set(batcher.pending())
                await asyncio.sleep(0.5)

        tasks.append(asyncio.create_task(_sample_queue(), name="queue-sampler"))

        if model_loader is not None:
            async def _load():
                # An uncaught exception here would die inside a never-awaited
                # task and leave /health at "loading" forever — log and retry.
                while True:
                    logger.info("loading LaBSE model...")
                    t0 = time.monotonic()
                    try:
                        embedder = await asyncio.to_thread(model_loader)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception(
                            "model load failed; retrying in {}s", load_retry_s
                        )
                        state["last_error"] = f"{type(exc).__name__}: {exc}"
                        await asyncio.sleep(load_retry_s)
                        continue
                    state["infer_fn"] = embedder.embed_batch
                    state["dim"] = getattr(embedder, "dim", None)
                    state["ready"] = True
                    state["last_error"] = None
                    logger.info("LaBSE model ready in {:.1f}s", time.monotonic() - t0)
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
                "dim": state["dim"],
                "last_error": state["last_error"],
            },
            status_code=200 if ready else 503,
        )

    async def embed(request: Request) -> JSONResponse:
        if not _authorized(request, token):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if not state["ready"]:
            L_REQUESTS.labels("unready").inc()
            return JSONResponse({"error": "model still loading"}, status_code=503)

        try:
            body = await request.json()
            texts = body["texts"]
            assert isinstance(texts, list)
            for t in texts:
                if not isinstance(t, str):
                    raise ValueError("each item in 'texts' must be a string")
        except (KeyError, ValueError, AssertionError, TypeError) as exc:
            L_REQUESTS.labels("bad_request").inc()
            return JSONResponse({"error": f"bad request: {exc}"}, status_code=400)

        if not texts:
            L_REQUESTS.labels("ok").inc()
            return JSONResponse({"embeddings": [], "dim": state["dim"]})

        L_INFLIGHT.inc()
        start = time.monotonic()
        try:
            embeddings: List[List[float]] = await asyncio.gather(
                *(batcher.submit(t) for t in texts)
            )
        except Exception as exc:  # noqa: BLE001 — inference failure -> 5xx (fail-open client)
            logger.exception("embedding failed")
            L_REQUESTS.labels("error").inc()
            return JSONResponse({"error": f"inference failed: {exc}"}, status_code=500)
        finally:
            L_INFLIGHT.dec()
            L_LATENCY.observe(time.monotonic() - start)

        L_REQUESTS.labels("ok").inc()
        L_TEXTS.inc(len(texts))
        return JSONResponse({"embeddings": embeddings, "dim": state["dim"]})

    async def docs(request: Request) -> HTMLResponse:
        return HTMLResponse(DOCS_HTML)

    async def openapi(request: Request) -> JSONResponse:
        return JSONResponse(OPENAPI)

    return Starlette(
        lifespan=lifespan,
        routes=[
            Route("/embed", embed, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
            Route("/docs", docs, methods=["GET"]),
            Route("/openapi.json", openapi, methods=["GET"]),
            Mount("/metrics", app=make_asgi_app()),
        ],
    )


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv

    from embedder import load_embedder_from_env

    load_dotenv()
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get("LABSE_LOG_LEVEL", "INFO"))

    app = build_app(
        model_loader=load_embedder_from_env,
        max_batch=int(os.environ.get("MAX_BATCH_SIZE", "64")),
        max_wait_ms=float(os.environ.get("MAX_BATCH_WAIT_MS", "8")),
        token=os.environ.get("LABSE_TOKEN"),
    )
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("LABSE_PORT", "8000")),
        log_level="warning",
    )
    asyncio.run(uvicorn.Server(config).serve())


if __name__ == "__main__":
    main()
