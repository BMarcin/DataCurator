"""Async client for the MetricX-QE inference service.

The service (see ``inference/metricx/README.md``) keeps MetricX-23-QE warm on
a GPU and scores translations reference-free: ``POST /score`` with
``{"pairs": [{"source", "hypothesis"}, ...]}`` returns order-aligned
``{"scores": [...]}`` on a 0–25 scale where **lower is better**.

Every way the service can fail to produce scores (unbound endpoint, transport
error, exhausted retries, a non-transient 4xx, a malformed payload) funnels
into one :class:`MetricXUnavailable`. Keep that funnel tight: any other
exception escaping ``score_pairs`` would be retried by the LLM stage's
resample loop and burn model calls on a client bug.

Both consumers — :class:`~DataCurator.pipeline.validators.MetricXValidator`
and :class:`~DataCurator.pipeline.modifiers.metricx.MetricXScoreModifier` —
let that exception propagate: a declared guard always runs, and a service
that cannot answer is an outage, not a silent skip. Because
:class:`MetricXUnavailable` is a
:class:`~DataCurator.pipeline.errors.ServiceUnavailableError` it reads as
connection-class to the distributed worker, which naks the chunk for
redelivery rather than uploading it with every record flagged as bad data.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional, Sequence, Tuple

import httpx
from loguru import logger
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from DataCurator.pipeline.errors import ServiceUnavailableError

# 4xx statuses that clear on their own, so they are worth another attempt:
# 403 is the edge/WAF signature block (Cloudflare's "error code: 1010"), 408
# and 425 are transport-ish timeouts, 429 is rate limiting. Everything else
# 4xx — a malformed body, a bad token, a wrong path — never fixes itself, and
# retrying it only stalls the record behind the full backoff budget.
TRANSIENT_STATUSES = frozenset({403, 408, 425, 429})


class MetricXUnavailable(ServiceUnavailableError):
    """The MetricX service could not produce scores for this request."""


class _Retryable(Exception):
    """Internal marker: a transient failure worth another attempt."""


class MetricXClient:
    """Thin async wrapper around the service's ``/score`` endpoint.

    The ``httpx.AsyncClient`` is built lazily behind a lock, so constructing
    a pipeline never needs the service reachable. Retries are short and
    bounded, and stay that way on purpose: they exist to absorb a blip (the
    model-loading 503, a momentary edge block), while a real outage is meant
    to surface fast so the caller's chunk can be nak'd and redelivered
    later — a long client-side budget would just stall every record in the
    chunk behind a dead service.
    """

    def __init__(
        self,
        remote_server: Optional[str],
        *,
        api_token: Optional[str] = None,
        timeout: float = 180.0,
        retries: int = 3,
    ) -> None:
        self.remote_server = remote_server.rstrip("/") if remote_server else None
        self.api_token = api_token
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None:
                headers = {}
                if self.api_token:
                    headers["Authorization"] = f"Bearer {self.api_token}"
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout), headers=headers)
            return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Score ``(source, hypothesis)`` pairs; raise :class:`MetricXUnavailable` on any failure."""
        if not pairs:
            return []
        if not self.remote_server:
            raise MetricXUnavailable("no MetricX endpoint bound (remote_server is unset)")
        client = await self._get_client()
        body = {"pairs": [{"source": src, "hypothesis": hyp} for src, hyp in pairs]}
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.retries),
                wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
                retry=retry_if_exception_type((httpx.TransportError, _Retryable)),
                reraise=True,
            ):
                with attempt:
                    response = await client.post(f"{self.remote_server}/score", json=body)
                    status = response.status_code
                    if status >= 500 or status in TRANSIENT_STATUSES:
                        # 503 covers the model-loading window; TRANSIENT_STATUSES
                        # covers the edge blocking us rather than the service.
                        raise _Retryable(f"MetricX returned {status} {response.text[:200]}")
                    if status != 200:  # the rest of 4xx: retrying won't help
                        raise MetricXUnavailable(
                            f"MetricX rejected the request: {status} {response.text[:200]}"
                        )
                    scores = response.json().get("scores")
                    if not isinstance(scores, list) or len(scores) != len(pairs):
                        # A contract violation, not a blip — log loudly, but it is
                        # still just "no scores" to the caller.
                        logger.error(
                            f"MetricX payload mismatch: {len(pairs)} pairs sent, got {scores!r:.200}"
                        )
                        raise MetricXUnavailable("MetricX response did not match the request pairs")
                    return [float(score) for score in scores]
        except MetricXUnavailable:
            raise
        except (httpx.TransportError, _Retryable) as exc:
            raise MetricXUnavailable(
                f"cannot reach MetricX at {self.remote_server!r} after {self.retries} attempts: {exc}"
            ) from exc
        except (ValueError, TypeError) as exc:  # non-JSON body, non-numeric scores
            raise MetricXUnavailable(f"malformed MetricX response: {exc}") from exc
        raise RuntimeError("unreachable: retry loop exited without returning")  # pragma: no cover
