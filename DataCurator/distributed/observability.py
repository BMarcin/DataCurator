"""Worker-side observability: fleet heartbeats and logging sinks.

Errors are data; logs answer "why". Full logs stay in rotating local files —
the only sink that survives a network incident. WARNING+ (and chunk lifecycle
at INFO, tagged ``lifecycle=True``) push to Loki over its HTTP API,
**fail-open**: logging never blocks or crashes the pipeline; if Loki is down
we drop-and-count and rely on the local files.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

# Reused, never modified: the reporter's best-effort system sampling.
from DataCurator.reporting.metrics import sample_system_metrics

from . import wire

FLEET_HEARTBEAT_INTERVAL_S = 10
LOKI_FLUSH_INTERVAL_S = 2.0
LOKI_MAX_BATCH = 500


def fleet_heartbeat_payload(worker) -> bytes:
    """The ephemeral observability heartbeat — never load-bearing."""
    payload: Dict[str, Any] = {
        "worker_id": worker.worker_id,
        "capabilities": sorted(worker.capabilities),
        "in_flight": worker._in_flight,
        "version": worker.version,
        "ts": time.time(),
    }
    metrics = sample_system_metrics(sample_gpu=True)
    if metrics:
        payload["metrics"] = metrics
    return json.dumps(payload, separators=(",", ":")).encode()


class LokiSink:
    """A ~30-line batching push sink for loguru (drop-and-count on failure)."""

    def __init__(self, url: str, extra_labels: Optional[Dict[str, str]] = None) -> None:
        self.endpoint = url.rstrip("/") + "/loki/api/v1/push"
        self.extra_labels = extra_labels or {}
        self.dropped = 0
        self._buffer: List[tuple] = []
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=5.0)
        self._flusher = threading.Thread(target=self._flush_loop, daemon=True)
        self._flusher.start()

    def __call__(self, message) -> None:  # the loguru sink
        record = message.record
        labels = {
            "app": "datacurator",
            "worker": str(record["extra"].get("worker", "-")),
            "job": str(record["extra"].get("job", "-")),
            "stage": str(record["extra"].get("stage", "-")),
            "level": record["level"].name,
            **self.extra_labels,
        }
        ns = str(int(record["time"].timestamp() * 1e9))
        with self._lock:
            if len(self._buffer) < LOKI_MAX_BATCH * 4:  # bounded: drop, never grow
                self._buffer.append((tuple(sorted(labels.items())), ns, str(message).rstrip("\n")))
            else:
                self.dropped += 1

    def _flush_loop(self) -> None:
        while True:
            time.sleep(LOKI_FLUSH_INTERVAL_S)
            with self._lock:
                batch, self._buffer = self._buffer[:LOKI_MAX_BATCH], self._buffer[LOKI_MAX_BATCH:]
            if not batch:
                continue
            streams: Dict[tuple, List[List[str]]] = {}
            for labels, ns, line in batch:
                streams.setdefault(labels, []).append([ns, line])
            body = {
                "streams": [
                    {"stream": dict(labels), "values": values} for labels, values in streams.items()
                ]
            }
            try:
                self._client.post(self.endpoint, json=body)
            except Exception:
                self.dropped += len(batch)  # fail-open


def _loki_filter(record) -> bool:
    return record["level"].no >= logger.level("WARNING").no or record["extra"].get("lifecycle", False)


def setup_worker_logging(worker_id: str) -> None:
    """Rotating local files (all levels) + the Loki push sink (fail-open)."""
    logger.add(
        f"logs/worker-{worker_id}.log",
        rotation="100 MB",
        retention=10,
        level="DEBUG",
        enqueue=True,
    )
    loki_url = os.environ.get("DC_LOKI_URL")
    if loki_url:
        logger.add(LokiSink(loki_url), level="INFO", filter=_loki_filter)
        logger.info(f"Loki sink attached: {loki_url}")
