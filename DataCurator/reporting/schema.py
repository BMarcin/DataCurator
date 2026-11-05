"""The job-reporting data contract.

Defines the *generic* status envelope that any job (a data pipeline today,
a training run tomorrow) publishes so a single external dashboard can render
it without knowing what produced it. The dashboard understands the envelope;
only a per-``type`` renderer interprets the free-form ``detail`` payload.

Nothing here is pipeline-specific — the runner builds its own ``detail`` and
hands it to :class:`~DataCurator.reporting.reporter.JobReporter`. Keep it
that way: adding a new job type must not require touching this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

#: Bumped only on a breaking change to the envelope shape. The dashboard
#: branches on this. See ``job-panel/README.md`` §3 for the full contract.
SCHEMA_VERSION = 1


class JobState(str, Enum):
    """Lifecycle state a producer writes into ``status.json``.

    ``STALE`` is intentionally absent: staleness is *computed by the
    dashboard* from a missed heartbeat, never written by the producer.
    """

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (contract format)."""
    return datetime.now(timezone.utc).isoformat()


def _compact(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys whose value is ``None`` so absent fields stay absent."""
    return {k: v for k, v in mapping.items() if v is not None}


@dataclass(frozen=True)
class Progress:
    """How far a job is through its work. ``total`` ``None`` => indeterminate."""

    done: int
    total: Optional[int] = None
    unit: str = "records"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the contract's ``progress`` object, omitting nulls."""
        return _compact({"done": self.done, "total": self.total, "unit": self.unit})


@dataclass(frozen=True)
class Link:
    """A deep-link into an external tool (e.g. TensorBoard, MLflow)."""

    label: str
    url: str

    def to_dict(self) -> Dict[str, str]:
        """Serialise to the contract's ``links[]`` entry."""
        return {"label": self.label, "url": self.url}


@dataclass(frozen=True)
class StageInfo:
    """One entry of a ``data-pipeline`` ``detail.stages`` list."""

    id: str
    state: str
    done: Optional[int] = None
    total: Optional[int] = None
    errored: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a ``detail.stages[]`` entry, omitting nulls."""
        return _compact(
            {
                "id": self.id,
                "state": self.state,
                "done": self.done,
                "total": self.total,
                "errored": self.errored,
            }
        )


def build_envelope(
    *,
    job_id: str,
    job_type: str,
    name: str,
    state: str,
    started_at: str,
    heartbeat_interval_s: float,
    host: Optional[Dict[str, Any]] = None,
    progress: Optional[Progress] = None,
    metrics: Optional[Dict[str, Any]] = None,
    links: Optional[List[Link]] = None,
    detail: Optional[Dict[str, Any]] = None,
    artifacts_rev: int = 0,
) -> Dict[str, Any]:
    """Assemble a complete ``status.json`` envelope as a plain dict.

    ``updated_at`` and ``heartbeat_at`` are stamped to *now* on every call —
    each push is a fresh liveness signal. Optional fields that are ``None``
    are omitted entirely so the reader can rely on "present means meaningful".

    ``artifacts_rev`` is a monotonic counter the producer bumps each time it
    uploads a new artifact (a closed output shard). It rides the frequently-
    pushed ``status.json`` so the dashboard can notice — without polling the
    manifest — that the artifact list changed and re-fetch it. Omitted while
    zero (no artifacts uploaded yet).
    """
    stamp = now_iso()
    return _compact(
        {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "type": job_type,
            "name": name,
            "state": state,
            "progress": progress.to_dict() if progress else None,
            "host": host or None,
            "metrics": metrics or None,
            "started_at": started_at,
            "updated_at": stamp,
            "heartbeat_at": stamp,
            "heartbeat_interval_s": heartbeat_interval_s,
            "links": [link.to_dict() for link in links] if links else None,
            "detail": detail or None,
            "artifacts_rev": artifacts_rev or None,
        }
    )
