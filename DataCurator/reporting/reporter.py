"""Best-effort job-progress reporter that publishes to an S3 bucket.

A :class:`JobReporter` pushes a generic status document, system metrics and
stage artifacts to ``s3://<bucket>/<prefix>/<job_id>/`` so an external
dashboard can show live progress *without ever connecting to this machine*.
The producer only ever PUTs to S3; nothing inbound is exposed.

Every method is best-effort, mirroring
:func:`DataCurator.pipeline.notifications.notify`: a failed upload logs a
warning and returns. A reporting hiccup must never abort a job that is
halfway through. When reporting is disabled (or misconfigured) the runner
gets a :class:`NullReporter` whose methods are no-ops, so call sites stay
unconditional.

This module is deliberately job-agnostic — it knows the *envelope*, not what
``detail`` means. See ``job-panel/README.md`` for the full contract and
``DataCurator.reporting.schema`` for the envelope shape.
"""
from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import orjson
from loguru import logger

from DataCurator.reporting.metrics import gpu_names, sample_system_metrics
from DataCurator.reporting.s3 import S3Sink
from DataCurator.reporting.schema import (
    JobState,
    Link,
    Progress,
    build_envelope,
    now_iso,
)

StateLike = Union[JobState, str]


def _coerce_state(state: StateLike) -> str:
    """Normalise a :class:`JobState` or raw string to its wire value."""
    return state.value if isinstance(state, JobState) else str(state)


class NullReporter:
    """A reporter that does nothing — used when reporting is off or misconfigured.

    Implements the full interface as no-ops so callers never branch on whether
    reporting is enabled. ``enabled`` lets a caller skip expensive prep work
    (e.g. building a detail payload) when there is nothing to report to.
    """

    enabled: bool = False
    heartbeat_interval_s: float = 10.0
    upload_artifacts: bool = False

    def start(self, detail: Optional[Dict[str, Any]] = None) -> None:
        """No-op start."""

    def update(
        self,
        *,
        progress: Optional[Progress] = None,
        detail: Optional[Dict[str, Any]] = None,
        state: Optional[StateLike] = None,
        force: bool = False,
    ) -> None:
        """No-op update."""

    def log_metric(self, step: int, series: str, value: float, *, force: bool = False) -> None:
        """No-op metric."""

    def upload_artifact(
        self,
        local_path: Path,
        *,
        label: Optional[str] = None,
        complete: bool = True,
        dest: Optional[str] = None,
    ) -> None:
        """No-op artifact upload."""

    def ensure_artifact(
        self,
        local_path: Path,
        *,
        label: Optional[str] = None,
        complete: bool = True,
        dest: Optional[str] = None,
    ) -> None:
        """No-op artifact reconcile."""

    def finish(
        self, state: StateLike = JobState.SUCCEEDED, detail: Optional[Dict[str, Any]] = None
    ) -> None:
        """No-op finish."""


class JobReporter:
    """Publishes a job's live status, metrics and artifacts to S3."""

    enabled: bool = True

    def __init__(
        self,
        sink: S3Sink,
        *,
        job_id: str,
        job_type: str,
        name: str,
        host: Optional[Dict[str, Any]] = None,
        links: Optional[List[Link]] = None,
        prefix: str = "jobs",
        heartbeat_interval_s: float = 10.0,
        sample_metrics: bool = True,
        upload_artifacts: bool = True,
        metrics_flush_s: float = 15.0,
    ) -> None:
        """Capture identity and cadence; nothing is uploaded until :meth:`start`."""
        self._sink = sink
        self.job_id = job_id
        self.job_type = job_type
        self.name = name
        self.host = host
        self.links = links
        self.prefix = prefix.strip("/")
        self.heartbeat_interval_s = heartbeat_interval_s
        self.sample_metrics = sample_metrics
        self.upload_artifacts = upload_artifacts
        self.metrics_flush_s = metrics_flush_s

        self.started_at = now_iso()
        self._state = JobState.QUEUED.value
        self._progress: Optional[Progress] = None
        self._detail: Optional[Dict[str, Any]] = None
        self._metrics: Optional[Dict[str, Any]] = None
        self._last_push = 0.0  # monotonic clock of the last status PUT
        self._metric_lines: List[bytes] = []
        self._last_metrics_flush = 0.0
        self._manifest: Dict[str, Dict[str, Any]] = {}
        # Monotonic artifact revision: bumped on every successful upload and
        # carried in each status push so the dashboard notices new shards
        # without polling the manifest. See :func:`build_envelope`.
        self._artifacts_rev = 0

    # -- key helpers --------------------------------------------------------
    def _key(self, name: str) -> str:
        """Object key for ``name`` under this job's prefix."""
        return f"{self.prefix}/{self.job_id}/{name}"

    # -- status -------------------------------------------------------------
    def start(self, detail: Optional[Dict[str, Any]] = None) -> None:
        """Mark the job running and push the first status document."""
        self._state = JobState.RUNNING.value
        if detail is not None:
            self._detail = detail
        self._push(force=True)

    def update(
        self,
        *,
        progress: Optional[Progress] = None,
        detail: Optional[Dict[str, Any]] = None,
        state: Optional[StateLike] = None,
        force: bool = False,
    ) -> None:
        """Update in-memory state and push it, throttled to the heartbeat cadence.

        Unless ``force`` is set, a push is skipped when the previous one was
        less than ``heartbeat_interval_s`` ago — so the runner can call this
        on every record without hammering S3, while liveness is still
        refreshed at least once per interval by the heartbeat loop.
        """
        if progress is not None:
            self._progress = progress
        if detail is not None:
            self._detail = detail
        if state is not None:
            self._state = _coerce_state(state)
        self._push(force=force)

    def _push(self, *, force: bool) -> None:
        """Build and upload ``status.json`` (best-effort, throttled)."""
        now = time.monotonic()
        if not force and (now - self._last_push) < self.heartbeat_interval_s:
            return
        if self.sample_metrics:
            self._metrics = sample_system_metrics()
        envelope = build_envelope(
            job_id=self.job_id,
            job_type=self.job_type,
            name=self.name,
            state=self._state,
            started_at=self.started_at,
            heartbeat_interval_s=self.heartbeat_interval_s,
            host=self.host,
            progress=self._progress,
            metrics=self._metrics,
            links=self.links,
            detail=self._detail,
            artifacts_rev=self._artifacts_rev,
        )
        try:
            self._sink.put_bytes(self._key("status.json"), orjson.dumps(envelope))
            self._last_push = now
        except Exception as exc:  # noqa: BLE001 - reporting must never abort the job.
            logger.warning(f"job reporting: status push failed: {exc}")

    # -- metrics time series ------------------------------------------------
    def log_metric(self, step: int, series: str, value: float, *, force: bool = False) -> None:
        """Append a point to the ``metrics.jsonl`` time series.

        S3 has no append, so the buffered series is re-uploaded whole, no more
        often than every ``metrics_flush_s`` (or immediately when ``force``).
        """
        line = orjson.dumps(
            {"ts": now_iso(), "step": step, "series": series, "value": value},
            option=orjson.OPT_APPEND_NEWLINE,
        )
        self._metric_lines.append(line)
        now = time.monotonic()
        if force or (now - self._last_metrics_flush) >= self.metrics_flush_s:
            try:
                self._sink.put_bytes(self._key("metrics.jsonl"), b"".join(self._metric_lines))
                self._last_metrics_flush = now
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"job reporting: metrics flush failed: {exc}")

    # -- artifacts ----------------------------------------------------------
    def upload_artifact(
        self,
        local_path: Path,
        *,
        label: Optional[str] = None,
        complete: bool = True,
        dest: Optional[str] = None,
    ) -> None:
        """Upload ``local_path`` and refresh ``manifest.json``.

        ``dest`` is the artifact's relative key under the job (e.g.
        ``artifacts/00_review/step-0000.jsonl``); it preserves the per-stage
        grouping the runner produces. When omitted the artifact lands flat under
        ``artifacts/<filename>``. A successful upload bumps ``_artifacts_rev`` so
        the next status push tells the dashboard the artifact list changed.
        """
        if not self.upload_artifacts:
            return
        key = dest or f"artifacts/{local_path.name}"
        try:
            self._sink.upload_file(local_path, self._key(key))
            self._register(key, local_path, label or local_path.name, complete)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"job reporting: artifact upload failed ({local_path}): {exc}")

    def ensure_artifact(
        self,
        local_path: Path,
        *,
        label: Optional[str] = None,
        complete: bool = True,
        dest: Optional[str] = None,
    ) -> None:
        """Register ``local_path`` in the manifest, uploading only if S3 lacks it.

        The reconcile counterpart to :meth:`upload_artifact`: on resume a stage's
        shards may already sit in S3 from an earlier process but be absent from
        *this* process's in-memory manifest. This re-lists them (HEAD-gated, so
        bytes already present are not re-sent) so ``manifest.json`` reflects every
        artifact on S3, not just the ones this process uploaded.
        """
        if not self.upload_artifacts:
            return
        key = dest or f"artifacts/{local_path.name}"
        try:
            if not self._sink.exists(self._key(key)):
                self._sink.upload_file(local_path, self._key(key))
            self._register(key, local_path, label or local_path.name, complete)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"job reporting: artifact reconcile failed ({local_path}): {exc}")

    def _register(self, key: str, local_path: Path, label: str, complete: bool) -> None:
        """Record ``key`` in the manifest and re-publish ``manifest.json``.

        Shared tail of :meth:`upload_artifact` and :meth:`ensure_artifact`: S3
        has no append, so the whole manifest is rebuilt from ``_manifest`` and
        re-PUT. Bumps ``_artifacts_rev`` so the next status push tells the
        dashboard the artifact list changed.
        """
        self._manifest[key] = {
            "key": key,
            "label": label,
            "size_bytes": local_path.stat().st_size,
            "updated_at": now_iso(),
            "complete": complete,
        }
        manifest = {"artifacts": list(self._manifest.values())}
        self._sink.put_bytes(self._key("manifest.json"), orjson.dumps(manifest))
        self._artifacts_rev += 1

    # -- terminal state -----------------------------------------------------
    def finish(
        self, state: StateLike = JobState.SUCCEEDED, detail: Optional[Dict[str, Any]] = None
    ) -> None:
        """Push the terminal status (``succeeded``/``failed``) and flush metrics."""
        if detail is not None:
            self._detail = detail
        self._state = _coerce_state(state)
        if self._metric_lines:
            try:
                self._sink.put_bytes(self._key("metrics.jsonl"), b"".join(self._metric_lines))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"job reporting: final metrics flush failed: {exc}")
        self._push(force=True)


Reporter = Union[JobReporter, NullReporter]


# --------------------------------------------------------------------------- #
# Building a reporter from Hydra config
# --------------------------------------------------------------------------- #
def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a DictConfig/dict, returning ``default`` when absent."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def _parse_links(raw: Any) -> List[Link]:
    """Turn a config ``links`` list of ``{label, url}`` into :class:`Link`s."""
    links: List[Link] = []
    for entry in raw or []:
        label = _get(entry, "label")
        url = _get(entry, "url")
        if label and url:
            links.append(Link(label=str(label), url=str(url)))
    return links


def build_reporter(
    reporting_cfg: Any, *, name: str, job_cfg: Any = None
) -> Reporter:
    """Build a :class:`JobReporter` from the ``reporting`` and ``job`` configs.

    Connection and cadence come from ``reporting_cfg`` (the backend); job
    identity — type and deep-links — comes from ``job_cfg`` (the experiment's
    ``job:`` block). The job_id is always ``name`` (the experiment name), so
    reruns of an experiment share one dashboard entry.

    Returns a :class:`NullReporter` (a safe no-op) whenever reporting is
    disabled, uses an unknown backend, or is missing the bucket it would need
    — the job should run regardless of whether the dashboard is wired up.
    """
    if not _get(reporting_cfg, "enabled", False):
        return NullReporter()

    backend = _get(reporting_cfg, "backend", "s3")
    if backend != "s3":
        logger.warning(f"job reporting: unknown backend {backend!r}; reporting disabled")
        return NullReporter()

    bucket = _get(reporting_cfg, "bucket")
    if not bucket:
        logger.warning("job reporting enabled but `reporting.bucket` is unset; reporting disabled")
        return NullReporter()

    job_id = name
    prefix = str(_get(reporting_cfg, "prefix", "jobs"))
    host = {"name": socket.gethostname()}
    gpus = gpu_names()
    if gpus:
        host["gpus"] = gpus

    try:
        sink = S3Sink(
            endpoint_url=_get(reporting_cfg, "endpoint_url"),
            region=str(_get(reporting_cfg, "region", "garage")),
            bucket=str(bucket),
            access_key_id=_get(reporting_cfg, "access_key_id"),
            secret_access_key=_get(reporting_cfg, "secret_access_key"),
        )
    except Exception as exc:  # noqa: BLE001 - a bad client config disables reporting.
        logger.warning(f"job reporting: could not build S3 client: {exc}; reporting disabled")
        return NullReporter()

    reporter = JobReporter(
        sink,
        job_id=job_id,
        job_type=str(_get(job_cfg, "type", "data-pipeline")),
        name=name,
        host=host,
        links=_parse_links(_get(job_cfg, "links")),
        prefix=prefix,
        heartbeat_interval_s=float(_get(reporting_cfg, "heartbeat_interval_s", 10)),
        sample_metrics=bool(_get(reporting_cfg, "sample_system_metrics", True)),
        upload_artifacts=bool(_get(reporting_cfg, "upload_artifacts", True)),
        metrics_flush_s=float(_get(reporting_cfg, "metrics_flush_s", 15)),
    )
    logger.info(f"job reporting enabled -> s3://{bucket}/{prefix}/{job_id}/")
    return reporter
