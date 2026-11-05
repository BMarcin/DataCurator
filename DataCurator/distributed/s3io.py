"""Chunk JSONL IO over the shared S3 store.

``DistStore`` extends the reporter's :class:`S3Sink` (reused, never modified)
with the listing/deletion the controller and monitor need. All methods are
synchronous boto3 — async callers wrap them in :func:`asyncio.to_thread`.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import orjson

from DataCurator.reporting.s3 import S3Sink


def upload_concurrency() -> int:
    """Concurrent S3 PUTs during seeding (env ``DC_SEED_UPLOAD_CONCURRENCY``, default 16).

    Chunk uploads are latency-bound network round-trips, so a thread pool
    (boto3 clients are thread-safe) collapses the serial wait; the client's
    connection pool is sized to match this.
    """
    return int(os.environ.get("DC_SEED_UPLOAD_CONCURRENCY", "16"))


def s3_retries() -> int:
    """Attempts per S3 call on a transient failure (env ``DC_S3_RETRIES``, default 6).

    The backoff doubles from 2 s up to ``DC_S3_RETRY_MAX_WAIT`` (default 60 s),
    so the default budget rides out roughly a minute of store trouble before
    the chunk fails and is nak'd for redelivery.
    """
    return int(os.environ.get("DC_S3_RETRIES") or "6")


def s3_retry_max_wait() -> float:
    """Cap on one backoff sleep between S3 retries (env ``DC_S3_RETRY_MAX_WAIT``, default 60)."""
    return float(os.environ.get("DC_S3_RETRY_MAX_WAIT") or "60")


class DistStore(S3Sink):
    """S3Sink plus prefix listing and batched deletion."""

    def list_keys(self, prefix: str) -> List[str]:
        """All object keys under ``prefix`` (paginated; a transient failure relists)."""

        def list_all() -> List[str]:
            """One full paginated listing."""
            keys: List[str] = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                keys.extend(obj["Key"] for obj in page.get("Contents", []))
            return keys

        return self._retrying(list_all)

    def delete_keys(self, keys: Iterable[str]) -> int:
        """Delete ``keys`` in batches of 1000 (the S3 API limit)."""
        batch: List[str] = []
        deleted = 0

        def flush() -> None:
            nonlocal deleted
            if batch:
                objects = [{"Key": k} for k in batch]
                self._retrying(
                    lambda: self._client.delete_objects(
                        Bucket=self.bucket, Delete={"Objects": objects, "Quiet": True}
                    )
                )
                deleted += len(batch)
                batch.clear()

        for key in keys:
            batch.append(key)
            if len(batch) == 1000:
                flush()
        flush()
        return deleted


def store_from_env() -> DistStore:
    """Build the data-bucket store from the environment.

    Reuses the reporter's ``JOB_S3_*`` connection variables; the data bucket
    may be overridden with ``DC_S3_BUCKET`` so job-status and pipeline data
    can live in separate buckets.
    """
    bucket = os.environ.get("DC_S3_BUCKET") or os.environ.get("JOB_S3_BUCKET")
    if not bucket:
        raise RuntimeError("set DC_S3_BUCKET (or JOB_S3_BUCKET) for the pipeline data bucket")
    return DistStore(
        endpoint_url=os.environ.get("JOB_S3_ENDPOINT"),
        region=os.environ.get("JOB_S3_REGION", "garage"),
        bucket=bucket,
        access_key_id=os.environ.get("JOB_S3_ACCESS_KEY_ID"),
        secret_access_key=os.environ.get("JOB_S3_SECRET_ACCESS_KEY"),
        max_pool_connections=upload_concurrency(),
        retries=s3_retries(),
        retry_max_wait=s3_retry_max_wait(),
    )


def s3_prefix() -> str:
    return os.environ.get("DC_S3_PREFIX", "pipeline")


def records_to_jsonl(records: Iterable[Dict[str, Any]]) -> bytes:
    """Serialise records as JSONL; an empty iterable yields empty bytes.

    The empty object still gets uploaded — "the output key exists" is the
    single completion signal everywhere.
    """
    return b"".join(orjson.dumps(r, option=orjson.OPT_APPEND_NEWLINE) for r in records)


def jsonl_to_records(data: bytes) -> List[Dict[str, Any]]:
    return [orjson.loads(line) for line in data.splitlines() if line.strip()]


def write_chunk(store: DistStore, key: str, records: Iterable[Dict[str, Any]]) -> None:
    store.put_bytes(key, records_to_jsonl(records), content_type="application/jsonl")


def write_chunks(
    store: DistStore,
    items: Iterable[Tuple[str, List[Dict[str, Any]]]],
    max_workers: Optional[int] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Write many ``(key, records)`` chunks concurrently; raise on the first failure.

    Each PUT is an independent network round-trip, so running them on a thread
    pool collapses N serial waits into ceil(N / max_workers) rounds. Defaults to
    :func:`upload_concurrency` workers. ``on_progress(done, total)`` is invoked
    from the calling thread as each chunk lands, for progress reporting.
    """
    items = list(items)
    if not items:
        return
    workers = upload_concurrency() if max_workers is None else max_workers
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        futures = [pool.submit(write_chunk, store, key, records) for key, records in items]
        done = 0
        # as_completed runs in the calling thread, so counting/progress needs no
        # lock; result() re-raises the first PUT that failed.
        for future in as_completed(futures):
            future.result()
            done += 1
            if on_progress is not None:
                on_progress(done, len(items))


def read_chunk(store: DistStore, key: str) -> List[Dict[str, Any]]:
    data = store.get_bytes(key)
    if data is None:
        raise FileNotFoundError(f"s3://{store.bucket}/{key} does not exist")
    return jsonl_to_records(data)
