"""Thin S3-compatible object sink used by the job reporter.

Wraps a :mod:`boto3` client configured for an arbitrary endpoint (Garage,
MinIO, AWS …). It exposes only the few operations the reporter needs —
put bytes, get bytes, upload a (possibly large) file — and leaves error
handling to the caller, which treats every reporting action as best-effort.

Every operation retries transient failures (5xx, throttling, dropped
connections) with exponential backoff before raising; a missing key is an
answer, not a failure, and is never retried.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, TypeVar

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, HTTPClientError
from botocore.exceptions import ConnectionError as BotoConnectionError
from loguru import logger
from tenacity import RetryCallState, Retrying, retry_if_exception, stop_after_attempt, wait_exponential

T = TypeVar("T")

# Error codes an S3 server returns for a condition that clears on its own.
TRANSIENT_CODES = frozenset(
    {"ServiceUnavailable", "SlowDown", "InternalError", "RequestTimeout", "Throttling", "ThrottlingException"}
)


def is_transient(exc: BaseException) -> bool:
    """True for S3 failures worth another attempt: transport errors, 5xx and throttling."""
    if isinstance(exc, (HTTPClientError, BotoConnectionError)):  # unreachable, reset, timed out
        return True
    if isinstance(exc, ClientError):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") or 0
        code = exc.response.get("Error", {}).get("Code")
        return status >= 500 or status == 429 or code in TRANSIENT_CODES
    return False


def _log_retry(state: RetryCallState) -> None:
    """Log each backoff so a flapping store is visible before it gives up."""
    exc = state.outcome.exception() if state.outcome else None
    logger.warning(
        f"S3 call failed (attempt {state.attempt_number}): {exc!r}; "
        f"retrying in {state.next_action.sleep if state.next_action else 0:.0f}s"
    )


class S3Sink:
    """A minimal put/get wrapper around one S3-compatible bucket."""

    def __init__(
        self,
        *,
        endpoint_url: Optional[str],
        region: str,
        bucket: str,
        access_key_id: Optional[str],
        secret_access_key: Optional[str],
        max_pool_connections: int = 10,
        retries: int = 6,
        retry_max_wait: float = 60.0,
    ) -> None:
        """Build a boto3 S3 client for ``bucket`` at ``endpoint_url``.

        Path-style addressing is forced because self-hosted S3 servers
        (Garage, MinIO) rarely support virtual-host buckets. ``max_pool_connections``
        (botocore's default is 10) caps how many requests can be in flight at once;
        raise it to match a thread pool that issues concurrent PUTs.

        Response checksums are validated only when requested: botocore's
        default wraps every body in a checksum stream, and on an *error*
        response its parser then crashes with ``AttributeError`` on that
        wrapper — masking the server's real status (e.g. Garage's 503) and
        skipping the retry. ``retries`` / ``retry_max_wait`` bound the
        exponential backoff each operation gets on a transient failure.
        """
        self.bucket = bucket
        self.retries = max(1, int(retries))
        self.retry_max_wait = float(retry_max_wait)
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(
                s3={"addressing_style": "path"},
                signature_version="s3v4",
                max_pool_connections=max_pool_connections,
                response_checksum_validation="when_required",
            ),
        )

    def _retrying(self, call: Callable[[], T]) -> T:
        """Run ``call``, retrying transient S3 failures with exponential backoff."""
        for attempt in Retrying(
            stop=stop_after_attempt(self.retries),
            wait=wait_exponential(multiplier=2, min=min(2.0, self.retry_max_wait), max=self.retry_max_wait),
            retry=retry_if_exception(is_transient),
            before_sleep=_log_retry,
            reraise=True,
        ):
            with attempt:
                return call()
        raise RuntimeError("unreachable: retry loop exited without returning")  # pragma: no cover

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/json") -> None:
        """Overwrite ``key`` with ``data``. An S3 PUT is atomic for readers."""
        self._retrying(
            lambda: self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)
        )

    def get_bytes(self, key: str) -> Optional[bytes]:
        """Return the object at ``key``, or ``None`` if it does not exist."""

        def get() -> Optional[bytes]:
            """One GET, body included — a body read cut short retries as a whole."""
            try:
                response = self._client.get_object(Bucket=self.bucket, Key=key)
                return response["Body"].read()
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                    return None
                raise

        return self._retrying(get)

    def exists(self, key: str) -> bool:
        """True if ``key`` exists, via a HEAD (no body transfer)."""

        def head() -> bool:
            """One HEAD; a 404 is the answer ``False``, not a failure."""
            try:
                self._client.head_object(Bucket=self.bucket, Key=key)
                return True
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                    return False
                raise

        return self._retrying(head)

    def upload_file(self, local_path: Path, key: str) -> None:
        """Upload a local file to ``key`` (boto3 handles multipart for big files)."""
        self._retrying(lambda: self._client.upload_file(str(local_path), self.bucket, key))
