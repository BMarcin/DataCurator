"""Thin S3-compatible object sink used by the job reporter.

Wraps a :mod:`boto3` client configured for an arbitrary endpoint (Garage,
MinIO, AWS …). It exposes only the few operations the reporter needs —
put bytes, get bytes, upload a (possibly large) file — and leaves error
handling to the caller, which treats every reporting action as best-effort.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


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
    ) -> None:
        """Build a boto3 S3 client for ``bucket`` at ``endpoint_url``.

        Path-style addressing is forced because self-hosted S3 servers
        (Garage, MinIO) rarely support virtual-host buckets.
        """
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        )

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/json") -> None:
        """Overwrite ``key`` with ``data``. An S3 PUT is atomic for readers."""
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get_bytes(self, key: str) -> Optional[bytes]:
        """Return the object at ``key``, or ``None`` if it does not exist."""
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def exists(self, key: str) -> bool:
        """True if ``key`` exists, via a HEAD (no body transfer)."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
                return False
            raise

    def upload_file(self, local_path: Path, key: str) -> None:
        """Upload a local file to ``key`` (boto3 handles multipart for big files)."""
        self._client.upload_file(str(local_path), self.bucket, key)
