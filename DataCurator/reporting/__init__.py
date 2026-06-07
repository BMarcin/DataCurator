"""Generic, best-effort job-progress reporting to an S3 dashboard backend.

Producers (the pipeline runner today, a training script tomorrow) push a
job-agnostic status envelope, system metrics and artifacts to S3 via
:class:`~DataCurator.reporting.reporter.JobReporter`; an external dashboard
reads them. See ``job-panel/README.md`` for the consumer-side contract.
"""
from __future__ import annotations

from DataCurator.reporting.reporter import (
    JobReporter,
    NullReporter,
    Reporter,
    build_reporter,
)
from DataCurator.reporting.schema import JobState, Link, Progress, StageInfo

__all__ = [
    "JobReporter",
    "NullReporter",
    "Reporter",
    "build_reporter",
    "JobState",
    "Link",
    "Progress",
    "StageInfo",
]
