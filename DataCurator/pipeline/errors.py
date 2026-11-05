"""Shared exception types for the pipeline's guards.

Lives in its own module so the low-level service clients (which
:mod:`DataCurator.pipeline.validators` imports) can raise the same marker the
validators do, without the reverse import cycle.
"""
from __future__ import annotations


class ServiceUnavailableError(ConnectionError):
    """A guard's backing service could not be reached or refused to answer.

    Raised by a validator or modifier whose check *could not run* — the
    LanguageTool or MetricX server is down, unbound, or rejecting requests.
    That is an infrastructure failure, never a verdict on the record, so it
    must not be confused with :class:`~DataCurator.pipeline.validators.\
ResponseValidationError` (the model's output is bad).

    Two consequences ride on the base class, both deliberate:

    * The distributed worker's ``CONNECTION_ERRORS`` includes the builtin
      ``ConnectionError``, so a chunk that fails this way is counted as
      *connection-class* and nak'd for redelivery instead of being uploaded
      with every record flagged as bad data.
    * ``LLMStage._blind_retryable`` refuses to retry it: the service client
      already spent its own bounded retry budget, so re-running the model
      would burn a generation on an outage that has nothing to do with the
      prompt.
    """
