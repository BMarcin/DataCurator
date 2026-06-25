"""Stage-level guards over an LLM stage's parsed output.

A :class:`ResponseValidator` inspects the structured output an
:class:`~DataCurator.pipeline.stages.llm.LLMStage` just parsed, with the full
:class:`~DataCurator.pipeline.stage.StageContext` in scope, and raises
:class:`ResponseValidationError` when the output is implausible. The stage runs
its validators *inside* the retry loop, so a failed guard triggers a fresh
sample; once retries are exhausted the error re-raises and the runner flags the
record (list ``ResponseValidationError`` in ``runner.flag_on_errors``).

The motivating case is a truncated rewrite: the model occasionally emits text
that breaks off mid-sentence even though the JSON parses cleanly
(``finish_reason: stop``, so the OpenAI SDK does not raise
``LengthFinishReasonError``). Such a rewrite is dramatically shorter than the
reference translation it should mirror, so :class:`LengthRatioValidator`
catches it by bounding the length ratio against a reference field in the
context (e.g. the Google-Translate draft). A second guard,
:class:`CharacterCountValidator`, targets escape artifacts — a model that
prefixes every word with ``\\`` still parses cleanly, but the backslash count
no longer matches the source it translates.

Validators are config-driven, mirroring stage modifiers: declare them under a
stage's ``validators`` list with a ``_target_`` and Hydra instantiates them.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Optional, Sequence

from loguru import logger

from DataCurator.pipeline.modifiers.candidates import _MISSING, _resolve_path
from DataCurator.pipeline.stage import StageContext


class ResponseValidationError(Exception):
    """A parsed LLM output failed a stage-level guard.

    Raised by a :class:`ResponseValidator` and retried like
    :class:`~DataCurator.pipeline.stages.llm.UnparseableOutputError` (a fresh
    sample may pass); a natural entry for ``runner.flag_on_errors`` so a record
    that still fails after retries is flagged rather than aborting the run.
    """


class ResponseValidator(ABC):
    """Inspect a stage's parsed output and raise on an implausible result.

    Subclasses implement :meth:`validate`, reading the parsed ``output`` (the
    response model dumped to a dict) and the shared ``context`` (which carries
    the reference fields the prompt was built from). Raise
    :class:`ResponseValidationError` to reject the output.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or type(self).__name__

    @abstractmethod
    def validate(self, output: Mapping, context: StageContext) -> None:
        """Validate ``output`` against ``context``; raise on failure."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class LengthRatioValidator(ResponseValidator):
    """Bound the length of output fields relative to a reference field.

    Each field in ``fields`` is compared, in length, to the ``reference`` field
    read from the context: ``ratio = len(field) / len(reference)``. A ratio
    outside ``[min_ratio, max_ratio]`` (either bound optional) raises
    :class:`ResponseValidationError`. ``unit`` selects characters (default) or
    whitespace-delimited words. Both ``fields`` and ``reference`` accept dotted
    paths into nested mappings.

    When the reference is missing from the context or empty the ratio is
    undefined, so the check is skipped (it never manufactures a failure from a
    missing reference); set ``required=True`` to instead raise on an absent
    reference.
    """

    def __init__(
        self,
        fields: Sequence[str],
        reference: str,
        *,
        min_ratio: Optional[float] = None,
        max_ratio: Optional[float] = None,
        unit: str = "char",
        required: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        if unit not in ("char", "word"):
            raise ValueError(f"unit must be 'char' or 'word', got {unit!r}")
        if min_ratio is None and max_ratio is None:
            raise ValueError("LengthRatioValidator needs at least one of min_ratio/max_ratio")
        self.fields = list(fields)
        self.reference = reference
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.unit = unit
        self.required = required

    def _measure(self, value: object) -> int:
        """Length of ``value`` in the configured unit."""
        text = str(value)
        return len(text.split()) if self.unit == "word" else len(text)

    def validate(self, output: Mapping, context: StageContext) -> None:
        ref_value = _resolve_path(context, self.reference)
        if ref_value is _MISSING:
            if self.required:
                raise ResponseValidationError(
                    f"{self.name}: reference field {self.reference!r} missing from context"
                )
            logger.debug(f"{self.name}: reference {self.reference!r} missing; skipping length check")
            return
        ref_len = self._measure(ref_value)
        if ref_len == 0:
            logger.debug(f"{self.name}: reference {self.reference!r} is empty; skipping length check")
            return

        for field in self.fields:
            value = _resolve_path(output, field)
            if value is _MISSING:
                raise ResponseValidationError(
                    f"{self.name}: output field {field!r} missing from parsed result"
                )
            ratio = self._measure(value) / ref_len
            if self.min_ratio is not None and ratio < self.min_ratio:
                raise ResponseValidationError(
                    f"{self.name}: {field!r} is {self._measure(value)} {self.unit}(s), "
                    f"ratio {ratio:.2f} < min {self.min_ratio} of reference "
                    f"{self.reference!r} ({ref_len} {self.unit}(s)) — likely truncated"
                )
            if self.max_ratio is not None and ratio > self.max_ratio:
                raise ResponseValidationError(
                    f"{self.name}: {field!r} is {self._measure(value)} {self.unit}(s), "
                    f"ratio {ratio:.2f} > max {self.max_ratio} of reference "
                    f"{self.reference!r} ({ref_len} {self.unit}(s))"
                )


class CharacterCountValidator(ResponseValidator):
    """Require translation-invariant characters to keep their reference count.

    Characters like ``\\``, ``!`` or ``?`` pass through a translation
    unchanged, so each output field should contain them exactly as often as
    the ``reference`` field read from the context. Every entry in
    ``characters`` is counted with ``str.count`` (multi-character substrings
    such as ``"\\n\\n"`` work too); a count differing from the reference by
    more than ``max_diff`` (default 0 — exact match) raises
    :class:`ResponseValidationError`. The motivating case is a model that
    escapes its own output, prefixing every word with ``\\``: the JSON parses
    cleanly, but the rewrite is flooded with backslashes absent from the
    source.

    As in :class:`LengthRatioValidator`, both ``fields`` and ``reference``
    accept dotted paths, and a reference missing from the context skips the
    check (set ``required=True`` to raise instead).
    """

    def __init__(
        self,
        fields: Sequence[str],
        reference: str,
        characters: Sequence[str],
        *,
        max_diff: int = 0,
        required: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        if not characters:
            raise ValueError("CharacterCountValidator needs at least one character to count")
        if max_diff < 0:
            raise ValueError(f"max_diff must be >= 0, got {max_diff}")
        self.fields = list(fields)
        self.reference = reference
        self.characters = [str(char) for char in characters]
        self.max_diff = max_diff
        self.required = required

    def validate(self, output: Mapping, context: StageContext) -> None:
        ref_value = _resolve_path(context, self.reference)
        if ref_value is _MISSING:
            if self.required:
                raise ResponseValidationError(
                    f"{self.name}: reference field {self.reference!r} missing from context"
                )
            logger.debug(f"{self.name}: reference {self.reference!r} missing; skipping count check")
            return
        ref_text = str(ref_value)

        for field in self.fields:
            value = _resolve_path(output, field)
            if value is _MISSING:
                raise ResponseValidationError(
                    f"{self.name}: output field {field!r} missing from parsed result"
                )
            text = str(value)
            mismatches = [
                f"{char!r} appears {text.count(char)}x vs {ref_text.count(char)}x in reference"
                for char in self.characters
                if abs(text.count(char) - ref_text.count(char)) > self.max_diff
            ]
            if mismatches:
                raise ResponseValidationError(
                    f"{self.name}: {field!r} broke translation-invariant character counts "
                    f"against {self.reference!r}: " + "; ".join(mismatches)
                )
