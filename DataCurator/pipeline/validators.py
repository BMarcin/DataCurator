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

import asyncio
import difflib
import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from typing import FrozenSet, List, Optional, Sequence, Tuple

from loguru import logger

from DataCurator.LanguageTool.LanguageTool import LanguageToolChecker
from DataCurator.pipeline.errors import ServiceUnavailableError
from DataCurator.pipeline.metricx import MetricXClient
from DataCurator.pipeline.modifiers.candidates import _MISSING, _resolve_path
from DataCurator.pipeline.modifiers.languagetool import _suppress_verbatim
from DataCurator.pipeline.stage import StageContext
from DataCurator.pipeline.transfer import foreign_scripts, pattern_misses, validate_pattern_args


class ResponseValidationError(Exception):
    """A parsed LLM output failed a stage-level guard.

    Raised by a :class:`ResponseValidator` and retried like
    :class:`~DataCurator.pipeline.stages.llm.UnparseableOutputError` (a fresh
    sample may pass); a natural entry for ``runner.flag_on_errors`` so a record
    that still fails after retries is flagged rather than aborting the run.

    ``feedback`` is an LLM-facing description of what is wrong, suitable for a
    correction turn when the stage reprompts (``reprompt_attempts > 0``).
    ``None`` means the failure is not the model's fault (unreachable checker,
    missing reference field): such a failure is only ever blind-retried.
    """

    def __init__(self, *args: object, feedback: Optional[str] = None) -> None:
        super().__init__(*args)
        self.feedback = feedback


class ResponseValidator(ABC):
    """Inspect a stage's parsed output and raise on an implausible result.

    Subclasses implement :meth:`validate`, reading the parsed ``output`` (the
    response model dumped to a dict) and the shared ``context`` (which carries
    the reference fields the prompt was built from). Raise
    :class:`ResponseValidationError` to reject the output.

    Content failures should attach ``feedback`` (built via :meth:`_feedback`)
    so a reprompting stage can tell the model what to fix. The wording is
    overridable per validator: ``feedback_template`` is a ``str.format``
    template restricted to the placeholders the subclass declares in
    ``FEEDBACK_PLACEHOLDERS`` (an unknown placeholder fails at construction,
    i.e. at Hydra instantiation, not mid-run).
    """

    #: Placeholders a custom ``feedback_template`` may reference.
    FEEDBACK_PLACEHOLDERS: FrozenSet[str] = frozenset()

    def __init__(
        self, name: Optional[str] = None, feedback_template: Optional[str] = None
    ) -> None:
        self.name = name or type(self).__name__
        if feedback_template is not None:
            unknown = {
                field
                for _, field, _, _ in string.Formatter().parse(feedback_template)
                if field
            } - self.FEEDBACK_PLACEHOLDERS
            if unknown:
                raise ValueError(
                    f"{self.name}: feedback_template references unknown placeholder(s) "
                    f"{sorted(unknown)!r}; available: {sorted(self.FEEDBACK_PLACEHOLDERS)!r}"
                )
        self.feedback_template = feedback_template

    def _feedback(self, default_template: str, **values: object) -> str:
        """Render the LLM-facing correction text from this failure's diagnostics."""
        return (self.feedback_template or default_template).format(**values)

    @abstractmethod
    async def validate(self, output: Mapping, context: StageContext) -> None:
        """Validate ``output`` against ``context``; raise on failure.

        Async so a validator may call an external service (LanguageTool,
        MetricX) without blocking the event loop; the built-in reference
        checks await nothing.
        """

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
    reference. ``skip_empty_fields`` skips an *output* field whose value is
    empty — for contracts where '' is a legitimate sentinel (the gate's
    ``improved_translation`` means "no fix"), not a truncation.
    """

    FEEDBACK_PLACEHOLDERS = frozenset(
        {"field", "length", "ref_length", "ratio", "min_ratio", "max_ratio", "unit"}
    )

    def __init__(
        self,
        fields: Sequence[str],
        reference: str,
        *,
        min_ratio: Optional[float] = None,
        max_ratio: Optional[float] = None,
        unit: str = "char",
        required: bool = False,
        skip_empty_fields: bool = False,
        name: Optional[str] = None,
        feedback_template: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, feedback_template=feedback_template)
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
        self.skip_empty_fields = skip_empty_fields

    def _measure(self, value: object) -> int:
        """Length of ``value`` in the configured unit."""
        text = str(value)
        return len(text.split()) if self.unit == "word" else len(text)

    async def validate(self, output: Mapping, context: StageContext) -> None:
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
            if self.skip_empty_fields and self._measure(value) == 0:
                logger.debug(f"{self.name}: {field!r} is empty; skipping length check")
                continue
            length = self._measure(value)
            ratio = length / ref_len
            feedback_values = dict(
                field=field,
                length=length,
                ref_length=ref_len,
                ratio=f"{ratio:.2f}",
                min_ratio=self.min_ratio,
                max_ratio=self.max_ratio,
                unit=self.unit,
            )
            if self.min_ratio is not None and ratio < self.min_ratio:
                raise ResponseValidationError(
                    f"{self.name}: {field!r} is {length} {self.unit}(s), "
                    f"ratio {ratio:.2f} < min {self.min_ratio} of reference "
                    f"{self.reference!r} ({ref_len} {self.unit}(s)) — likely truncated",
                    feedback=self._feedback(
                        "your {field} is {length} {unit}(s) long but the text it must "
                        "mirror is {ref_length} {unit}(s) (ratio {ratio}, allowed minimum "
                        "{min_ratio}) — it looks truncated or incomplete. Produce the "
                        "complete text at a comparable length.",
                        **feedback_values,
                    ),
                )
            if self.max_ratio is not None and ratio > self.max_ratio:
                raise ResponseValidationError(
                    f"{self.name}: {field!r} is {length} {self.unit}(s), "
                    f"ratio {ratio:.2f} > max {self.max_ratio} of reference "
                    f"{self.reference!r} ({ref_len} {self.unit}(s))",
                    feedback=self._feedback(
                        "your {field} is {length} {unit}(s) long but the text it must "
                        "mirror is only {ref_length} {unit}(s) (ratio {ratio}, allowed "
                        "maximum {max_ratio}) — it is too long. Tighten it without "
                        "dropping content.",
                        **feedback_values,
                    ),
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
    accept dotted paths, a reference missing from the context skips the
    check (set ``required=True`` to raise instead), and ``skip_empty_fields``
    skips an output field whose value is empty (the gate's ``""`` sentinel).
    """

    FEEDBACK_PLACEHOLDERS = frozenset({"field", "mismatches"})

    def __init__(
        self,
        fields: Sequence[str],
        reference: str,
        characters: Sequence[str],
        *,
        max_diff: int = 0,
        required: bool = False,
        skip_empty_fields: bool = False,
        name: Optional[str] = None,
        feedback_template: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, feedback_template=feedback_template)
        if not characters:
            raise ValueError("CharacterCountValidator needs at least one character to count")
        if max_diff < 0:
            raise ValueError(f"max_diff must be >= 0, got {max_diff}")
        self.fields = list(fields)
        self.reference = reference
        self.characters = [str(char) for char in characters]
        self.max_diff = max_diff
        self.required = required
        self.skip_empty_fields = skip_empty_fields

    async def validate(self, output: Mapping, context: StageContext) -> None:
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
            if self.skip_empty_fields and not text:
                logger.debug(f"{self.name}: {field!r} is empty; skipping count check")
                continue
            mismatched = [
                (char, text.count(char), ref_text.count(char))
                for char in self.characters
                if abs(text.count(char) - ref_text.count(char)) > self.max_diff
            ]
            if mismatched:
                mismatches = "; ".join(
                    f"{char!r} appears {got}x vs {want}x in reference"
                    for char, got, want in mismatched
                )
                raise ResponseValidationError(
                    f"{self.name}: {field!r} broke translation-invariant character counts "
                    f"against {self.reference!r}: " + mismatches,
                    feedback=self._feedback(
                        "these characters must appear in your {field} exactly as often "
                        "as in the source text — {mismatches}. Adjust the text so the "
                        "counts match the source.",
                        field=field,
                        mismatches="; ".join(
                            f"{char!r} appears {got}x in your text but {want}x in the source"
                            for char, got, want in mismatched
                        ),
                    ),
                )


class ScriptConsistencyValidator(ResponseValidator):
    """Reject an output that uses a script its reference does not.

    A guarded script (Cyrillic, Han, Katakana, fullwidth CJK punctuation, …;
    see :data:`~DataCurator.pipeline.transfer.GUARDED_SCRIPT_RANGES`)
    appearing in an output field without a counterpart in the ``reference``
    context field is a leakage artifact — qwen occasionally drops CJK
    characters or ，-style punctuation into a Polish rewrite. Latin, digits,
    punctuation, emoji and unlisted scripts never count, and a script the
    reference itself uses is always allowed (Cyrillic quoted in the English
    source may echo in the Polish output); ``allow_scripts`` whitelists more.

    ``fields``/``reference`` semantics (dotted paths, missing-reference skip
    unless ``required``, ``skip_empty_fields`` sentinel skip) match
    :class:`LengthRatioValidator`.
    """

    FEEDBACK_PLACEHOLDERS = frozenset({"field", "leaks"})

    def __init__(
        self,
        fields: Sequence[str],
        reference: str,
        *,
        allow_scripts: Sequence[str] = (),
        required: bool = False,
        skip_empty_fields: bool = False,
        name: Optional[str] = None,
        feedback_template: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, feedback_template=feedback_template)
        self.fields = list(fields)
        self.reference = reference
        self.allow_scripts = tuple(allow_scripts)
        self.required = required
        self.skip_empty_fields = skip_empty_fields

    async def validate(self, output: Mapping, context: StageContext) -> None:
        ref_value = _resolve_path(context, self.reference)
        if ref_value is _MISSING:
            if self.required:
                raise ResponseValidationError(
                    f"{self.name}: reference field {self.reference!r} missing from context"
                )
            logger.debug(f"{self.name}: reference {self.reference!r} missing; skipping script check")
            return
        ref_text = str(ref_value)

        for field in self.fields:
            value = _resolve_path(output, field)
            if value is _MISSING:
                raise ResponseValidationError(
                    f"{self.name}: output field {field!r} missing from parsed result"
                )
            text = str(value)
            if self.skip_empty_fields and not text:
                logger.debug(f"{self.name}: {field!r} is empty; skipping script check")
                continue
            leaks = foreign_scripts(ref_text, text, allow=self.allow_scripts)
            if leaks:
                details = "; ".join(
                    f"{script}: {sample!r}" for script, sample in sorted(leaks.items())
                )
                raise ResponseValidationError(
                    f"{self.name}: {field!r} contains scripts absent from "
                    f"{self.reference!r} — {details}",
                    feedback=self._feedback(
                        "your {field} contains characters from writing systems that do "
                        "not appear in the source text: {leaks}. Remove them or replace "
                        "them with a correct ones.",
                        field=field,
                        leaks="; ".join(
                            f"{script} ({sample!r})" for script, sample in sorted(leaks.items())
                        ),
                    ),
                )


class PatternPreservationValidator(ResponseValidator):
    """Require numbers, URLs and emails of the reference to survive translation.

    Uses :func:`~DataCurator.pipeline.transfer.pattern_misses`: URLs and
    emails must appear verbatim in each output field (a miss always fails,
    regardless of ``max_diff``); numbers are compared as multisets of
    canonical digit strings — grouping/decimal restyling (``1,000.5`` →
    ``1000,5``) matches, English ordinal/decade tokens (``20th``, ``1980s``)
    are exempt — and fail only when the miss count exceeds ``max_diff``
    (spelled-out↔digit conversions shift the odd number legitimately).
    ``mode="both"`` also counts numbers the output invented.

    ``fields``/``reference`` semantics (dotted paths, missing-reference skip
    unless ``required``, ``skip_empty_fields`` sentinel skip) match
    :class:`LengthRatioValidator`.
    """

    FEEDBACK_PLACEHOLDERS = frozenset({"field", "missing", "extra"})

    def __init__(
        self,
        fields: Sequence[str],
        reference: str,
        *,
        kinds: Sequence[str] = ("number", "url", "email"),
        mode: str = "missing",
        max_diff: int = 0,
        required: bool = False,
        skip_empty_fields: bool = False,
        name: Optional[str] = None,
        feedback_template: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, feedback_template=feedback_template)
        validate_pattern_args(kinds, mode)
        if max_diff < 0:
            raise ValueError(f"max_diff must be >= 0, got {max_diff}")
        self.fields = list(fields)
        self.reference = reference
        self.kinds = tuple(kinds)
        self.mode = mode
        self.max_diff = max_diff
        self.required = required
        self.skip_empty_fields = skip_empty_fields

    async def validate(self, output: Mapping, context: StageContext) -> None:
        ref_value = _resolve_path(context, self.reference)
        if ref_value is _MISSING:
            if self.required:
                raise ResponseValidationError(
                    f"{self.name}: reference field {self.reference!r} missing from context"
                )
            logger.debug(f"{self.name}: reference {self.reference!r} missing; skipping pattern check")
            return
        ref_text = str(ref_value)

        for field in self.fields:
            value = _resolve_path(output, field)
            if value is _MISSING:
                raise ResponseValidationError(
                    f"{self.name}: output field {field!r} missing from parsed result"
                )
            text = str(value)
            if self.skip_empty_fields and not text:
                logger.debug(f"{self.name}: {field!r} is empty; skipping pattern check")
                continue
            misses = pattern_misses(ref_text, text, kinds=self.kinds, mode=self.mode)
            hard = [kind for kind in ("url", "email") if misses.get(kind)]
            number_misses = len(misses.get("number", [])) + len(misses.get("extra_number", []))
            if hard or number_misses > self.max_diff:
                details = "; ".join(f"{kind}: {tokens}" for kind, tokens in sorted(misses.items()))
                lost = "; ".join(
                    f"{kind} {tokens}"
                    for kind, tokens in sorted(misses.items())
                    if kind != "extra_number" and tokens
                )
                invented = ", ".join(str(t) for t in misses.get("extra_number", []))
                problems = []
                if lost:
                    problems.append("missing from your text: {missing}")
                if invented:
                    problems.append("numbers you added that are not in the source: {extra}")
                raise ResponseValidationError(
                    f"{self.name}: {field!r} lost patterns present in "
                    f"{self.reference!r} — {details}",
                    feedback=self._feedback(
                        "your {field} must carry over the source's numbers, URLs and "
                        "emails verbatim — " + "; ".join(problems) + ". Reinsert missing "
                        "tokens exactly as they appear in the source and remove invented "
                        "ones.",
                        field=field,
                        missing=lost,
                        extra=invented,
                    ),
                )


def _truncate(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


_WORD_RE = re.compile(r"\w+")


def _words(text: str) -> List[str]:
    """Casefolded word tokens of ``text``."""
    return _WORD_RE.findall(text.casefold())


def dice_coefficient(text: str, reference: str, stem_prefix: int = 5) -> float:
    """Sørensen–Dice over word-token multisets, each token cut to ``stem_prefix`` chars.

    Order-insensitive, so a reordered sentence keeps its score; the prefix
    cut makes inflected forms of one word ("Juliana"/"Julian") count as a
    match. Two empty texts score 1.0.
    """
    a = Counter(word[:stem_prefix] for word in _words(text))
    b = Counter(word[:stem_prefix] for word in _words(reference))
    total = sum(a.values()) + sum(b.values())
    return 2 * sum((a & b).values()) / total if total else 1.0


def repeated_ngram(
    text: str, reference: Optional[str] = None, n: int = 3
) -> Tuple[Optional[Tuple[str, ...]], int, int]:
    """The word n-gram ``text`` repeats most beyond ``reference``.

    Returns ``(ngram, count_in_text, excess)`` where ``excess`` is the count
    in ``text`` minus the count in ``reference`` (0 occurrences when no
    reference), maximised over the n-grams of ``text``; ``(None, 0, 0)``
    when ``text`` has fewer than ``n`` words. Repetition the reference
    already carries (lists, refrains) is thereby exempt, while a generation
    loop shows up as a large excess.
    """
    words = _words(text)
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    ref_words = _words(reference) if reference else []
    ref_grams = Counter(tuple(ref_words[i : i + n]) for i in range(len(ref_words) - n + 1))
    best: Tuple[Optional[Tuple[str, ...]], int, int] = (None, 0, 0)
    for gram, count in grams.items():
        excess = count - ref_grams.get(gram, 0)
        if excess > best[2]:
            best = (gram, count, excess)
    return best


class RepetitionValidator(ResponseValidator):
    """Reject outputs that repeat a word n-gram far more than the reference does.

    Catches generation loops that still close their JSON within the token
    limit — a phrase emitted over and over passes every length and
    similarity floor when the reference is long. The excess (see
    :func:`repeated_ngram`) is measured against ``reference`` (a same-language
    field such as the draft, so legitimately repeated phrases are exempt);
    without a reference, raw counts are used. The default ``max_excess=5`` on
    trigrams was calibrated on 65k accepted fix rewrites, whose largest
    excess was 5.

    A missing reference field is treated as no reference unless
    ``required=True``; ``skip_empty_fields`` skips an empty output field.
    """

    FEEDBACK_PLACEHOLDERS = frozenset({"field", "phrase", "count", "ref_count", "max_excess"})

    def __init__(
        self,
        fields: Sequence[str],
        reference: Optional[str] = None,
        *,
        ngram: int = 3,
        max_excess: int = 5,
        required: bool = False,
        skip_empty_fields: bool = False,
        name: Optional[str] = None,
        feedback_template: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, feedback_template=feedback_template)
        if ngram < 1:
            raise ValueError(f"ngram must be >= 1, got {ngram}")
        if max_excess < 0:
            raise ValueError(f"max_excess must be >= 0, got {max_excess}")
        self.fields = list(fields)
        self.reference = reference
        self.ngram = ngram
        self.max_excess = max_excess
        self.required = required
        self.skip_empty_fields = skip_empty_fields

    async def validate(self, output: Mapping, context: StageContext) -> None:
        ref_text: Optional[str] = None
        if self.reference is not None:
            ref_value = _resolve_path(context, self.reference)
            if ref_value is _MISSING:
                if self.required:
                    raise ResponseValidationError(
                        f"{self.name}: reference field {self.reference!r} missing from context"
                    )
            else:
                ref_text = str(ref_value or "")
        for field in self.fields:
            value = _resolve_path(output, field)
            if value is _MISSING:
                raise ResponseValidationError(
                    f"{self.name}: output field {field!r} missing from parsed result"
                )
            text = str(value)
            if self.skip_empty_fields and not text:
                continue
            gram, count, excess = repeated_ngram(text, ref_text, self.ngram)
            if gram is None or excess <= self.max_excess:
                continue
            phrase = " ".join(gram)
            raise ResponseValidationError(
                f"{self.name}: {field!r} repeats {phrase!r} {count}x "
                f"({count - excess}x in reference), excess {excess} > max {self.max_excess}",
                feedback=self._feedback(
                    "your {field} repeats the phrase \"{phrase}\" {count} times (the draft "
                    "has it {ref_count} times) — this looks like a generation loop. "
                    "Write each part of the text exactly once.",
                    field=field,
                    phrase=phrase,
                    count=count,
                    ref_count=count - excess,
                    max_excess=self.max_excess,
                ),
            )


class SimilarityValidator(ResponseValidator):
    """Require output fields to stay textually close to a reference field.

    Each field in ``fields`` is compared to the ``reference`` field read from
    the context via a normalized edit-distance similarity
    (:meth:`difflib.SequenceMatcher.ratio`, 0..1); a ratio below
    ``min_similarity`` raises :class:`ResponseValidationError`. The motivating
    case is batch cross-contamination: a batched stage that mismatches outputs
    to inputs hands a record a batchmate's translation, which is wildly
    dissimilar from the record's own draft (a swap measures ~0.2 where a
    genuine light edit measures ~0.9), while length and character counts can
    still look plausible.

    The reference must be in the same language as the output — compare a
    rewrite against the draft it edits (e.g. ``target_text``), never against
    the untranslated source. Texts are normalized before comparison:
    ``normalize_case`` casefolds, ``collapse_whitespace`` joins all runs of
    whitespace to single spaces.

    The edit-distance ratio punishes word reordering, which short Polish
    rewrites do legitimately ("Gdzie nie udało się zadomowić Juliana
    Fontany?" vs "Gdzie Julian Fontana nie zdołał się ustabilizować?").
    Setting ``min_dice`` adds an order-insensitive second opinion — the
    Sørensen–Dice coefficient over word tokens cut to their first
    ``stem_prefix`` characters (a crude stemmer that tolerates inflection) —
    and a field then fails only when **both** scores are under their minimum.

    As in :class:`LengthRatioValidator`, ``fields`` and ``reference`` accept
    dotted paths, a missing/empty reference skips the check unless
    ``required=True``, and ``skip_empty_fields`` skips an output field whose
    value is empty (the gate's ``""`` "no fix" sentinel).
    """

    FEEDBACK_PLACEHOLDERS = frozenset({"field", "ratio", "min_similarity", "dice", "min_dice"})

    def __init__(
        self,
        fields: Sequence[str],
        reference: str,
        *,
        min_similarity: float,
        min_dice: Optional[float] = None,
        stem_prefix: int = 5,
        normalize_case: bool = True,
        collapse_whitespace: bool = True,
        required: bool = False,
        skip_empty_fields: bool = False,
        name: Optional[str] = None,
        feedback_template: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, feedback_template=feedback_template)
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError(f"min_similarity must be within [0, 1], got {min_similarity}")
        if min_dice is not None and not 0.0 <= min_dice <= 1.0:
            raise ValueError(f"min_dice must be within [0, 1], got {min_dice}")
        if stem_prefix < 1:
            raise ValueError(f"stem_prefix must be >= 1, got {stem_prefix}")
        self.fields = list(fields)
        self.reference = reference
        self.min_similarity = min_similarity
        self.min_dice = min_dice
        self.stem_prefix = stem_prefix
        self.normalize_case = normalize_case
        self.collapse_whitespace = collapse_whitespace
        self.required = required
        self.skip_empty_fields = skip_empty_fields

    def _normalize(self, value: object) -> str:
        text = str(value)
        if self.normalize_case:
            text = text.casefold()
        if self.collapse_whitespace:
            text = " ".join(text.split())
        return text

    async def validate(self, output: Mapping, context: StageContext) -> None:
        ref_value = _resolve_path(context, self.reference)
        if ref_value is _MISSING:
            if self.required:
                raise ResponseValidationError(
                    f"{self.name}: reference field {self.reference!r} missing from context"
                )
            logger.debug(f"{self.name}: reference {self.reference!r} missing; skipping similarity check")
            return
        ref_text = self._normalize(ref_value)
        if not ref_text:
            logger.debug(f"{self.name}: reference {self.reference!r} is empty; skipping similarity check")
            return

        for field in self.fields:
            value = _resolve_path(output, field)
            if value is _MISSING:
                raise ResponseValidationError(
                    f"{self.name}: output field {field!r} missing from parsed result"
                )
            text = self._normalize(value)
            if self.skip_empty_fields and not text:
                logger.debug(f"{self.name}: {field!r} is empty; skipping similarity check")
                continue
            ratio = difflib.SequenceMatcher(None, text, ref_text, autojunk=False).ratio()
            if ratio >= self.min_similarity:
                continue
            dice = None
            if self.min_dice is not None:
                dice = dice_coefficient(text, ref_text, self.stem_prefix)
                if dice >= self.min_dice:
                    logger.debug(
                        f"{self.name}: {field!r} similarity {ratio:.2f} < {self.min_similarity} "
                        f"but Dice {dice:.2f} >= {self.min_dice}; accepting"
                    )
                    continue
            dice_note = "" if dice is None else f" and Dice {dice:.2f} < min {self.min_dice}"
            raise ResponseValidationError(
                f"{self.name}: {field!r} similarity {ratio:.2f} < min {self.min_similarity}"
                f"{dice_note} against reference {self.reference!r} — output likely belongs "
                f"to another record ({_truncate(text)!r} vs {_truncate(ref_text)!r})",
                feedback=self._feedback(
                    "your {field} is too different from the draft it should lightly "
                    "edit (similarity {ratio}, required at least {min_similarity}) — "
                    "it may answer a different input entirely. Stay close to the "
                    "given draft and change only what needs fixing.",
                    field=field,
                    ratio=f"{ratio:.2f}",
                    min_similarity=self.min_similarity,
                    dice="n/a" if dice is None else f"{dice:.2f}",
                    min_dice=self.min_dice,
                ),
            )


class LanguageToolValidator(ResponseValidator):
    """Reject rewrites in which LanguageTool finds an allow-listed issue.

    Detection only: the validator calls
    :meth:`~DataCurator.LanguageTool.LanguageTool.LanguageToolChecker.get_issues`
    — never the fixing methods — so LT can veto a rewrite (the LLM resamples)
    but a wrong LT suggestion can never enter the dataset. Run it with a
    **narrow allow-list** (Polish spellcheck + word repetition, probed via
    ``scripts/probe_languagetool_rules.py``), not the full inventory: its job
    is catching invented non-words ("nowog", "którejktórego") and doubled
    words in fix rewrites, which today nothing checks.

    False-positive filters, applied before rejection:

    - ``suppress_source_field`` drops findings whose span appears verbatim
      (case-insensitively) in that context field. The reference is a
      *different language* than the validated fields, so the only spans that
      can match are terms copied through verbatim — which the prompts mandate.
    - ``skip_capitalized`` drops capitalized unknowns (proper nouns).

    An unreachable LT server raises
    :class:`~DataCurator.pipeline.errors.ServiceUnavailableError` rather than
    silently disabling the guard: a record must never pass a check that did
    not run. The checker's non-connection fallback issue ("Błąd
    przetwarzania…", no ``rule_id``/``error_text``) survives both filters, so
    a persistent LT processing failure still fails validation — same
    rationale, but as a content failure, since the server did answer.
    """

    FEEDBACK_PLACEHOLDERS = frozenset({"field", "findings"})

    def __init__(
        self,
        fields: Sequence[str],
        *,
        language: str = "pl-PL",
        allowed_fixes: Optional[Sequence[str]] = None,
        denied_fixes: Optional[Sequence[str]] = None,
        suppress_source_field: Optional[str] = None,
        skip_capitalized: bool = True,
        remote_server: Optional[str] = None,
        retries: int = 5,
        skip_empty_fields: bool = True,
        name: Optional[str] = None,
        feedback_template: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, feedback_template=feedback_template)
        self.fields = list(fields)
        self.suppress_source_field = suppress_source_field
        self.skip_capitalized = skip_capitalized
        self.skip_empty_fields = skip_empty_fields
        self._checker_kwargs = dict(
            language=language,
            remote_server=remote_server,
            allowed_fixes=list(allowed_fixes) if allowed_fixes is not None else None,
            denied_fixes=list(denied_fixes) if denied_fixes is not None else None,
            retries=retries,
        )
        self._checker: Optional[LanguageToolChecker] = None
        self._lock = asyncio.Lock()

    async def _get_checker(self) -> LanguageToolChecker:
        """Lazily construct the (network-connecting) checker on first use."""
        async with self._lock:
            if self._checker is None:
                self._checker = LanguageToolChecker(**self._checker_kwargs)
        return self._checker

    def _filter(self, issues: List[dict], context: StageContext) -> List[dict]:
        if self.suppress_source_field:
            reference = _resolve_path(context, self.suppress_source_field)
            issues = _suppress_verbatim(
                issues, "" if reference is _MISSING else str(reference or ""), casefold=True
            )
        if self.skip_capitalized:
            issues = [
                issue
                for issue in issues
                if not (issue.get("error_text") and str(issue["error_text"])[:1].isupper())
            ]
        return issues

    async def validate(self, output: Mapping, context: StageContext) -> None:
        for field in self.fields:
            value = _resolve_path(output, field)
            if value is _MISSING:
                raise ResponseValidationError(
                    f"{self.name}: output field {field!r} missing from parsed result"
                )
            text = str(value)
            if self.skip_empty_fields and not text:
                logger.debug(f"{self.name}: {field!r} is empty; skipping LanguageTool check")
                continue
            try:
                # The constructor pings the server, so acquisition can raise too.
                checker = await self._get_checker()
                issues = await asyncio.to_thread(checker.get_issues, text)
            except ServiceUnavailableError:
                raise
            except ConnectionError as exc:
                # The checker signals an unreachable server with the builtin
                # ConnectionError; re-raise it as the pipeline's own marker so
                # the outage is explicit rather than inferred from the base class.
                raise ServiceUnavailableError(
                    f"{self.name}: LanguageTool unreachable: {exc}"
                ) from exc
            issues = self._filter(issues, context)
            if issues:
                findings = "; ".join(
                    f"{issue.get('rule_id') or '<no rule>'}: {_truncate(str(issue.get('error_text') or issue.get('message', '')))!r}"
                    for issue in issues
                )
                # Feedback only from model-actionable findings; the checker's
                # fallback pseudo-issue (processing failure, no rule_id or
                # error_text) is not the model's fault and must never reprompt.
                actionable = [i for i in issues if i.get("rule_id") or i.get("error_text")]
                feedback = None
                if actionable:
                    lines = []
                    for issue in actionable:
                        line = f"- {issue.get('rule_id') or 'issue'}: {issue.get('message', '')}"
                        if issue.get("error_text"):
                            line += f" | affected text: {str(issue['error_text'])!r}"
                        if issue.get("context"):
                            line += f" | context: {issue['context']}"
                        if issue.get("suggestions"):
                            line += f" | suggestions: {', '.join(str(s) for s in issue['suggestions'][:5])}"
                        lines.append(line)
                    feedback = self._feedback(
                        "a Polish grammar and spelling checker found problems in your "
                        "{field}:\n{findings}\nRewrite the affected spans so the text "
                        "is correct Polish.",
                        field=field,
                        findings="\n".join(lines),
                    )
                raise ResponseValidationError(
                    f"{self.name}: {field!r} has LanguageTool findings: {findings}",
                    feedback=feedback,
                )


class MetricXValidator(ResponseValidator):
    """Cross-lingual adequacy gate: MetricX-QE error score against the source.

    Every other guard here is meaning-blind — lengths, character counts and
    edit-distance against a *same-language* draft cannot see a fluent rewrite
    that drops a clause or mistranslates a term. MetricX-QE scores a
    hypothesis directly against the (other-language) source on a 0–25 error
    scale where **lower is better**; a field scoring above ``max_score``
    raises, so the LLM resamples and exhausted retries flag the record.

    The score comes from the hosted inference service (one HTTP round trip
    covering all ``fields``). An unreachable service does not skip the check:
    once the client's own bounded retry is spent, :class:`MetricXUnavailable`
    propagates as a connection-class outage, so the record is never judged on
    a check that did not run. ``required`` governs only a missing
    ``reference`` field, as it does on the other guards here.

    ``max_score`` is calibrated, not assumed — probe with
    ``scripts/probe_metricx_threshold.py`` (and recalibrate if the service's
    dtype changes; fp16/bf16 shift scores vs the float32 eval).
    """

    FEEDBACK_PLACEHOLDERS = frozenset({"field", "score", "max_score"})

    def __init__(
        self,
        fields: Sequence[str],
        reference: str,
        *,
        max_score: float,
        remote_server: Optional[str] = None,
        api_token: Optional[str] = None,
        required: bool = False,
        skip_empty_fields: bool = False,
        timeout: float = 180.0,
        retries: int = 3,
        name: Optional[str] = None,
        feedback_template: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, feedback_template=feedback_template)
        if not 0.0 <= max_score <= 25.0:
            raise ValueError(f"max_score must be within the 0-25 MetricX scale, got {max_score}")
        self.fields = list(fields)
        self.reference = reference
        self.max_score = max_score
        self.required = required
        self.skip_empty_fields = skip_empty_fields
        self._client = MetricXClient(
            remote_server, api_token=api_token, timeout=timeout, retries=retries
        )

    async def validate(self, output: Mapping, context: StageContext) -> None:
        ref_value = _resolve_path(context, self.reference)
        if ref_value is _MISSING:
            if self.required:
                raise ResponseValidationError(
                    f"{self.name}: reference field {self.reference!r} missing from context"
                )
            logger.debug(f"{self.name}: reference {self.reference!r} missing; skipping MetricX check")
            return
        ref_text = str(ref_value)
        if not ref_text:
            logger.debug(f"{self.name}: reference {self.reference!r} is empty; skipping MetricX check")
            return

        checked: List[Tuple[str, str]] = []  # (field, hypothesis)
        for field in self.fields:
            value = _resolve_path(output, field)
            if value is _MISSING:
                raise ResponseValidationError(
                    f"{self.name}: output field {field!r} missing from parsed result"
                )
            text = str(value)
            if self.skip_empty_fields and not text:
                logger.debug(f"{self.name}: {field!r} is empty; skipping MetricX check")
                continue
            checked.append((field, text))
        if not checked:
            return

        # MetricXUnavailable propagates untouched: it is an outage, not a
        # verdict, and wrapping it as a ResponseValidationError would both
        # flag the record as bad data and hide it from the worker's
        # connection-class accounting.
        scores = await self._client.score_pairs([(ref_text, text) for _, text in checked])
        for (field, text), score in zip(checked, scores):
            if score > self.max_score:
                raise ResponseValidationError(
                    f"{self.name}: {field!r} MetricX score {score:.2f} > max {self.max_score} "
                    f"against {self.reference!r} — inadequate translation "
                    f"({_truncate(text)!r})",
                    feedback=self._feedback(
                        "your {field} was judged an inadequate translation of the source "
                        "(quality error score {score} on a 0–25 scale where lower is "
                          "better; limit {max_score}). Retranslate faithfully, preserving "
                        "all of the source's meaning.",
                        field=field,
                        score=f"{score:.1f}",
                        max_score=self.max_score,
                    ),
                )


class EditSpanValidator(ResponseValidator):
    """Ground and bound a list of span edits over a record's segments.

    The guard of a span-edit stage (``PairAdjustment``): the model returns
    ``{segment_id, source_term, span, replacement}`` edits instead of a
    reprinted text, and this validator checks each one against the
    ``segments`` list in the context (``[{id, <source_key>, <text_key>}, ...]``)
    before :class:`~DataCurator.pipeline.modifiers.edits.ApplyEditsModifier`
    applies them:

    * ``segment_id`` names a segment, and ``span`` occurs in its
      ``text_key`` text and differs from ``replacement``;
    * ``source_term`` occurs (casefolded) in the ``source_key`` text of the
      edited segment and of at least ``min_segments_with_term`` segments in
      all — only a term the sources share may be harmonized, so an edit can
      never import wording one source does not have;
    * ``span`` and ``replacement`` are at most ``max_words`` words each, and
      there are at most ``max_edits`` edits — terminology, not rewrites.

    Every problem found is reported in one error, so a reprompt can fix them
    together. An empty edit list always passes.
    """

    FEEDBACK_PLACEHOLDERS = frozenset({"problems"})

    def __init__(
        self,
        edits: str = "edits",
        segments: str = "segments",
        *,
        text_key: str = "translation",
        source_key: str = "text",
        min_segments_with_term: int = 2,
        max_words: int = 6,
        max_edits: int = 6,
        name: Optional[str] = None,
        feedback_template: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, feedback_template=feedback_template)
        if min_segments_with_term < 1:
            raise ValueError(f"min_segments_with_term must be >= 1, got {min_segments_with_term}")
        self.edits = edits
        self.segments = segments
        self.text_key = text_key
        self.source_key = source_key
        self.min_segments_with_term = min_segments_with_term
        self.max_words = max_words
        self.max_edits = max_edits

    def _problems(self, edits: Sequence[Mapping], segments: Mapping[str, Mapping]) -> List[str]:
        """Describe, LLM-facing, every edit that breaks a rule."""
        problems: List[str] = []
        if len(edits) > self.max_edits:
            problems.append(
                f"you returned {len(edits)} edits; at most {self.max_edits} are allowed — "
                "keep only real terminology mismatches"
            )
        for edit in edits:
            segment_id = str(edit.get("segment_id", ""))
            span = str(edit.get("span", ""))
            replacement = str(edit.get("replacement", ""))
            term = str(edit.get("source_term", "")).casefold().strip()
            label = f"edit {span!r} -> {replacement!r} in [{segment_id}]"
            segment = segments.get(segment_id)
            if segment is None:
                problems.append(f"{label}: there is no segment [{segment_id}]; edit only {sorted(segments)}")
                continue
            if not span or span not in str(segment.get(self.text_key) or ""):
                problems.append(f"{label}: the span does not occur verbatim in [{segment_id}]'s translation")
            if span == replacement:
                problems.append(f"{label}: the replacement equals the span")
            for which, text in (("span", span), ("replacement", replacement)):
                if len(text.split()) > self.max_words:
                    problems.append(f"{label}: the {which} exceeds {self.max_words} words — this is a rewrite, not a term")
            sharing = [
                sid for sid, seg in segments.items()
                if term and term in str(seg.get(self.source_key) or "").casefold()
            ]
            if segment_id not in sharing or len(sharing) < self.min_segments_with_term:
                problems.append(
                    f"{label}: source_term {edit.get('source_term')!r} must appear verbatim in the "
                    f"English of [{segment_id}] and of at least {self.min_segments_with_term} "
                    f"segments in all (found in {sharing or 'none'}) — only a term the sources "
                    "share may be harmonized"
                )
        return problems

    async def validate(self, output: Mapping, context: StageContext) -> None:
        edits = _resolve_path(output, self.edits)
        if edits is _MISSING:
            raise ResponseValidationError(f"{self.name}: output field {self.edits!r} missing from parsed result")
        if not edits:
            return
        segments = _resolve_path(context, self.segments)
        if segments is _MISSING or not segments:
            raise ResponseValidationError(f"{self.name}: segments {self.segments!r} missing from context")
        problems = self._problems(edits, {str(s["id"]): s for s in segments})
        if problems:
            raise ResponseValidationError(
                f"{self.name}: {len(problems)} invalid edit(s): " + "; ".join(problems),
                feedback=self._feedback(
                    "some of your edits break the rules: {problems}. Drop or correct those "
                    "edits; an empty edit list is a valid answer.",
                    problems="; ".join(problems),
                ),
            )
