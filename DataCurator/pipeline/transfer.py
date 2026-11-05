"""Deterministic transfer checks between a source text and its translation.

Pure functions over a ``(source, hypothesis)`` pair — no model, no service:
translation-invariant properties that any faithful EN→PL translation must
keep. :func:`foreign_scripts` finds script leakage (a Cyrillic or CJK
character in the Polish output with no counterpart in the English source);
:func:`pattern_misses` finds dropped numbers, URLs and emails.

Shared by the response validators
(:class:`~DataCurator.pipeline.validators.ScriptConsistencyValidator`,
:class:`~DataCurator.pipeline.validators.PatternPreservationValidator`),
which resample an LLM rewrite, and by
:class:`~DataCurator.pipeline.modifiers.transfer.TransferCheckModifier`,
which writes the findings as a per-record signal column for a
``SetWhenModifier`` to route on — the Google-Translate draft cannot be
resampled, only sent through the fix path.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter
from typing import Collection, Dict, List, Optional, Sequence, Tuple

# Scripts that can fail a check when they appear in the hypothesis without a
# counterpart in the source. Allow-by-default: Latin, digits, punctuation,
# emoji and any script not listed here never count, so an exotic-but-benign
# codepoint cannot fail a translation. The cjk_punctuation bucket catches the
# fullwidth marks (，。！) qwen leaks even when no Han character appears.
GUARDED_SCRIPT_RANGES: Tuple[Tuple[int, int, str], ...] = tuple(
    sorted(
        [
            (0x0370, 0x03FF, "greek"),
            (0x1F00, 0x1FFF, "greek"),
            (0x0400, 0x04FF, "cyrillic"),
            (0x0500, 0x052F, "cyrillic"),
            (0x1C80, 0x1C8F, "cyrillic"),
            (0x2DE0, 0x2DFF, "cyrillic"),
            (0xA640, 0xA69F, "cyrillic"),
            (0x0530, 0x058F, "armenian"),
            (0x0590, 0x05FF, "hebrew"),
            (0x0600, 0x06FF, "arabic"),
            (0x0750, 0x077F, "arabic"),
            (0x08A0, 0x08FF, "arabic"),
            (0xFB50, 0xFDFF, "arabic"),
            (0xFE70, 0xFEFF, "arabic"),
            (0x0900, 0x097F, "devanagari"),
            (0x0E00, 0x0E7F, "thai"),
            (0x10A0, 0x10FF, "georgian"),
            (0x1100, 0x11FF, "hangul"),
            (0x3130, 0x318F, "hangul"),
            (0xAC00, 0xD7AF, "hangul"),
            (0x3040, 0x309F, "hiragana"),
            (0x30A0, 0x30FF, "katakana"),
            (0x31F0, 0x31FF, "katakana"),
            (0xFF66, 0xFF9F, "katakana"),
            (0x3400, 0x4DBF, "han"),
            (0x4E00, 0x9FFF, "han"),
            (0xF900, 0xFAFF, "han"),
            (0x20000, 0x2A6DF, "han"),
            (0x3000, 0x303F, "cjk_punctuation"),
            (0xFF01, 0xFF65, "cjk_punctuation"),
        ]
    )
)
_RANGE_STARTS = [start for start, _, _ in GUARDED_SCRIPT_RANGES]

PATTERN_KINDS: Tuple[str, ...] = ("number", "url", "email")
PATTERN_MODES: Tuple[str, ...] = ("missing", "both")

_URL_RE = re.compile(r"""https?://[^\s<>"'“”]+|www\.[^\s<>"'“”]+""", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# Space-grouped thousands only bind in exact 3-digit groups ("1 000 000"),
# so "1990 2" stays two tokens; [.,] groups/decimals bind freely, so
# "1,000.5", "1000,5" and "1.000,5" all canonicalize to the same digits.
_NUMBER_RE = re.compile(r"\d{1,3}(?:[ \u00A0\u202F]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)*")
# English ordinal/decade suffixes ("1st", "20th", "1980s"): such numbers
# legitimately renumber in Polish ("XX wieku", "latach 80."), so suffixed
# source tokens are excluded from the preservation check.
_ORDINAL_SUFFIX_RE = re.compile(r"(?:st|nd|rd|th|s)\b", re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?…\"')]}"


def script_of(codepoint: int) -> Optional[str]:
    """Guarded script containing ``codepoint``, or ``None`` when unguarded."""
    index = bisect_right(_RANGE_STARTS, codepoint) - 1
    if index >= 0:
        start, end, script = GUARDED_SCRIPT_RANGES[index]
        if codepoint <= end:
            return script
    return None


def foreign_scripts(
    source: str, hypothesis: str, *, allow: Collection[str] = ()
) -> Dict[str, str]:
    """Guarded scripts present in ``hypothesis`` but absent from ``source``.

    Returns ``{script: sample}`` where ``sample`` holds up to 8 distinct
    offending characters (ellipsized past that); an empty dict means clean.
    A script the source itself uses is never foreign (Cyrillic quoted in the
    English source may echo in the Polish output); ``allow`` whitelists
    further scripts.
    """
    allowed = set(allow)
    allowed.update(script for char in source if (script := script_of(ord(char))))
    findings: Dict[str, List[str]] = {}
    for char in hypothesis:
        script = script_of(ord(char))
        if script is None or script in allowed:
            continue
        chars = findings.setdefault(script, [])
        if char not in chars:
            chars.append(char)
    return {
        script: "".join(chars[:8]) + ("…" if len(chars) > 8 else "")
        for script, chars in findings.items()
    }


def validate_pattern_args(kinds: Sequence[str], mode: str) -> None:
    """Raise ``ValueError`` on an unknown pattern kind or mode (config typo guard)."""
    unknown = set(kinds) - set(PATTERN_KINDS)
    if unknown:
        raise ValueError(f"unknown pattern kind(s) {sorted(unknown)}; expected subset of {PATTERN_KINDS}")
    if mode not in PATTERN_MODES:
        raise ValueError(f"mode must be one of {PATTERN_MODES}, got {mode!r}")


def _strip_spans(text: str, spans: List[Tuple[int, int]]) -> str:
    """Blank the given ``(start, end)`` spans of ``text`` with spaces."""
    if not spans:
        return text
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _extract_numbers(text: str, *, exclude_suffixed: bool) -> Counter:
    """Multiset of number tokens in ``text``, canonicalized to digits only.

    ``exclude_suffixed`` drops tokens carrying an English ordinal/decade
    suffix — used on the source side, where they legitimately renumber.
    """
    counts: Counter = Counter()
    for match in _NUMBER_RE.finditer(text):
        if exclude_suffixed and _ORDINAL_SUFFIX_RE.match(text, match.end()):
            continue
        counts[re.sub(r"\D", "", match.group())] += 1
    return counts


def pattern_misses(
    source: str,
    hypothesis: str,
    *,
    kinds: Sequence[str] = PATTERN_KINDS,
    mode: str = "missing",
) -> Dict[str, List[str]]:
    """Invariant tokens of ``source`` that ``hypothesis`` fails to preserve.

    Returns a zero-suppressed ``{kind: [tokens]}`` — an empty dict means every
    checked token survived. URLs and emails must appear verbatim (trailing
    punctuation trimmed); numbers are compared as multisets of canonical
    digit strings, so any grouping/decimal restyling matches. URL/email spans
    are blanked before number extraction, so their digits are never counted
    twice. ``mode="both"`` additionally reports numbers the hypothesis
    invented under ``"extra_number"`` — strict, and prone to flagging
    legitimate Polish renumbering ("latach 80."), hence not the default.
    """
    validate_pattern_args(kinds, mode)
    findings: Dict[str, List[str]] = {}
    source_spans: List[Tuple[int, int]] = []
    hypothesis_spans: List[Tuple[int, int]] = []
    for kind, regex in (("url", _URL_RE), ("email", _EMAIL_RE)):
        tokens = []
        for match in regex.finditer(source):
            source_spans.append(match.span())
            tokens.append(match.group().rstrip(_TRAILING_PUNCT))
        hypothesis_spans.extend(match.span() for match in regex.finditer(hypothesis))
        if kind in kinds:
            misses = [token for token in tokens if token and token not in hypothesis]
            if misses:
                findings[kind] = misses
    if "number" in kinds:
        source_counts = _extract_numbers(_strip_spans(source, source_spans), exclude_suffixed=True)
        hypothesis_counts = _extract_numbers(
            _strip_spans(hypothesis, hypothesis_spans), exclude_suffixed=False
        )
        missing = sorted((source_counts - hypothesis_counts).elements())
        if missing:
            findings["number"] = missing
        if mode == "both":
            extra = sorted((hypothesis_counts - source_counts).elements())
            if extra:
                findings["extra_number"] = extra
    return findings
