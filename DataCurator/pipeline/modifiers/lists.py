"""List-shaping stage modifiers.

Three small :class:`~DataCurator.pipeline.stage.FieldModifier`\\ s that
reshape a list field in place (or into ``target``):

* :class:`ListDedupModifier` — drop duplicate items, keeping first occurrences;
* :class:`ListLimitModifier` — cap the list at its first ``limit`` items;
* :class:`ListPickModifier` — keep a random sample of ``count`` items.

The motivating case is bounding what a record feeds a prompt: a pathological
record with dozens of (often repeated) LanguageTool findings inflates both
the prompt and the structured output that must evaluate every finding.
Chained as dedup → limit, the findings list stays representative but bounded.
The random picker covers sampling instead of truncating — e.g. drawing a few
items from a long candidate list without always favouring its head.
"""
from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, List, Optional

import orjson

from DataCurator.pipeline.stage import FieldModifier, ModifierPhase, StageContext

# Sentinel distinguishing "key path resolved to a value" from "path absent".
_MISSING: Any = object()


def _as_list(name: str, value: Any) -> List[Any]:
    """Return ``value`` as a list, raising a clear error for anything else.

    Strings are sequences too, so only genuine list/tuple values are accepted
    — a misconfigured ``source`` pointing at text should fail loudly, not be
    exploded into characters.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"{name}: expected a list field, got {type(value).__name__}")


def _canonical(value: Any) -> Any:
    """Return a hashable stand-in for ``value`` (JSON-serialise unhashables)."""
    try:
        hash(value)
        return value
    except TypeError:
        return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


class ListDedupModifier(FieldModifier):
    """Drop duplicate items from a list field, keeping first occurrences."""

    def __init__(
        self,
        source: str,
        target: Optional[str] = None,
        *,
        key: Optional[str] = None,
        phase: ModifierPhase = ModifierPhase.BEFORE,
        required: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Configure what makes two items duplicates.

        Without ``key``, whole items are compared (dicts and other
        unhashables via their canonical JSON form). With ``key`` — a field
        name or dotted path into mapping items — items are duplicates when
        the path resolves to the same value; an item where the path is
        missing is always kept, since it cannot be judged.
        """
        super().__init__(source, target, phase=phase, required=required, name=name)
        self.key = key

    async def transform(self, value: Any, context: StageContext) -> List[Any]:
        """Return the list with later duplicates removed, order preserved."""
        items = _as_list(self.name, value)
        seen: set = set()
        kept: List[Any] = []
        for item in items:
            key = self._dedup_key(item)
            if key is not _MISSING:
                if key in seen:
                    continue
                seen.add(key)
            kept.append(item)
        return kept

    def _dedup_key(self, item: Any) -> Any:
        """Resolve the comparison key for ``item`` (``_MISSING`` if absent)."""
        value = item
        if self.key is not None:
            for part in self.key.split("."):
                if not isinstance(value, Mapping) or part not in value:
                    return _MISSING
                value = value[part]
        return _canonical(value)


class ListLimitModifier(FieldModifier):
    """Truncate a list field to its first ``limit`` items."""

    def __init__(
        self,
        source: str,
        target: Optional[str] = None,
        *,
        limit: int,
        overflow_field: Optional[str] = None,
        phase: ModifierPhase = ModifierPhase.BEFORE,
        required: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Configure the cap.

        ``limit`` is the maximum number of items kept (the head of the list).
        When ``overflow_field`` is set, the number of trimmed items is written
        there (0 when nothing was trimmed — falsy, so a template can render an
        honest ``+N more`` note only when items were actually dropped).
        """
        super().__init__(source, target, phase=phase, required=required, name=name)
        if int(limit) < 0:
            raise ValueError(f"{self.name}: limit must be >= 0, got {limit}")
        self.limit = int(limit)
        self.overflow_field = overflow_field

    async def transform(self, value: Any, context: StageContext) -> List[Any]:
        """Return the first ``limit`` items, recording the overflow if asked."""
        items = _as_list(self.name, value)
        if self.overflow_field:
            context.set(self.overflow_field, max(0, len(items) - self.limit))
        return items[: self.limit]


class ListPickModifier(FieldModifier):
    """Keep a random sample of ``count`` items from a list field."""

    def __init__(
        self,
        source: str,
        target: Optional[str] = None,
        *,
        count: int = 1,
        preserve_order: bool = True,
        unwrap: bool = False,
        phase: ModifierPhase = ModifierPhase.BEFORE,
        required: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Configure the sample.

        ``count`` items are drawn without replacement; a list of ``count`` or
        fewer items is kept whole. ``preserve_order`` keeps the sample in the
        items' original relative order instead of the draw order. ``unwrap``
        (only with ``count=1``) writes the picked item itself rather than a
        one-item list; an empty source list then raises, since there is
        nothing to unwrap.
        """
        super().__init__(source, target, phase=phase, required=required, name=name)
        if int(count) < 1:
            raise ValueError(f"{self.name}: count must be >= 1, got {count}")
        if unwrap and int(count) != 1:
            raise ValueError(f"{self.name}: unwrap requires count=1, got count={count}")
        self.count = int(count)
        self.preserve_order = bool(preserve_order)
        self.unwrap = bool(unwrap)
        self._rng = random.Random()

    async def transform(self, value: Any, context: StageContext) -> Any:
        """Return the sampled items (or the bare item when unwrapping)."""
        items = _as_list(self.name, value)
        if self.unwrap and not items:
            raise ValueError(f"{self.name}: cannot pick from empty list {self.source!r}")
        if len(items) <= self.count:
            picked = list(items)
        else:
            indices = self._rng.sample(range(len(items)), self.count)
            if self.preserve_order:
                indices.sort()
            picked = [items[i] for i in indices]
        return picked[0] if self.unwrap else picked
