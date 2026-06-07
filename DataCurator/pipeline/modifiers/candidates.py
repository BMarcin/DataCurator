"""Candidate-list assembly stage modifier.

Folds several context values into a single list of ``{id, text}`` objects
under one target field — the shape a prompt template iterates over when it
presents N labelled candidates (``[Q1] … [Q2] … [Q3] …``).

The motivating case is a final selection stage that must show the model the
three rewrites an earlier review stage produced. Those live as sibling keys
inside a nested dict (``review_question.improved_primary``,
``…improved_polished``, ``…improved_alternative``), and neither
:class:`~DataCurator.pipeline.modifiers.rename.RenameModifier` (which moves a
single existing field) nor
:class:`~DataCurator.pipeline.modifiers.constants.ConstantsModifier` (which
writes literals) can fold them into a list. This modifier reads each source —
a plain field name or a dotted path into nested dicts — and emits
``[{"id": ..., "text": ...}, ...]`` in declared order.

``id``\\ s are generated as ``f"{id_prefix}{n}"`` for ``n`` starting at 1, so
``id_prefix: Q`` yields ``Q1``, ``Q2``, ``Q3`` — matching the labels the
prompt renders.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, List, Optional, Sequence

from DataCurator.pipeline.stage import ModifierPhase, StageContext, StageModifier

# Sentinel distinguishing "path resolved to a real value" from "path absent".
_MISSING: Any = object()


def _resolve_path(context: StageContext, path: str) -> Any:
    """Resolve a dotted ``a.b.c`` path through nested mappings, or ``_MISSING``.

    The first text is a context field; each later text indexes into the
    mapping the previous one yielded. Any missing text — or a non-mapping
    encountered mid-path — returns ``_MISSING`` rather than raising, so the
    caller decides whether absence is an error.
    """
    head, _, rest = path.partition(".")
    if head not in context:
        return _MISSING
    value: Any = context[head]
    for part in rest.split(".") if rest else []:
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


class CandidatesModifier(StageModifier):
    """Collect several context values into a ``[{id, text}, ...]`` list."""

    def __init__(
        self,
        target: str,
        sources: Sequence[str],
        *,
        id_prefix: str = "",
        required: bool = True,
        exists_ok: bool = False,
        phase: ModifierPhase = ModifierPhase.BEFORE,
        name: Optional[str] = None,
    ) -> None:
        """Configure the assembly.

        ``sources`` is the ordered list of field names / dotted paths to read;
        each present value becomes one ``{"id": f"{id_prefix}{n}", "text":
        value}`` entry written under ``target``. ``required`` (default ``True``)
        makes a missing source raise; set it ``False`` to silently drop absent
        sources, in which case ids number only the entries that were found.
        ``exists_ok`` guards ``target``: ``False`` (default) raises if it
        already exists, ``True`` overwrites it.
        """
        super().__init__(name=name)
        self.target = target
        self.sources: List[str] = list(sources)
        self.id_prefix = id_prefix
        self.required = required
        self.exists_ok = exists_ok
        self.phases = (phase,)

    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        """Build the candidate list from ``sources`` and write it to ``target``."""
        candidates: List[dict] = []
        for path in self.sources:
            value = _resolve_path(context, path)
            if value is _MISSING:
                if self.required:
                    raise KeyError(
                        f"{self.name}: required source {path!r} missing from context"
                    )
                continue
            candidates.append({"id": f"{self.id_prefix}{len(candidates) + 1}", "text": value})

        if not self.exists_ok and self.target in context:
            raise KeyError(
                f"{self.name}: output field {self.target!r} already exists in context; "
                f"set exists_ok=True to overwrite"
            )
        context.set(self.target, candidates)
