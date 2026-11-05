"""Derive the gate verdict from its own issue severities.

The gate's routing contract — ``needs_edit`` iff any issue is major or
serious — is stated in the schema and the prompt, but the model's ``verdict``
field occasionally contradicts the issues it just listed, and the fix/pick
filters route on that field. This *after* modifier recomputes the verdict
from the severities and overwrites it, so routing rests on arithmetic rather
than on the model honouring an iff. When the model disagreed, its original
verdict is kept under ``model_verdict`` inside the same mapping — the key's
presence is the mismatch marker, so counting gate inconsistencies in the
output JSONL is a one-liner.

A flip to ``publishable`` can leave minor issues unfixed: the model, believing
``needs_edit``, left ``improved_translation`` empty per its contract, so the
record finalizes to the raw draft — the same routing a consistent gate would
have produced.
"""
from __future__ import annotations

from typing import Optional, Sequence

from loguru import logger

from DataCurator.pipeline.stage import ModifierPhase, StageContext, StageModifier


class DeriveVerdictModifier(StageModifier):
    """Overwrite ``<source>.verdict`` with the one its ``issues`` imply.

    Reads the mapping under ``source`` (a gate stage's ``output_field``),
    derives ``needs_edit`` iff any issue's ``severity`` is in
    ``edit_severities``, and overwrites ``verdict``. On a mismatch the
    model's verdict is preserved under ``mismatch_key``.

    When ``source`` is absent the modifier skips silently, unless
    ``required`` is set (then it raises) — after-modifiers only run on
    stage success, so with the default wiring the field is always there.
    """

    def __init__(
        self,
        source: str = "gate_answer",
        *,
        edit_severities: Sequence[str] = ("major", "serious"),
        mismatch_key: str = "model_verdict",
        required: bool = True,
        phase: ModifierPhase = ModifierPhase.AFTER,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.source = source
        self.edit_severities = frozenset(edit_severities)
        self.mismatch_key = mismatch_key
        self.required = required
        self.phases = (phase,)

    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        """Recompute the verdict; record the model's when it disagreed."""
        if self.source not in context:
            if self.required:
                raise KeyError(f"{self.name}: {self.source!r} missing from context")
            return

        answer = dict(context[self.source])
        issues = answer.get("issues") or []
        derived = (
            "needs_edit"
            if any(issue.get("severity") in self.edit_severities for issue in issues)
            else "publishable"
        )
        if answer.get("verdict") != derived:
            logger.warning(
                f"{self.name}: model verdict {answer.get('verdict')!r} contradicts its "
                f"issue severities; overriding with {derived!r}"
            )
            answer[self.mismatch_key] = answer.get("verdict")
            answer["verdict"] = derived
            context.set(self.source, answer)
