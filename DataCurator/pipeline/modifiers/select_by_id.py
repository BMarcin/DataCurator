"""Resolve a selected candidate id back to its text.

The inverse of :class:`~DataCurator.pipeline.modifiers.candidates.CandidatesModifier`:
that one folds several context values into a ``[{id, text}, ...]`` list for an LLM
to choose from; this one takes the id the LLM chose and writes the matching text
back out. Used as an *after* modifier on the pick stage so the model returns only
``selected_id`` (never the whole translation reprinted), and we reconstruct the
winning text ourselves — cheaper, and verbatim by construction.
"""
from __future__ import annotations

from typing import Optional

from DataCurator.pipeline.modifiers.candidates import _MISSING, _resolve_path
from DataCurator.pipeline.stage import ModifierPhase, StageContext, StageModifier


class SelectByIdModifier(StageModifier):
    """Write the ``text`` of the ``candidates`` entry whose ``id`` matches a selection.

    Reads ``id_source`` (a field name or dotted path, e.g.
    ``the_best_translation.selected_id``) and ``candidates`` (a ``[{id, text}, ...]``
    list) from the context, and writes the matching entry's ``text`` to ``target``.

    When either input is absent the modifier skips silently, unless ``required`` is
    set (then it raises) — this lets it sit on a stage some records bypass. A
    ``selected`` id that is present but matches no candidate always raises, so the
    record is flagged rather than silently losing its post-edit.
    """

    def __init__(
        self,
        id_source: str,
        candidates: str,
        target: str,
        *,
        required: bool = True,
        exists_ok: bool = True,
        phase: ModifierPhase = ModifierPhase.AFTER,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.id_source = id_source
        self.candidates = candidates
        self.target = target
        self.required = required
        self.exists_ok = exists_ok
        self.phases = (phase,)

    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        """Resolve ``id_source`` against ``candidates`` and publish the winner's text."""
        selected = _resolve_path(context, self.id_source)
        candidates = _resolve_path(context, self.candidates)
        if selected is _MISSING or candidates is _MISSING:
            if self.required:
                raise KeyError(
                    f"{self.name}: {self.id_source!r} or {self.candidates!r} missing "
                    f"from context"
                )
            return

        text = next(
            (c["text"] for c in candidates if c.get("id") == selected), _MISSING
        )
        if text is _MISSING:
            raise KeyError(
                f"{self.name}: selected id {selected!r} is not among the candidate ids"
            )

        if not self.exists_ok and self.target in context:
            raise KeyError(
                f"{self.name}: output field {self.target!r} already exists in context; "
                f"set exists_ok=True to overwrite"
            )
        context.set(self.target, text)
