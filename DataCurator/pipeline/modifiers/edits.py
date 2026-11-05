"""Apply an LLM's span edits to a list of segments.

The actuation half of a span-edit stage: the model returns
``[{segment_id, span, replacement, ...}]`` (validated beforehand by
:class:`~DataCurator.pipeline.validators.EditSpanValidator`) instead of
reprinting whole texts, and this *after* modifier applies them
deterministically — a text changes only where an edit says so.
"""
from __future__ import annotations

from typing import Optional

from DataCurator.pipeline.modifiers.candidates import _MISSING, _resolve_path
from DataCurator.pipeline.stage import ModifierPhase, StageContext, StageModifier


class ApplyEditsModifier(StageModifier):
    """Write ``segments`` with ``edits`` applied to ``target``.

    ``segments`` is a ``[{id, <text_key>, ...}]`` list in the context;
    ``edits`` a ``[{segment_id, span, replacement}]`` list (a dotted path such
    as ``adjustment.edits``). Each edit replaces every exact occurrence of
    ``span`` in its segment's text, in list order. ``target`` receives one
    ``{id, source_id, text, edited}`` entry per segment — also when there are
    no edits, so every record carries the same result column.

    An edit naming an unknown segment, or whose span no longer occurs (an
    earlier edit consumed it), raises: the record is flagged rather than
    silently losing part of its adjustment. With ``edits`` absent the modifier
    skips unless ``required`` is set.
    """

    def __init__(
        self,
        edits: str,
        segments: str,
        target: str,
        *,
        text_key: str = "translation",
        required: bool = True,
        phase: ModifierPhase = ModifierPhase.AFTER,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.edits = edits
        self.segments = segments
        self.target = target
        self.text_key = text_key
        self.required = required
        self.phases = (phase,)

    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        """Apply the edits segment by segment and publish the result list."""
        edits = _resolve_path(context, self.edits)
        segments = _resolve_path(context, self.segments)
        if edits is _MISSING or segments is _MISSING:
            if self.required:
                raise KeyError(f"{self.name}: {self.edits!r} or {self.segments!r} missing from context")
            return

        texts = {str(s["id"]): str(s[self.text_key]) for s in segments}
        edited: set = set()
        for edit in edits or []:
            segment_id, span = str(edit["segment_id"]), edit["span"]
            if segment_id not in texts:
                raise KeyError(f"{self.name}: edit targets unknown segment {segment_id!r}")
            if not span or span not in texts[segment_id]:
                raise KeyError(f"{self.name}: span {span!r} no longer occurs in segment {segment_id!r}")
            texts[segment_id] = texts[segment_id].replace(span, edit["replacement"])
            edited.add(segment_id)

        context.set(
            self.target,
            [
                {
                    "id": str(s["id"]),
                    "source_id": s.get("source_id"),
                    "text": texts[str(s["id"])],
                    "edited": str(s["id"]) in edited,
                }
                for s in segments
            ],
        )
