"""Field-coalescing stage.

Writes the first present, non-empty of several context values — plain field
names or dotted paths into nested dicts — into one target field. This is how
a routed pipeline gives every record a uniform result column: records the
gate sent through fix+pick carry ``the_best_translation.selected_translation``,
records that passed the gate untouched only carry the draft, and a final
coalesce stage folds both cases into e.g. ``final_translation``.

A CPU stage (no model), so the distributed wire routes it to any worker.
Error-flagged records never reach ``process`` — the runner and worker pass
them through untouched — so they simply lack the target field until a rerun
heals them.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from DataCurator.pipeline.modifiers.candidates import _MISSING, _resolve_path
from DataCurator.pipeline.stage import Stage, StageContext


class CoalesceStage(Stage):
    """Write the first present, non-empty of ``sources`` into ``target``."""

    def __init__(
        self,
        target: str,
        sources: Sequence[str],
        *,
        required: bool = True,
        exists_ok: bool = False,
        name: Optional[str] = None,
        modifiers=None,
    ) -> None:
        """Configure the coalesce.

        ``sources`` is the ordered list of field names / dotted paths to try;
        the first one that resolves to a non-empty value (not ``None``/``""``)
        wins. ``required`` (default ``True``) makes a record where none
        resolves raise; set it ``False`` to leave ``target`` unset instead.
        ``exists_ok`` guards ``target`` like in
        :class:`~DataCurator.pipeline.modifiers.candidates.CandidatesModifier`.
        """
        super().__init__(name=name, modifiers=modifiers)
        self.target = target
        self.sources: List[str] = list(sources)
        self.required = required
        self.exists_ok = exists_ok

    async def process(self, context: StageContext) -> None:
        """Resolve ``sources`` in order and publish the first usable value."""
        if not self.exists_ok and self.target in context:
            raise KeyError(
                f"{self.name}: output field {self.target!r} already exists in context; "
                f"set exists_ok=True to overwrite"
            )
        for path in self.sources:
            value = _resolve_path(context, path)
            if value is not _MISSING and value not in (None, ""):
                context.set(self.target, value)
                return
        if self.required:
            raise KeyError(
                f"{self.name}: no source among {self.sources!r} resolved to a "
                f"non-empty value"
            )
