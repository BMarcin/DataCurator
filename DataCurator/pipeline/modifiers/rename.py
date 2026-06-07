"""Field-rename stage modifier.

Renames (or copies) one or more context fields according to a ``fields``
mapping of ``input -> output`` names. Each input value is written under its
output name; with ``keep=False`` (the default) the original input field is
then dropped, so the field is *renamed*, while ``keep=True`` leaves the input
in place, so the field is *copied*.

A handy ``before``-phase pre-pass for aligning a dataset's field names with
what a prompt template or downstream stage expects, without touching the data
itself. Unlike the single-field modifiers it does not subclass
:class:`~DataCurator.pipeline.stage.FieldModifier`, because it rewrites several
fields at once and may delete the originals.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from DataCurator.pipeline.stage import ModifierPhase, StageContext, StageModifier


class RenameModifier(StageModifier):
    """Rename (or copy) context fields per an ``input -> output`` mapping."""

    def __init__(
        self,
        fields: Mapping[str, str],
        *,
        keep: bool = False,
        required: bool = False,
        exists_ok: bool = False,
        phase: ModifierPhase = ModifierPhase.BEFORE,
        name: Optional[str] = None,
    ) -> None:
        """Configure the rename mapping.

        ``fields`` maps each existing input field name to the output name it
        should be written under. ``keep`` decides what happens to the input:
        ``False`` (default) removes it after the copy — a true rename — while
        ``True`` leaves it in place, copying the value under the new name.

        A field listed in ``fields`` but absent from the record is skipped,
        unless ``required=True``, in which case a missing input raises.

        ``exists_ok`` guards the *output*: ``False`` (default) raises if an
        output name already exists in the context and isn't itself being
        renamed away, so a rename never silently clobbers an unrelated field;
        ``True`` overwrites the existing value instead.
        """
        super().__init__(name=name)
        self.fields: Dict[str, str] = dict(fields)
        self.keep = keep
        self.required = required
        self.exists_ok = exists_ok
        self.phases = (phase,)

    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        """Apply every rename, reading inputs from a snapshot of the context.

        Input values are captured up front so a mapping whose output name is
        also another mapping's input (e.g. ``{a: b, b: c}``) never chains: each
        rename sees the original values, not ones written earlier in the pass.
        """
        pending: List[Tuple[str, str, Any]] = []
        for source, target in self.fields.items():
            if source not in context:
                if self.required:
                    raise KeyError(
                        f"{self.name}: required input field {source!r} missing from context"
                    )
                continue
            pending.append((source, target, context[source]))

        if not self.exists_ok:
            # A target only clobbers a real field if it already exists and is
            # neither the field being renamed onto itself nor a source whose
            # value is simultaneously being moved out of the way.
            sources = {source for source, _target, _value in pending}
            clashes = sorted(
                target
                for source, target, _value in pending
                if target != source and target in context and target not in sources
            )
            if clashes:
                raise KeyError(
                    f"{self.name}: output field(s) {clashes!r} already exist in context; "
                    f"set exists_ok=True to overwrite"
                )

        for _source, target, value in pending:
            context.set(target, value)

        if not self.keep:
            # Never drop a field that is itself the output of another rename.
            targets = {target for _source, target, _value in pending}
            for source, target, _value in pending:
                if source != target and source not in targets:
                    del context[source]
