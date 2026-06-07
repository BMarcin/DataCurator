"""Constant-injection stage modifier.

Adds fixed, config-supplied values to the context — fields that are the
same for every record and therefore do not live in the dataset, such as the
source/target language *names* a prompt template needs (``"English"``,
``"Polish"``). Unlike :class:`~DataCurator.pipeline.modifiers.rename.RenameModifier`,
which moves the *value of an existing field*, this modifier reads nothing:
it writes the literals straight from config, so it is the way to introduce a
brand-new field that has no source column.

By default each value is added with :meth:`StageContext.add` so a constant
that would clobber an existing field fails loudly; set ``overwrite=True`` (or
``exists_ok=True``) to replace existing values instead.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from omegaconf import OmegaConf

from DataCurator.pipeline.stage import ModifierPhase, StageContext, StageModifier


class ConstantsModifier(StageModifier):
    """Write fixed ``name -> value`` literals into the context."""

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        overwrite: bool = False,
        exists_ok: bool = False,
        phase: ModifierPhase = ModifierPhase.BEFORE,
        name: Optional[str] = None,
    ) -> None:
        """Capture the constants to inject.

        ``values`` maps each field name to the literal value to write.
        ``overwrite`` decides what happens when a name already exists in the
        context: ``False`` (default) raises, so a constant never silently
        masks a real dataset field; ``True`` replaces the existing value.
        ``exists_ok`` is a synonym for ``overwrite``: set it to ``True`` and a
        constant whose field already exists is written over instead of raising.
        """
        super().__init__(name=name)
        # Resolve OmegaConf nodes to plain Python so values are real str/int/...
        if OmegaConf.is_config(values):
            values = OmegaConf.to_container(values, resolve=True)  # type: ignore[assignment]
        self.values: Dict[str, Any] = dict(values)
        self.overwrite = overwrite
        self.exists_ok = exists_ok
        self.phases = (phase,)

    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        """Inject every constant, honouring the ``overwrite``/``exists_ok`` policy."""
        allow_existing = self.overwrite or self.exists_ok
        for field, value in self.values.items():
            if allow_existing:
                context.set(field, value)
            else:
                context.add(field, value)
