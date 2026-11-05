"""Deterministic transfer-check stage modifier — signal acquisition only.

Runs one pure check from :mod:`DataCurator.pipeline.transfer` over a
``(source_field, hypothesis_field)`` pair and writes the findings mapping to
``target``. Like :class:`~DataCurator.pipeline.modifiers.metricx.MetricXScoreModifier`
it takes no decisions: pair it with a
:class:`~DataCurator.pipeline.modifiers.conditional.SetWhenModifier` whose
expression reads the column (a non-empty mapping is truthy) to act on it —
the gate override that routes a defective Google-Translate draft through the
fix path — or use it alone as per-record telemetry.

``{}`` means checked-and-clean; an **absent** column means the pair was
never checked (an input field missing or empty), so a downstream truthiness
read fails open — the MetricX "absence = not scored" convention.
"""
from __future__ import annotations

from typing import Optional, Sequence

from loguru import logger

from DataCurator.pipeline.stage import ModifierPhase, StageContext, StageModifier
from DataCurator.pipeline.transfer import foreign_scripts, pattern_misses, validate_pattern_args

_CHECKS = ("script", "patterns")


class TransferCheckModifier(StageModifier):
    """Write one transfer check's findings over a text pair to ``target``."""

    def __init__(
        self,
        *,
        check: str,
        target: str,
        source_field: str = "source_text",
        hypothesis_field: str = "target_text",
        allow_scripts: Sequence[str] = (),
        kinds: Sequence[str] = ("number", "url", "email"),
        mode: str = "missing",
        phase: ModifierPhase = ModifierPhase.AFTER,
        name: Optional[str] = None,
    ) -> None:
        """Validate the ``check`` selector and its per-check arguments eagerly.

        ``check`` picks the pure function: ``script`` →
        :func:`~DataCurator.pipeline.transfer.foreign_scripts` (writes
        ``{script: offending chars}``), ``patterns`` →
        :func:`~DataCurator.pipeline.transfer.pattern_misses` (writes
        ``{kind: miss count}`` — counts, so a downstream expression can
        compare with ``>``). ``allow_scripts`` applies to the former,
        ``kinds``/``mode`` to the latter.
        """
        super().__init__(name=name)
        if check not in _CHECKS:
            raise ValueError(f"{self.name}: check must be one of {_CHECKS}, got {check!r}")
        validate_pattern_args(kinds, mode)
        self.check = check
        self.target = target
        self.source_field = source_field
        self.hypothesis_field = hypothesis_field
        self.allow_scripts = tuple(allow_scripts)
        self.kinds = tuple(kinds)
        self.mode = mode
        self.phases = (phase,)

    async def modify(self, context: StageContext, phase: ModifierPhase) -> None:
        """Run the configured check and write its findings to ``target``."""
        source = str(context.get(self.source_field) or "")
        hypothesis = str(context.get(self.hypothesis_field) or "")
        if not source or not hypothesis:
            logger.debug(
                f"{self.name}: {self.source_field!r}/{self.hypothesis_field!r} "
                f"missing or empty; nothing to check"
            )
            return
        if self.check == "script":
            findings: dict = foreign_scripts(source, hypothesis, allow=self.allow_scripts)
        else:
            findings = {
                kind: len(tokens)
                for kind, tokens in pattern_misses(
                    source, hypothesis, kinds=self.kinds, mode=self.mode
                ).items()
            }
        if findings:
            logger.debug(f"{self.name}: {self.check} findings {findings} -> {self.target!r}")
        context.set(self.target, findings)
