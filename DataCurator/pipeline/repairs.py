"""Deterministic repairs over an LLM stage's parsed output.

An :class:`OutputRepair` rewrites fields of the structured output an
:class:`~DataCurator.pipeline.stages.llm.LLMStage` just parsed, *before* the
stage's validators see it. It is the place for corrections that are safe to
apply mechanically — the model's answer is fixed in place instead of being
rejected and resampled for a defect a rule can undo.

Repairs are config-driven like validators: declare them under a stage's
``repairs`` list with a ``_target_`` and Hydra instantiates them.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from DataCurator.LanguageTool.LanguageTool import LanguageToolChecker
from DataCurator.pipeline.errors import ServiceUnavailableError
from DataCurator.pipeline.modifiers.candidates import _MISSING, _resolve_path
from DataCurator.pipeline.modifiers.conditional import _set_path
from DataCurator.pipeline.stage import StageContext


class OutputRepair(ABC):
    """Rewrite a stage's parsed output (dumped to a dict) before validation."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or type(self).__name__

    @abstractmethod
    async def repair(self, output: Dict[str, Any], context: StageContext) -> Dict[str, Any]:
        """Return the repaired ``output``; ``context`` is read-only reference data."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class LanguageToolRepair(OutputRepair):
    """Apply LanguageTool's suggestion for a narrow set of rules to output fields.

    Only rules whose correction is unambiguous belong here — the default is
    ``WORD_REPEAT_RULE``, whose suggestion drops a doubled word ("były były"
    -> "były"). Spelling rules stay detection-only
    (:class:`~DataCurator.pipeline.validators.LanguageToolValidator`): their
    first suggestion for a rare but correct term is a different word.

    An unreachable server raises
    :class:`~DataCurator.pipeline.errors.ServiceUnavailableError`, as the
    validator does, so the outage is never mistaken for "nothing to repair".
    """

    def __init__(
        self,
        fields: Sequence[str],
        *,
        rules: Sequence[str] = ("WORD_REPEAT_RULE",),
        language: str = "pl-PL",
        remote_server: Optional[str] = None,
        retries: int = 5,
        skip_empty_fields: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """Store the fields to repair and the rule allow-list; the checker connects lazily."""
        super().__init__(name=name)
        if not rules:
            raise ValueError(f"{self.name}: 'rules' needs at least one LanguageTool rule id")
        self.fields = list(fields)
        self.rules: List[str] = list(rules)
        self.skip_empty_fields = skip_empty_fields
        self._checker_kwargs = dict(
            language=language,
            remote_server=remote_server,
            allowed_fixes=self.rules,
            retries=retries,
        )
        self._checker: Optional[LanguageToolChecker] = None
        self._lock = asyncio.Lock()

    async def _get_checker(self) -> LanguageToolChecker:
        """Lazily construct the (network-connecting) checker on first use."""
        async with self._lock:
            if self._checker is None:
                self._checker = LanguageToolChecker(**self._checker_kwargs)
        return self._checker

    async def repair(self, output: Dict[str, Any], context: StageContext) -> Dict[str, Any]:
        """Correct each configured field with the allow-listed rules' suggestions."""
        repaired = StageContext(dict(output))
        for field in self.fields:
            value = _resolve_path(repaired, field)
            if value is _MISSING:
                continue  # the validators own "field missing"
            text = str(value)
            if self.skip_empty_fields and not text:
                continue
            try:
                checker = await self._get_checker()
                fixed, _ = await asyncio.to_thread(checker.fix_and_get_fixes, text)
            except ServiceUnavailableError:
                raise
            except ConnectionError as exc:
                raise ServiceUnavailableError(
                    f"{self.name}: LanguageTool unreachable: {exc}"
                ) from exc
            if fixed != text:
                logger.debug(f"{self.name}: repaired {field!r} with {self.rules}")
                _set_path(repaired, field, fixed)
        return repaired.to_dict()
