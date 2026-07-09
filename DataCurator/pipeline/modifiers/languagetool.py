"""LanguageTool stage modifier.

Wraps :class:`~DataCurator.LanguageTool.LanguageTool.LanguageToolChecker`
as a :class:`~DataCurator.pipeline.stage.FieldModifier`. Attached in the
``before`` phase of an LLM stage it uses LanguageTool as a detector: it
leaves the text field unchanged and stashes the detected allow-listed
issues in another context field so the prompt can show them to the model.

The underlying checker is created lazily on first use, so building a
pipeline does not require the LanguageTool server to be reachable.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Tuple

from DataCurator.LanguageTool.LanguageTool import LanguageToolChecker
from DataCurator.pipeline.stage import FieldModifier, ModifierPhase, StageContext


class LanguageToolModifier(FieldModifier):
    """Detect LanguageTool issues in ``source`` (leaving it unchanged);
    optionally record them in ``issues_field``."""

    def __init__(
        self,
        source: str,
        target: Optional[str] = None,
        *,
        language: str = "pl-PL",
        remote_server: Optional[str] = "http://languagetool.loc",
        allowed_fixes: Optional[List[str]] = None,
        denied_fixes: Optional[List[str]] = None,
        retries: int = 5,
        max_passes: int = 3,
        issues_field: Optional[str] = None,
        phase: ModifierPhase = ModifierPhase.BEFORE,
        required: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Store checker configuration; the checker itself is built on demand."""
        super().__init__(source, target, phase=phase, required=required, name=name)
        self.issues_field = issues_field
        self._checker_kwargs = dict(
            language=language,
            remote_server=remote_server,
            allowed_fixes=allowed_fixes,
            denied_fixes=denied_fixes,
            retries=retries,
            max_passes=max_passes,
        )
        self._checker: Optional[LanguageToolChecker] = None
        self._lock = asyncio.Lock()

    async def _get_checker(self) -> LanguageToolChecker:
        """Lazily construct the (network-connecting) checker on first use."""
        async with self._lock:
            if self._checker is None:
                self._checker = LanguageToolChecker(**self._checker_kwargs)
        return self._checker

    async def transform(self, value: Any, context: StageContext) -> str:
        """Detect LanguageTool issues, recording them if configured; pass the
        text through unchanged.

        LanguageTool is used as a detector, not a fixer: the field value is
        returned untouched and the detected allow-listed issues are stashed
        in ``issues_field`` for the prompt to show the model. The checker is
        synchronous, so it runs in a worker thread to avoid blocking the
        event loop while many records are processed.
        """
        text = str(value)
        checker = await self._get_checker()
        issues = await asyncio.to_thread(checker.get_issues, text)
        if self.issues_field:
            context.set(self.issues_field, issues)
        return text
