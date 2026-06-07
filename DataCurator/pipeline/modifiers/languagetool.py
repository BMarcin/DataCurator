"""LanguageTool stage modifier.

Wraps :class:`~DataCurator.LanguageTool.LanguageTool.LanguageToolChecker`
as a :class:`~DataCurator.pipeline.stage.FieldModifier`. Attached in the
``before`` phase of an LLM stage it normalises a text field (applying only
the allow-listed auto-fixes) before the prompt is built, and can optionally
stash the remaining unresolved issues in another context field so the
prompt can show them to the model.

The underlying checker is created lazily on first use, so building a
pipeline does not require the LanguageTool server to be reachable.
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Tuple

from DataCurator.LanguageTool.LanguageTool import LanguageToolChecker
from DataCurator.pipeline.stage import FieldModifier, ModifierPhase, StageContext


class LanguageToolModifier(FieldModifier):
    """Fix ``source`` with LanguageTool; optionally record unresolved issues."""

    def __init__(
        self,
        source: str,
        target: Optional[str] = None,
        *,
        language: str = "pl-PL",
        remote_server: Optional[str] = "http://languagetool.loc",
        allowed_fixes: Optional[List[str]] = None,
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
        """Return the LanguageTool-fixed text, recording issues if configured.

        The checker is synchronous, so it runs in a worker thread to avoid
        blocking the event loop while many records are processed.
        """
        checker = await self._get_checker()
        fixed, issues = await asyncio.to_thread(checker.format_issues, str(value))
        if self.issues_field:
            context.set(self.issues_field, issues)
        return fixed
