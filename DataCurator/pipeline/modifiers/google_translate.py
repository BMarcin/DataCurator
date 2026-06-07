"""Google Translate stage modifier.

Wraps :class:`~DataCurator.GoogleTranslate.GoogleTranslate.GoogleTranslator`
as a :class:`~DataCurator.pipeline.stage.FieldModifier`, so a reference
machine translation can be produced as a pre-pass and dropped into the
context for a downstream LLM stage to compare against. Translation
failures leave the original value untouched (zero-drop).
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx

from DataCurator.GoogleTranslate.GoogleTranslate import GoogleTranslator
from DataCurator.pipeline.stage import FieldModifier, ModifierPhase, StageContext


class GoogleTranslateModifier(FieldModifier):
    """Translate ``source`` and write the result to ``target`` (added if new)."""

    def __init__(
        self,
        source: str,
        target: Optional[str] = None,
        *,
        source_language_code: str,
        target_language_code: str,
        retries: int = 5,
        timeout: float = 30.0,
        concurrency: int = 8,
        phase: ModifierPhase = ModifierPhase.BEFORE,
        required: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Configure the language pair and HTTP knobs of the translator."""
        super().__init__(source, target, phase=phase, required=required, name=name)
        self._translator = GoogleTranslator(
            source_language_code=source_language_code,
            target_language_code=target_language_code,
            retries=retries,
            timeout=timeout,
            concurrency=concurrency,
        )
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily build and reuse a single async HTTP client across records."""
        async with self._lock:
            if self._client is None:
                self._client = self._translator._client()
        return self._client

    async def transform(self, value: Any, context: StageContext) -> str:
        """Return the translation of ``value``, or the original on failure."""
        client = await self._get_client()
        translated = await self._translator.translate_with_retries(client, str(value))
        return translated if translated is not None else str(value)
