"""Standalone Google Translate stage.

Translates one text field per record via Google Translate and writes the
result into another field. Unlike :class:`~DataCurator.pipeline.modifiers.google_translate.GoogleTranslateModifier`
(which runs in an LLM stage's ``before`` phase), this is a first-class
:class:`~DataCurator.pipeline.stage.Stage`: it appears in the pipeline
``order``, persists its own sharded output, and is independently resumable,
so downstream stages reuse the translation instead of re-hitting the service
on every rerun. Reuses the async :class:`~DataCurator.GoogleTranslate.GoogleTranslate.GoogleTranslator`
client. A blank answer to a non-blank text is retried like any failure; once
the retries are spent the record raises :class:`~DataCurator.GoogleTranslate.GoogleTranslate.GoogleTranslateUnavailable`
— a connection-class error, so the distributed worker naks a chunk Google is
throttling instead of uploading drafts that are empty or still English.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx

from DataCurator.GoogleTranslate.GoogleTranslate import GoogleTranslateUnavailable, GoogleTranslator
from DataCurator.pipeline.stage import Stage, StageContext


class GoogleTranslateStage(Stage):
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
        remote_server: Optional[str] = None,
        client_name: Optional[str] = None,
        required: bool = False,
        name: Optional[str] = None,
        modifiers: Any = None,
    ) -> None:
        """Configure the language pair, HTTP knobs and endpoint of the translator."""
        super().__init__(name=name, modifiers=modifiers)
        self.source = source
        self.target = target or source
        self.required = required
        self._translator = GoogleTranslator(
            source_language_code=source_language_code,
            target_language_code=target_language_code,
            retries=retries,
            timeout=timeout,
            concurrency=concurrency,
            remote_server=remote_server,
            client_name=client_name,
        )
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily build and reuse a single async HTTP client across records."""
        async with self._lock:
            if self._client is None:
                self._client = self._translator._client()
        return self._client

    async def process(self, context: StageContext) -> None:
        """Translate ``source`` into ``target``; raise once every retry failed.

        If ``source`` is absent the stage is a no-op unless ``required`` is
        set, in which case a ``KeyError`` is raised — mirroring
        :class:`~DataCurator.pipeline.stage.FieldModifier`.
        """
        if self.source not in context:
            if self.required:
                raise KeyError(
                    f"{self.name}: required field {self.source!r} missing from context"
                )
            return
        value = str(context[self.source])
        client = await self._get_client()
        translated = await self._translator.translate_with_retries(client, value)
        if translated is None:
            raise GoogleTranslateUnavailable(
                f"{self.name}: no translation of {self.source!r} after "
                f"{self._translator.retries} attempts"
            )
        context.set(self.target, translated)
