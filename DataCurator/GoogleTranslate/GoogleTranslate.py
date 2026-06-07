"""Async Google Translate client used as a pipeline pre-pass.

Extracted from the standalone `run_google_translate.py` script so the same
translation logic can run as a stage step before the LLM call.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from loguru import logger
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential


GOOGLE_TRANSLATE_URL = "https://clients5.google.com/translate_a/t"


class GoogleTranslator:
    def __init__(
        self,
        source_language_code: str,
        target_language_code: str,
        retries: int = 5,
        timeout: float = 30.0,
        concurrency: int = 8,
    ) -> None:
        """Configure the translator with language pair and HTTP knobs."""
        self.source_language_code = source_language_code
        self.target_language_code = target_language_code
        self.retries = retries
        self.timeout = timeout
        self.concurrency = concurrency
        # Built lazily inside the running loop on first use; see _get_semaphore.
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Lazily build the concurrency gate bound to the running event loop.

        Created on first use rather than in ``__init__`` (which may run
        outside any loop, e.g. while Hydra builds the pipeline), and with no
        ``await`` between the ``None`` check and the assignment so concurrent
        callers can't race to create two competing gates.
        """
        semaphore = self._semaphore
        if semaphore is None:
            semaphore = self._semaphore = asyncio.Semaphore(self.concurrency)
        return semaphore

    def _client(self) -> httpx.AsyncClient:
        """Build an :class:`httpx.AsyncClient` honouring the configured timeout and concurrency."""
        timeout = httpx.Timeout(self.timeout)
        limits = httpx.Limits(
            max_connections=self.concurrency,
            max_keepalive_connections=self.concurrency,
        )
        return httpx.AsyncClient(timeout=timeout, limits=limits, http2=False)

    async def translate(self, client: httpx.AsyncClient, text: str) -> str:
        """Translate ``text`` once via Google Translate, normalising the response shape."""
        if not text:
            return ""
        params = {
            "client": "dict-chrome-ex",
            "sl": self.source_language_code,
            "tl": self.target_language_code,
            "q": text,
        }
        response = await client.get(GOOGLE_TRANSLATE_URL, params=params)
        response.raise_for_status()
        payload = response.json()

        # Response shapes seen in the wild:
        #   ["translated text", "detected_lang"]
        #   [["seg1", "seg2", ...], "detected_lang"]
        #   [[["seg1", "src1"], ["seg2", "src2"], ...], ...]
        first = payload[0]
        if isinstance(first, str):
            return first
        if isinstance(first, list):
            parts: list[str] = []
            for seg in first:
                if isinstance(seg, str):
                    parts.append(seg)
                elif isinstance(seg, list) and seg and isinstance(seg[0], str):
                    parts.append(seg[0])
            return "".join(parts)
        raise RuntimeError(f"Unexpected Google Translate response shape: {payload!r}")

    async def translate_with_retries(
        self, client: httpx.AsyncClient, text: str
    ) -> Optional[str]:
        """Translate ``text`` with exponential-backoff retries; return ``None`` on permanent failure."""
        async def _call() -> str:
            """Single translation attempt wrapped by tenacity."""
            return await self.translate(client, text)

        try:
            # Hold one concurrency slot for the whole retry sequence, so a
            # backing-off request doesn't free its slot to a new caller and
            # let total in-flight requests drift past ``concurrency``.
            async with self._get_semaphore():
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.retries),
                    wait=wait_exponential(multiplier=1, min=2, max=30),
                    reraise=True,
                ):
                    with attempt:
                        return await _call()
        except Exception as e:
            logger.opt(exception=e).error(
                f"Google Translate request failed after {self.retries} retries "
                f"(src_preview={text[:120]!r})"
            )
            return None
        return None
