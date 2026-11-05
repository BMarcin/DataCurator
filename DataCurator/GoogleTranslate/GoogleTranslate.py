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

from DataCurator.pipeline.errors import ServiceUnavailableError


class EmptyTranslationError(RuntimeError):
    """Google answered 200 with a blank translation of a non-blank text.

    Seen in bursts under throttling; it is a failed call, not a result, so the
    retry loop treats it like any other error.
    """


class GoogleTranslateUnavailable(ServiceUnavailableError):
    """Google Translate produced no usable translation after every retry."""


class GoogleTranslator:
    def __init__(
        self,
        source_language_code: str,
        target_language_code: str,
        retries: int = 5,
        timeout: float = 30.0,
        concurrency: int = 8,
        remote_server: Optional[str] = None,
        client_name: Optional[str] = None,
    ) -> None:
        """Configure the translator with language pair, HTTP knobs and endpoint.

        ``remote_server`` is the full URL of the translate endpoint and ``client_name``
        the value of its ``client`` query parameter. Both are bindings (``bindings.yaml``
        on a worker, ``GOOGLE_TRANSLATE_URL`` and ``GOOGLE_TRANSLATE_CLIENT`` standalone);
        leaving them unset only fails on first use, so a disabled stage still builds.
        """
        self.source_language_code = source_language_code
        self.target_language_code = target_language_code
        self.retries = retries
        self.timeout = timeout
        self.concurrency = concurrency
        self.remote_server = remote_server
        self.client_name = client_name
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
        """Translate ``text`` once via Google Translate, normalising the response shape.

        A blank translation of a non-blank ``text`` raises
        :class:`EmptyTranslationError` instead of being returned.
        """
        if not text.strip():
            return text
        params = {
            "client": self.client_name,
            "sl": self.source_language_code,
            "tl": self.target_language_code,
            "q": text,
        }
        response = await client.get(self.remote_server, params=params)
        response.raise_for_status()
        payload = response.json()

        # Response shapes seen in the wild:
        #   ["translated text", "detected_lang"]
        #   [["seg1", "seg2", ...], "detected_lang"]
        #   [[["seg1", "src1"], ["seg2", "src2"], ...], ...]
        first = payload[0]
        if isinstance(first, str):
            translated = first
        elif isinstance(first, list):
            parts: list[str] = []
            for seg in first:
                if isinstance(seg, str):
                    parts.append(seg)
                elif isinstance(seg, list) and seg and isinstance(seg[0], str):
                    parts.append(seg[0])
            translated = "".join(parts)
        else:
            raise RuntimeError(f"Unexpected Google Translate response shape: {payload!r}")
        if not translated.strip():
            raise EmptyTranslationError(f"empty translation for a {len(text)}-char text: {payload!r:.200}")
        return translated

    async def translate_with_retries(
        self, client: httpx.AsyncClient, text: str
    ) -> Optional[str]:
        """Translate ``text`` with exponential-backoff retries; return ``None`` on permanent failure.

        An unbound endpoint raises :class:`GoogleTranslateUnavailable` instead: it is a
        configuration error, and returning ``None`` would let the modifier silently
        keep untranslated text.
        """
        if not self.remote_server or not self.client_name:
            raise GoogleTranslateUnavailable(
                "no Google Translate endpoint bound (remote_server and client_name are required)"
            )

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
