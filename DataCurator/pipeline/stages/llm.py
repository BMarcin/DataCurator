"""Generic LLM stage for any OpenAI-compatible endpoint (e.g. vLLM).

``LLMStage`` runs one chat completion per record and parses the reply into
a Pydantic model (structured output). The conversation can be supplied two
ways:

* pre-built — read a list of ``{role, content}`` messages straight from a
  context field (``input_field``); or
* templated — render a ``prompt`` (a list of ``{role, template}`` entries
  naming Jinja2 files under ``prompts_dir``) against the record's fields.
  Prompt text lives in ``./prompts``, not in config. This is the path that
  makes modifiers useful: a LanguageTool/Google-Translate modifier fixes or
  adds a field in the ``before`` phase, and the template then renders the
  prompt from the modified values.

Requests are bounded by an internal semaphore (``model.concurrency``) so
the number of in-flight LLM calls is capped independently of how many
records the runner processes at once, and each call is retried with
exponential backoff via tenacity.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Type

import hydra
from jinja2 import Environment, FileSystemLoader, StrictUndefined, meta
from loguru import logger
from omegaconf import OmegaConf
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from DataCurator.pipeline.stage import Stage, StageContext


def _as_container(value: Any) -> Any:
    """Convert an OmegaConf node to a plain Python container; pass others through."""
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


class LLMStage(Stage):
    """Run a structured-output chat completion per record against vLLM/OpenAI."""

    def __init__(
        self,
        model: Any,
        response_model: Any,
        input_field: str = "messages",
        prompt: Any = None,
        prompts_dir: str = "prompts",
        output_field: str = "llm_output",
        name: Optional[str] = None,
        modifiers: Any = None,
    ) -> None:
        """Resolve the response schema and pre-build the request parameters."""
        super().__init__(name=name, modifiers=modifiers)
        self.input_field = input_field
        self.output_field = output_field
        self.response_model: Type[BaseModel] = self._resolve_response_model(response_model)
        self._prompt_templates = self._load_prompt(prompt, prompts_dir)

        cfg = _as_container(model) or {}
        self.api_base: Optional[str] = cfg.get("api_base")
        self.api_key: str = cfg.get("api_key") or "EMPTY"
        self.concurrency: int = int(cfg.get("concurrency", 8))
        self.retries: int = int(cfg.get("retries", 3))
        self._request_kwargs = self._build_request_kwargs(cfg)

        # Created lazily inside the event loop on first use.
        self._client: Optional[AsyncOpenAI] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    # -- setup helpers ------------------------------------------------------
    @staticmethod
    def _resolve_response_model(response_model: Any) -> Type[BaseModel]:
        """Accept a Pydantic class or a dotted import path and return the class."""
        cls = response_model if isinstance(response_model, type) else hydra.utils.get_class(response_model)
        if not issubclass(cls, BaseModel):
            raise TypeError(f"response_model {cls!r} is not a pydantic BaseModel")
        return cls

    @staticmethod
    def _load_prompt(prompt: Any, prompts_dir: str) -> Optional[List[Dict[str, Any]]]:
        """Load a templated prompt from files under ``prompts_dir``.

        Each entry is ``{role, template, required}`` where ``template`` names
        a Jinja2 file in ``prompts_dir`` and ``required`` is the set of
        variables the template reads but never assigns itself — i.e. the
        values the pipeline must populate. Templates are loaded eagerly so a
        missing file fails when the pipeline is built, not mid-run, and
        ``StrictUndefined`` makes any unpopulated variable that slips past the
        up-front check (e.g. a missing nested attribute) raise at render time
        instead of silently rendering as an empty string.
        """
        prompt = _as_container(prompt)
        if not prompt:
            return None
        env = Environment(
            loader=FileSystemLoader(prompts_dir),
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )
        compiled: List[Dict[str, Any]] = []
        for message in prompt:
            name = message["template"]
            source = env.loader.get_source(env, name)[0]
            # Variables referenced but not assigned in the template, minus the
            # environment's own globals (range, dict, ...), are exactly what the
            # pipeline is expected to supply via the context.
            required = meta.find_undeclared_variables(env.parse(source)) - set(env.globals)
            compiled.append(
                {"role": message["role"], "template": env.get_template(name), "required": required}
            )
        return compiled

    @staticmethod
    def _build_request_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Map a model config into OpenAI request kwargs, routing vLLM extras.

        Native OpenAI fields go top-level; vLLM-only sampling controls and
        the Qwen3 thinking toggle are nested into ``extra_body``, which the
        config's own ``extra_body`` map then overrides.
        """
        kwargs: Dict[str, Any] = {"model": cfg["name"]}
        for key in ("temperature", "max_tokens", "top_p", "presence_penalty"):
            if cfg.get(key) is not None:
                kwargs[key] = cfg[key]

        extra_body: Dict[str, Any] = {}
        for key in ("top_k", "min_p", "repetition_penalty"):
            if cfg.get(key) is not None:
                extra_body[key] = cfg[key]
        if cfg.get("enable_thinking") is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": cfg["enable_thinking"]}
        extra_body.update(cfg.get("extra_body") or {})
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _ensure_client(self) -> None:
        """Lazily build the async client and semaphore inside the running loop."""
        if self._client is None:
            self._client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)

    # -- execution ----------------------------------------------------------
    def _build_messages(self, context: StageContext) -> List[Dict[str, Any]]:
        """Render the templated prompt, or read a pre-built conversation.

        Before rendering, every variable a template references must be
        present in the context; a template variable the pipeline never
        populated would otherwise render as an empty string and silently
        corrupt the prompt, so raise instead.
        """
        if self._prompt_templates is not None:
            variables = context.to_dict()
            messages: List[Dict[str, Any]] = []
            for m in self._prompt_templates:
                missing = m["required"] - variables.keys()
                if missing:
                    raise KeyError(
                        f"{self.name}: prompt template references variable(s) "
                        f"{sorted(missing)!r} that the pipeline did not populate; "
                        f"available fields: {sorted(variables)!r}"
                    )
                messages.append({"role": m["role"], "content": m["template"].render(**variables)})
            return messages
        if self.input_field not in context:
            raise KeyError(f"{self.name}: no prompt configured and field {self.input_field!r} missing")
        return _as_container(context[self.input_field])

    async def process(self, context: StageContext) -> None:
        """Build the conversation, call the model, and store the parsed result."""
        messages = self._build_messages(context)
        parsed = await self._complete(messages)
        context.set(self.output_field, parsed.model_dump())

    async def _complete(self, messages: List[Dict[str, Any]]) -> BaseModel:
        """Call the model with bounded concurrency and exponential-backoff retries."""
        self._ensure_client()
        assert self._semaphore is not None
        async with self._semaphore:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.retries),
                wait=wait_exponential(multiplier=1, min=2, max=30),
                reraise=True,
            ):
                with attempt:
                    parsed = await self._call_llm(messages)
                    if parsed is None:
                        raise ValueError("model returned no parseable structured output")
                    return parsed
        raise RuntimeError("unreachable: retry loop exited without returning")  # pragma: no cover

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> Optional[BaseModel]:
        """Issue a single structured-output completion and return the parsed model."""
        assert self._client is not None
        completion = await self._client.chat.completions.parse(
            messages=messages,
            response_format=self.response_model,
            **self._request_kwargs,
        )
        return completion.choices[0].message.parsed
