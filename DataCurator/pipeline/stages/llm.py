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
from DataCurator.pipeline.validators import ResponseValidationError, ResponseValidator


class UnparseableOutputError(Exception):
    """The model's reply could not be parsed into the response schema.

    Raised when the structured-output parse yields no object — typically a
    refusal or a reply that does not satisfy the schema. It is retried like any
    other error (a fresh sample may parse), and is a natural entry for
    ``runner.flag_on_errors`` so a record that still fails after retries is
    flagged rather than aborting the whole run.
    """


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
        validators: Any = None,
        keep_rejected_output: bool = False,
        rejected_output_field: str = "rejected_output",
    ) -> None:
        """Resolve the response schema and pre-build the request parameters."""
        super().__init__(name=name, modifiers=modifiers)
        self.input_field = input_field
        self.output_field = output_field
        # When a stage-level validator rejects an output across all retries (so
        # the record is flagged), optionally keep the parsed output that failed
        # the guard under `rejected_output_field` instead of discarding it.
        self.keep_rejected_output = keep_rejected_output
        self.rejected_output_field = rejected_output_field
        self.response_model: Type[BaseModel] = self._resolve_response_model(response_model)
        self._prompt_templates = self._load_prompt(prompt, prompts_dir)
        # Stage-level guards over the parsed output, run inside the retry loop so
        # a failed guard triggers a fresh sample. Hydra instantiates the nested
        # `_target_` entries, so the list arrives as built validators.
        self.validators: List[ResponseValidator] = list(validators or [])

        cfg = _as_container(model) or {}
        self.api_base: Optional[str] = cfg.get("api_base")
        self.api_key: str = cfg.get("api_key") or "EMPTY"
        self.concurrency: int = int(cfg.get("concurrency", 8))
        self.retries: int = int(cfg.get("retries", 3))
        timeout = cfg.get("timeout")
        self.timeout: Optional[float] = float(timeout) if timeout is not None else None
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
            kwargs: Dict[str, Any] = {"base_url": self.api_base, "api_key": self.api_key}
            if self.timeout is not None:
                kwargs["timeout"] = self.timeout
            self._client = AsyncOpenAI(**kwargs)
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
        """Build the conversation, call the model, and store the parsed result.

        When a validator (guard) rejects the output across all retries and
        ``keep_rejected_output`` is set, stash the parsed output that failed the
        guard under ``rejected_output_field`` before re-raising, so it survives
        on the flagged record (carried out via ``Stage.run``'s context).
        """
        messages = self._build_messages(context)
        try:
            parsed = await self._complete(messages, context)
        except ResponseValidationError as exc:
            output = getattr(exc, "output", None)
            if self.keep_rejected_output and output is not None:
                context.set(self.rejected_output_field, output)
            raise
        context.set(self.output_field, parsed.model_dump())

    async def _complete(self, messages: List[Dict[str, Any]], context: StageContext) -> BaseModel:
        """Call the model with bounded concurrency and exponential-backoff retries.

        Stage-level validators run alongside the call within the retried block,
        so a parse that satisfies the schema but fails a guard (e.g. a truncated
        rewrite far shorter than its reference) is resampled like any other
        failure; the validation error re-raises once retries are exhausted.
        """
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
                    self._validate(parsed, context)
                    return parsed
        raise RuntimeError("unreachable: retry loop exited without returning")  # pragma: no cover

    def _validate(self, parsed: BaseModel, context: StageContext) -> None:
        """Run every attached validator over the parsed output; raise on failure."""
        if not self.validators:
            return
        data = parsed.model_dump()
        for validator in self.validators:
            try:
                validator.validate(data, context)
            except ResponseValidationError as exc:
                # Carry the parsed output that failed the guard so `process` can
                # keep it on a flagged record when `keep_rejected_output` is set.
                exc.output = data
                raise

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> BaseModel:
        """Issue a single structured-output completion and return the parsed model.

        Raise :class:`UnparseableOutputError` when the parse yields no object,
        surfacing the model's refusal reason when it supplied one.
        """
        assert self._client is not None
        completion = await self._client.chat.completions.parse(
            messages=messages,
            response_format=self.response_model,
            **self._request_kwargs,
        )
        message = completion.choices[0].message
        if message.parsed is None:
            detail = f": {message.refusal}" if message.refusal else ""
            raise UnparseableOutputError(f"model returned no parseable structured output{detail}")
        return message.parsed
