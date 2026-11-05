"""Generic LLM stage for any OpenAI-compatible endpoint (e.g. vLLM).

``LLMStage`` runs one chat completion per record and parses the reply into
a Pydantic model (structured output). The schema reaches the model one of two
ways, picked by the model preset's ``structured_output`` (see
``STRUCTURED_OUTPUT_MODES``), since which one works is a property of the
backend, not of the stage. The conversation can be supplied two ways:

* pre-built — read a list of ``{role, content}`` messages straight from a
  context field (``input_field``); or
* templated — render a ``prompt`` (a list of ``{role, template | text}``
  entries: a path to a Jinja2 file, or the template inline) against the
  record's fields. Prompt files are experiment assets kept outside git; see
  :mod:`DataCurator.pipeline.prompts`. This is the path that makes modifiers
  useful: a LanguageTool/Google-Translate modifier fixes or adds a field in the
  ``before`` phase, and the template then renders the prompt from the
  modified values.

Requests are bounded by an internal semaphore (``model.concurrency``) so
the number of in-flight LLM calls is capped independently of how many
records the runner processes at once, and each call is retried with
exponential backoff via tenacity.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Type

import hydra
from jinja2 import Environment, StrictUndefined, meta
from loguru import logger
from omegaconf import OmegaConf
from openai import AsyncOpenAI, LengthFinishReasonError, pydantic_function_tool
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from DataCurator.pipeline.stage import Stage, StageContext
from DataCurator.pipeline.errors import ServiceUnavailableError
from DataCurator.pipeline.prompts import load_message
from DataCurator.pipeline.repairs import OutputRepair
from DataCurator.pipeline.validators import ResponseValidationError, ResponseValidator, repeated_ngram

# How the response schema is delivered. ``json`` sends it as ``response_format``
# (a json_schema grammar, the default); ``tool`` sends the identical strict
# schema as a forced function call, for a backend whose grammar never engages on
# a plain answer -- Muse-Glimmer only leaves its reasoning phase when a tool
# channel opens, so ``json`` is silently inert there (the reply comes back as
# prose and no error is raised).
STRUCTURED_OUTPUT_MODES = ("json", "tool")

# Correction turn appended (after the failed answer, replayed as an assistant
# turn) when a validator rejects an output and the stage reprompts.
_CORRECTION_TEMPLATE = """Your previous answer (shown above) was rejected by automated validation:
- {feedback}

Return a corrected answer as a JSON object with exactly the same schema as before.
Fix only the problem described; keep everything else in your answer unchanged."""

# What a guard failure does once every correction round and retry is spent:
# ``flag`` raises (the runner flags the record), ``keep`` stores the rejected
# output and continues — config then routes on ``guard_failure_field``.
GUARD_FAILURE_MODES = ("flag", "keep")

# Correction turn after an answer cut off at ``max_tokens`` (the truncated
# text, clipped, is replayed as the assistant turn before it).
_TRUNCATION_TEMPLATE = """Your previous answer (shown above, clipped) was cut off at the output length limit before the JSON was complete{loop}.
Answer again from scratch as a complete JSON object with the same schema. Write each part of the text exactly once and keep the answer concise."""
_TRUNCATION_CLIP_CHARS = 2000
# A trigram repeated more often than this in a truncated answer is named in the
# correction as the loop (the RepetitionValidator default, no reference).
_TRUNCATION_LOOP_MIN_COUNT = 6


def truncated_text(exc: LengthFinishReasonError) -> str:
    """The partial answer a length-truncated completion carried (content or tool arguments)."""
    try:
        message = exc.completion.choices[0].message
    except (AttributeError, IndexError):
        return ""
    if message.content:
        return message.content
    for call in message.tool_calls or []:
        arguments = getattr(getattr(call, "function", None), "arguments", None)
        if arguments:
            return arguments
    return ""


def truncation_turns(exc: LengthFinishReasonError) -> List[Dict[str, str]]:
    """The clipped truncated answer as an assistant turn, plus the correction.

    When the partial answer repeats a trigram ``_TRUNCATION_LOOP_MIN_COUNT``
    times or more, the correction names it — the usual cause of hitting the
    limit is a generation loop, not a genuinely long answer.
    """
    text = truncated_text(exc)
    gram, count, _ = repeated_ngram(text)
    loop = (
        f', apparently because it fell into a loop repeating "{" ".join(gram)}" {count} times'
        if gram is not None and count >= _TRUNCATION_LOOP_MIN_COUNT
        else ""
    )
    clipped = text if len(text) <= _TRUNCATION_CLIP_CHARS else text[:_TRUNCATION_CLIP_CHARS] + " …"
    return [
        {"role": "assistant", "content": clipped or "(empty)"},
        {"role": "user", "content": _TRUNCATION_TEMPLATE.format(loop=loop)},
    ]


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
        output_field: str = "llm_output",
        name: Optional[str] = None,
        modifiers: Any = None,
        validators: Any = None,
        keep_rejected_output: bool = False,
        rejected_output_field: str = "rejected_output",
        reprompt_attempts: int = 0,
        repairs: Any = None,
        on_guard_failure: str = "flag",
        guard_failure_field: Optional[str] = None,
        reprompt_on_truncation: bool = False,
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
        # Correction rounds on validator failures carrying feedback: the failed
        # answer is replayed as an assistant turn plus a user turn describing
        # what was wrong, and the model is re-asked. 0 disables reprompting —
        # every failure is blind-resampled as before. Connection/parse failures
        # (and feedback-less guard failures) are never reprompted.
        self.reprompt_attempts = int(reprompt_attempts)
        # An answer cut off at max_tokens (openai.LengthFinishReasonError) gets a
        # correction round from the same budget instead of a blind resample —
        # usually it is a generation loop. Needs reprompt_attempts > 0.
        self.reprompt_on_truncation = bool(reprompt_on_truncation)
        # Guard failure with every round and retry spent: flag the record, or
        # keep the rejected output and write the error to guard_failure_field so
        # an after-modifier can route the record (e.g. to a later fix stage).
        if on_guard_failure not in GUARD_FAILURE_MODES:
            raise ValueError(
                f"on_guard_failure must be {'|'.join(GUARD_FAILURE_MODES)}, got {on_guard_failure!r}"
            )
        self.on_guard_failure = on_guard_failure
        self.guard_failure_field = guard_failure_field or f"{output_field}_guard_failure"
        # Deterministic fixes applied to each parsed output before the
        # validators see it; Hydra instantiates the `_target_` entries.
        self.repairs: List[OutputRepair] = list(repairs or [])
        self.response_model: Type[BaseModel] = self._resolve_response_model(response_model)
        self._prompt_templates = self._load_prompt(prompt)
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
        self.structured_output: str = str(cfg.get("structured_output") or "json").lower()
        if self.structured_output not in STRUCTURED_OUTPUT_MODES:
            raise ValueError(
                f"model.structured_output must be "
                f"{'|'.join(STRUCTURED_OUTPUT_MODES)}, got {cfg.get('structured_output')!r}"
            )
        self._request_kwargs = self._build_request_kwargs(cfg)
        # Request kwargs per response schema. The strict-schema conversion is
        # done once per shape rather than per request; a batched stage asks for
        # one wrapper schema per id-tuple.
        self._schema_cache: Dict[Type[BaseModel], Dict[str, Any]] = {}
        # Held as an attribute so tests can swap in wait_fixed(0).
        self._retry_wait = wait_exponential(multiplier=1, min=2, max=30)

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
    def _load_prompt(prompt: Any) -> Optional[List[Dict[str, Any]]]:
        """Compile a templated prompt from inline text or ``.j2`` files.

        Each entry is ``{role, template}`` (a path to a Jinja2 file) or
        ``{role, text}`` (the template inline); the runner inlines files into
        ``text`` before the stage is built, see
        :func:`~DataCurator.pipeline.prompts.inline_prompt_files`. Every entry
        compiles to ``{role, template, required}``, where ``required`` is the
        set of variables the template reads but never assigns itself — i.e.
        the values the pipeline must populate. Templates are compiled eagerly
        so a missing file or bad syntax fails when the pipeline is built, not
        mid-run, and ``StrictUndefined`` makes any unpopulated variable that
        slips past the up-front check (e.g. a missing nested attribute) raise
        at render time instead of silently rendering as an empty string.
        """
        prompt = _as_container(prompt)
        if not prompt:
            return None
        env = Environment(keep_trailing_newline=True, undefined=StrictUndefined)
        compiled: List[Dict[str, Any]] = []
        for message in prompt:
            source = load_message(message)["text"]
            # Variables referenced but not assigned in the template, minus the
            # environment's own globals (range, dict, ...), are exactly what the
            # pipeline is expected to supply via the context.
            required = meta.find_undeclared_variables(env.parse(source)) - set(env.globals)
            compiled.append(
                {"role": message["role"], "template": env.from_string(source), "required": required}
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
            if self._keepable(exc):
                self._keep_guard_failure(context, output, exc)
                return
            if self.keep_rejected_output and output is not None:
                context.set(self.rejected_output_field, output)
            raise
        context.set(self.output_field, parsed.model_dump())

    def _keepable(self, exc: BaseException) -> bool:
        """True when ``on_guard_failure: keep`` applies to this final error: a
        guard rejection carrying the parsed output it rejected. Outages, parse
        and length errors, and batch misalignments (no output of the record's
        own) never qualify."""
        return (
            self.on_guard_failure == "keep"
            and isinstance(exc, ResponseValidationError)
            and getattr(exc, "output", None) is not None
        )

    def _keep_guard_failure(
        self, context: StageContext, output: Dict[str, Any], exc: ResponseValidationError
    ) -> None:
        """Store the guard-rejected output as the stage result, plus the reason."""
        logger.info(f"{self.name}: keeping guard-rejected output ({exc})")
        context.set(self.output_field, output)
        context.set(self.guard_failure_field, str(exc))

    @property
    def _reprompts_truncation(self) -> bool:
        return self.reprompt_on_truncation and self.reprompt_attempts > 0

    async def _repair(self, parsed: BaseModel, context: StageContext) -> BaseModel:
        """Run every attached repair over the parsed output, re-validating the schema."""
        if not self.repairs:
            return parsed
        data = parsed.model_dump()
        for repair in self.repairs:
            data = await repair.repair(data, context)
        return type(parsed).model_validate(data)

    def _blind_retryable(self, exc: BaseException) -> bool:
        """Inner-budget policy: with reprompting off, retry everything (the
        historical behavior); with it on, everything except a feedback-carrying
        guard failure, which escapes to the correction loop instead.

        A guard whose *service* is down is the one exception to "retry
        everything": its client already spent a bounded retry budget, and the
        model's output was never the problem, so another generation is pure
        waste. It propagates instead, and the distributed worker reads it as
        connection-class. LLM-endpoint transport errors are unaffected —
        neither ``openai.APIConnectionError`` nor ``httpx.TransportError``
        derives from the builtin ``ConnectionError``.
        """
        if isinstance(exc, ServiceUnavailableError):
            return False
        if isinstance(exc, LengthFinishReasonError) and self._reprompts_truncation:
            return False  # escapes to a truncation correction round
        return self.reprompt_attempts == 0 or not (
            isinstance(exc, ResponseValidationError) and exc.feedback is not None
        )

    @staticmethod
    def _correction_turns(output: Dict[str, Any], feedback: str) -> List[Dict[str, str]]:
        """The rejected answer replayed as an assistant turn, plus the correction."""
        return [
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
            {"role": "user", "content": _CORRECTION_TEMPLATE.format(feedback=feedback)},
        ]

    def _record_rounds(self, context: StageContext, rounds: int) -> None:
        """Record how many correction rounds this record cost — accepted (0 =
        first try) or flagged — so the count is a reliable column rather than a
        rare needle. Stages with reprompting off write nothing."""
        if self.reprompt_attempts:
            context.set(f"{self.output_field}_reprompt_rounds", rounds)

    async def _complete(self, messages: List[Dict[str, Any]], context: StageContext) -> BaseModel:
        """Call the model with bounded concurrency and exponential-backoff retries.

        Stage-level validators run alongside the call within the retried block,
        so a parse that satisfies the schema but fails a guard (e.g. a truncated
        rewrite far shorter than its reference) is resampled like any other
        failure; the validation error re-raises once retries are exhausted.

        With ``reprompt_attempts > 0``, a guard failure that carries feedback
        instead extends the conversation — the failed answer plus a correction
        turn — and re-asks, up to the budget; each correction round gets a
        fresh transport-retry budget, so the worst case is
        ``retries × (reprompt_attempts + 1)`` requests. Parse failures,
        LLM-endpoint connection failures and feedback-less guard failures stay
        on the blind path; a guard whose backing service is unreachable does
        not, and costs exactly one request (see :meth:`_blind_retryable`).
        """
        self._ensure_client()
        assert self._semaphore is not None
        convo = list(messages)  # copy: an `input_field` list is the record's own
        rounds = 0
        try:
            async with self._semaphore:
                while True:
                    try:
                        async for attempt in AsyncRetrying(
                            stop=stop_after_attempt(self.retries),
                            wait=self._retry_wait,
                            retry=retry_if_exception(self._blind_retryable),
                            reraise=True,
                        ):
                            with attempt:
                                parsed = await self._repair(await self._call_llm(convo), context)
                                await self._validate(parsed, context)
                                if rounds:
                                    logger.info(
                                        f"{self.name}: output accepted after {rounds} correction round(s)"
                                    )
                                return parsed
                    except ResponseValidationError as exc:
                        output = getattr(exc, "output", None)
                        if exc.feedback is None or output is None or rounds >= self.reprompt_attempts:
                            raise
                        rounds += 1
                        logger.debug(
                            f"{self.name}: correction round {rounds}/{self.reprompt_attempts}: {exc}"
                        )
                        convo = convo + self._correction_turns(output, exc.feedback)
                    except LengthFinishReasonError as exc:
                        if not self._reprompts_truncation or rounds >= self.reprompt_attempts:
                            raise
                        rounds += 1
                        logger.debug(
                            f"{self.name}: correction round {rounds}/{self.reprompt_attempts}: "
                            f"output truncated at the length limit"
                        )
                        convo = convo + truncation_turns(exc)
        finally:
            # Stamped on the way out, not only on the way to a result: a record
            # flagged after the budget ran out is exactly the one whose round
            # count you want, and the partial context rides onto it (`Stage.run`).
            self._record_rounds(context, rounds)

    async def _validate(self, parsed: BaseModel, context: StageContext) -> None:
        """Run every attached validator over the parsed output; raise on failure."""
        if not self.validators:
            return
        data = parsed.model_dump()
        for validator in self.validators:
            try:
                await validator.validate(data, context)
            except ResponseValidationError as exc:
                # Carry the parsed output that failed the guard so `process` can
                # keep it on a flagged record when `keep_rejected_output` is set.
                exc.output = data
                raise

    # -- structured output ---------------------------------------------------
    def _schema_kwargs(
        self, schema: Type[BaseModel], description: Optional[str] = None
    ) -> Dict[str, Any]:
        """The request kwargs pinning the reply to ``schema``, per mode.

        Both modes send the same strict JSON schema — the SDK builds it with
        ``to_strict_json_schema`` either way — so switching modes changes how
        the schema is delivered, never what it says. ``description`` overrides
        the function description the model reads in tool mode (the schema's own
        docstring by default); it is ignored in ``json`` mode.
        """
        kwargs = self._schema_cache.get(schema)
        if kwargs is None:
            if self.structured_output == "tool":
                tool = pydantic_function_tool(schema, description=description)
                kwargs = {
                    "tools": [tool],
                    # Named tool choice: the model cannot answer any other way,
                    # which is what makes this equivalent to structured output
                    # rather than a suggestion.
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": tool["function"]["name"]},
                    },
                }
            else:
                kwargs = {"response_format": schema}
            self._schema_cache[schema] = kwargs
        return kwargs

    @staticmethod
    def _parsed_tool_call(message: Any, schema: Type[BaseModel]) -> BaseModel:
        """The forced function call's arguments, as ``schema``.

        A reply carrying no tool call at all is this mode's signature failure —
        the backend ignored ``tool_choice`` and answered in prose — so it says
        so, with a snippet of what arrived instead.
        """
        if not message.tool_calls:
            snippet = (message.content or "")[:200] or message.refusal
            detail = f": {snippet!r}" if snippet else ""
            raise UnparseableOutputError(f"model answered without calling the forced tool{detail}")
        function = message.tool_calls[0].function
        parsed = getattr(function, "parsed_arguments", None)
        # The SDK auto-parses only a call whose name matches a tool we sent; one
        # answering under a different name still carries the arguments we asked
        # for, so validate them ourselves rather than resampling.
        return parsed if parsed is not None else schema.model_validate_json(function.arguments)

    async def _parse_completion(
        self,
        messages: List[Dict[str, Any]],
        schema: Type[BaseModel],
        description: Optional[str] = None,
    ) -> BaseModel:
        """One completion decoded into ``schema`` under the configured mode.

        Raise :class:`UnparseableOutputError` whenever nothing usable comes
        back — a refusal, a reply that ignored the schema, or arguments the
        schema rejects (in tool mode the SDK validates those inside ``parse``,
        so it raises there instead of yielding ``None``) — so the caller's retry
        loop resamples rather than the run aborting. Truncation is untouched:
        the SDK raises ``openai.LengthFinishReasonError`` on
        ``finish_reason == "length"`` before it looks at content or tool calls.
        """
        assert self._client is not None
        try:
            completion = await self._client.chat.completions.parse(
                messages=messages,
                **self._schema_kwargs(schema, description),
                **self._request_kwargs,
            )
            message = completion.choices[0].message
            parsed = (
                self._parsed_tool_call(message, schema)
                if self.structured_output == "tool"
                else message.parsed
            )
        except (ValidationError, json.JSONDecodeError) as exc:
            raise UnparseableOutputError(f"model returned output the schema rejects: {exc}") from exc
        if parsed is None:
            detail = f": {message.refusal}" if message.refusal else ""
            raise UnparseableOutputError(f"model returned no parseable structured output{detail}")
        return parsed

    async def _call_llm(self, messages: List[Dict[str, Any]]) -> BaseModel:
        """Issue a single structured-output completion and return the parsed model."""
        return await self._parse_completion(messages, self.response_model)
