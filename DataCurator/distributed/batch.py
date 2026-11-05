"""Batched LLM requests: N records per completion (wiki/Distributed.md).

The render/split/validate machinery lives at the stage layer —
``BatchedLLMStage`` subclasses the untouched ``LLMStage`` — while only the
grouping lives in the worker's record loop. The win is prompt economy (one
shared prefix processed once, per-request overhead divided by N), not server
parallelism; vLLM already batches across requests.

Per-record semantics are preserved around the group: modifiers and
validators run per item, only failing items are re-sent (re-batched
together — or, with ``reprompt_attempts > 0``, re-asked in the same
conversation under their original ids with a correction turn describing the
validator's findings; connection/parse failures and feedback-less guard
failures never see a correction turn), and a dead batch resolves per
``batch.on_failure``:

- ``flag``   — every record of the dead batch flags (batchmates pay for a
  poison item);
- ``single`` — each record retries as a 1-item batch first, so only records
  that fail on their own merits flag.

The prompt must be *written for batching*: templates receive ``items`` (the
list of per-record contexts, each carrying its ``item_id``) and iterate over
it, so one file handles every group size including the single-record fallback.

Outputs are matched to inputs by an echoed ``item_id``, not by position: each
response object must open with the ITEM number it answers, a truthfully
misordered batch is realigned, and a missing or duplicated id fails only the
affected positions (``BatchItemMismatchError``) — positional zip alone let a
model that duplicated a batchmate's answer corrupt records silently.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Type

from loguru import logger
from openai import LengthFinishReasonError
from pydantic import BaseModel, Field, create_model
from pydantic import conlist

# Reused, never modified.
from DataCurator.pipeline.stage import ModifierPhase, StageContext, _CONTEXT_ATTR
from DataCurator.pipeline.stages.llm import LLMStage, UnparseableOutputError, truncation_turns
from DataCurator.pipeline.validators import ResponseValidationError

# (output-or-None, error-or-None) per input record, in input order.
GroupOutcome = Tuple[Optional[dict], Optional[Exception]]

# Echo field the model must repeat per batch item; injected into each item's
# render context and stripped from the output before it reaches the record.
ITEM_ID_FIELD = "item_id"

# Correction turn appended (after the failed batch answer, replayed as an
# assistant turn) when validators reject some items and the stage reprompts.
# Ids stay the ones the items carried in the original batch prompt.
# The function description the model reads when the stage runs in `tool` mode
# (`model.structured_output`): the wrappers are built per id-tuple with
# `create_model`, so they carry no docstring of their own.
_BATCH_TOOL_DESCRIPTION = "Return exactly one object per requested ITEM, each echoing its item_id."

_BATCH_CORRECTION_TEMPLATE = """Some items in your previous answer (shown above) were rejected by automated validation:

{problems}

Return a JSON object of the form {{"items": [...]}} containing corrected objects for ONLY these items: {ids} — exactly one object per listed item, each carrying its original "item_id". Do not include items that were already accepted. Fix only the problems listed; keep everything else about each answer unchanged."""


class BatchItemMismatchError(ResponseValidationError):
    """A batch item's echoed id doesn't identify it (missing or duplicated).

    Subclasses :class:`ResponseValidationError` so the runner's existing
    ``flag_on_errors`` entry covers it, while the distinct ``error_type``
    keeps misalignments visible in flagged-record metrics.
    """


class BatchedLLMStage(LLMStage):
    """LLMStage that packs a group of records into one completion."""

    def __init__(self, *args: Any, batch_size: int = 5, batch_on_failure: str = "flag", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if batch_on_failure not in ("flag", "single"):
            raise ValueError(f"batch.on_failure must be flag|single, got {batch_on_failure!r}")
        if ITEM_ID_FIELD in self.response_model.model_fields:
            raise ValueError(
                f"{self.name}: response model {self.response_model.__name__} already has a "
                f"{ITEM_ID_FIELD!r} field, which batching reserves for the item echo"
            )
        self.batch_size = int(batch_size)
        self.batch_on_failure = batch_on_failure
        self._item_models: Dict[Tuple[str, ...], Type[BaseModel]] = {}
        self._wrappers: Dict[Tuple[str, ...], Type[BaseModel]] = {}

    # -- schema: the response is the list -------------------------------------
    @staticmethod
    def _contiguous_ids(n: int) -> Tuple[str, ...]:
        return tuple(str(k) for k in range(1, n + 1))

    def _item_model_for_ids(self, ids: Tuple[str, ...]) -> Type[BaseModel]:
        """The response model plus a leading ``item_id`` echo.

        The id is an enum of the valid ids, *not* forced per position: the
        model chooses it freely, so a reordered or duplicated item is
        detectable — a grammar that stamped the right id over wrong content
        would hide the swap. It is declared first so it decodes before the
        content: the model commits to which ITEM it answers before generating
        it. A correction turn constrains the enum to the ids being redone."""
        item_model = self._item_models.get(ids)
        if item_model is None:
            suffix = str(len(ids)) if ids == self._contiguous_ids(len(ids)) else "_" + "_".join(ids)
            item_model = create_model(
                f"{self.response_model.__name__}Item{suffix}",
                **{ITEM_ID_FIELD: (Literal[ids], Field(description="The ITEM number this object answers."))},
                **{name: (f.annotation, f) for name, f in self.response_model.model_fields.items()},
            )
            self._item_models[ids] = item_model
        return item_model

    def _wrapper_for_ids(self, ids: Tuple[str, ...]) -> Type[BaseModel]:
        """``{items: [{item_id, <response_model>}] × len(ids)}`` — exactly one
        item per requested id, enforced in the decoding schema where the
        backend supports array bounds and re-checked after the parse
        regardless."""
        wrapper = self._wrappers.get(ids)
        if wrapper is None:
            n = len(ids)
            prefix = f"BatchOf{n}" if ids == self._contiguous_ids(n) else "Batch_" + "_".join(ids) + "_"
            wrapper = create_model(
                f"{prefix}{self.response_model.__name__}",
                items=(conlist(self._item_model_for_ids(ids), min_length=n, max_length=n), ...),
            )
            self._wrappers[ids] = wrapper
        return wrapper

    def _item_model_for(self, n: int) -> Type[BaseModel]:
        return self._item_model_for_ids(self._contiguous_ids(n))

    def _wrapper_for(self, n: int) -> Type[BaseModel]:
        return self._wrapper_for_ids(self._contiguous_ids(n))

    @staticmethod
    def _align_items(
        parsed_items: List[BaseModel], expected_ids: Optional[Sequence[str]] = None
    ) -> List[Optional[BaseModel]]:
        """Reassign items to positions by their echoed id.

        Position ``k`` receives the item whose ``item_id`` is
        ``expected_ids[k]`` (``str(k + 1)`` by default) when that id appears
        exactly once; a missing or duplicated id leaves ``None`` so only those
        positions fail. A truthful out-of-order batch is thereby recovered,
        not rejected."""
        expected = (
            list(expected_ids)
            if expected_ids is not None
            else [str(k + 1) for k in range(len(parsed_items))]
        )
        by_id: Dict[str, List[BaseModel]] = {}
        for item in parsed_items:
            by_id.setdefault(getattr(item, ITEM_ID_FIELD), []).append(item)
        aligned: List[Optional[BaseModel]] = []
        for id_ in expected:
            candidates = by_id.get(id_, [])
            aligned.append(candidates[0] if len(candidates) == 1 else None)
        echoed = [getattr(item, ITEM_ID_FIELD) for item in parsed_items]
        if echoed != expected:
            logger.warning(f"batch items arrived misordered: echoed ids {echoed}")
        return aligned

    def _build_batch_messages(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Render the batch-aware prompt. Templates receive ``items`` only."""
        if self._prompt_templates is None:
            raise ValueError(f"{self.name}: batching requires a templated prompt")
        messages: List[Dict[str, Any]] = []
        for m in self._prompt_templates:
            missing = m["required"] - {"items"}
            if missing:
                raise KeyError(
                    f"{self.name}: a batch prompt renders from `items` alone, but the "
                    f"template references {sorted(missing)!r} — author it for batching"
                )
            messages.append({"role": m["role"], "content": m["template"].render(items=items)})
        return messages

    async def _call_llm_batch(
        self, messages: List[Dict[str, Any]], ids: Sequence[str]
    ) -> List[BaseModel]:
        parsed = await self._parse_completion(
            messages, self._wrapper_for_ids(tuple(ids)), description=_BATCH_TOOL_DESCRIPTION
        )
        items = list(parsed.items)
        if len(items) != len(ids):  # belt over the schema's braces
            raise UnparseableOutputError(
                f"batch count mismatch: {len(items)} items for {len(ids)} inputs"
            )
        return items

    # -- the retried batch call --------------------------------------------------
    async def _complete_batch(
        self, contexts: List[StageContext]
    ) -> List[Tuple[Optional[BaseModel], Optional[Exception]]]:
        """One group through the model: only failing items are re-sent, passing
        items keep their outputs; nothing valid is resampled."""
        if self.reprompt_attempts <= 0:
            return await self._complete_batch_blind(contexts)
        return await self._complete_batch_convo(contexts)

    async def _complete_batch_blind(
        self, contexts: List[StageContext]
    ) -> List[Tuple[Optional[BaseModel], Optional[Exception]]]:
        """The historical loop: failing items are blindly re-batched (fresh
        ids, no feedback) until accepted or ``retries`` is exhausted."""
        self._ensure_client()
        assert self._semaphore is not None
        results: List[Tuple[Optional[BaseModel], Optional[Exception]]] = [
            (None, None) for _ in contexts
        ]
        pending = list(range(len(contexts)))
        last_error: Dict[int, Exception] = {}
        for attempt in range(1, self.retries + 1):
            items = []
            for pos, i in enumerate(pending, start=1):
                item = contexts[i].to_dict()  # shallow copy; the context stays clean
                item[ITEM_ID_FIELD] = str(pos)
                items.append(item)
            try:
                messages = self._build_batch_messages(items)
                async with self._semaphore:
                    parsed_items = await self._call_llm_batch(
                        messages, self._contiguous_ids(len(items))
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # whole-request failure: retry the subset
                for i in pending:
                    last_error[i] = exc
                if attempt < self.retries:
                    await asyncio.sleep(min(2 * 2 ** (attempt - 1), 30))
                continue
            still_failing: List[int] = []
            for pos, (i, parsed) in enumerate(zip(pending, self._align_items(parsed_items)), start=1):
                if parsed is None:
                    last_error[i] = BatchItemMismatchError(
                        f"no uniquely echoed {ITEM_ID_FIELD} {str(pos)!r} in the batch response"
                    )
                    still_failing.append(i)
                    continue
                try:
                    parsed = await self._repair(parsed, contexts[i])
                    await self._validate(parsed, contexts[i])
                    results[i] = (parsed, None)
                    # A reprompt-enabled stage that fell back here (feedback-less
                    # failure) still records the count — 0, since no correction
                    # round rescued the item — so every enabled-stage record
                    # carries the field. Disabled stages write nothing.
                    self._record_rounds(contexts[i], 0)
                except ResponseValidationError as exc:
                    last_error[i] = exc
                    still_failing.append(i)
            pending = still_failing  # re-batched together, smaller group
            if not pending:
                break
        for i in pending:
            results[i] = (None, last_error[i])
            self._record_rounds(contexts[i], 0)  # flagged items carry it too
        return results

    async def _transport_call(
        self, messages: List[Dict[str, Any]], ids: Sequence[str]
    ) -> List[BaseModel]:
        """One conversation turn with blind transport retries: connection and
        parse failures re-send the SAME messages (never a correction turn),
        with today's backoff and a fresh ``retries`` budget per turn. With
        ``reprompt_on_truncation`` a length-truncated answer is raised at once,
        for the caller's truncation correction round."""
        assert self._semaphore is not None
        last: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                async with self._semaphore:
                    return await self._call_llm_batch(messages, ids)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if isinstance(exc, LengthFinishReasonError) and self._reprompts_truncation:
                    raise
                last = exc
                if attempt < self.retries:
                    await asyncio.sleep(min(2 * 2 ** (attempt - 1), 30))
        assert last is not None
        raise last

    @staticmethod
    def _batch_correction_turn(corrections: List[Tuple[str, str]]) -> str:
        problems = "\n\n".join(f"ITEM {id_}: {feedback}" for id_, feedback in corrections)
        ids = ", ".join(id_ for id_, _ in corrections)
        return _BATCH_CORRECTION_TEMPLATE.format(problems=problems, ids=ids)

    async def _complete_batch_convo(
        self, contexts: List[StageContext]
    ) -> List[Tuple[Optional[BaseModel], Optional[Exception]]]:
        """One conversation per group: items rejected by a validator get a
        correction turn (the failed batch answer replayed, plus per-item
        feedback under their ORIGINAL ids) and are re-asked as a subset, up to
        ``reprompt_attempts`` rounds. Passing items are never resampled.
        Items failing without feedback (an unreachable checker is not the
        model's fault) leave the conversation and go through the blind loop
        instead, exactly as with reprompting off."""
        self._ensure_client()
        assert self._semaphore is not None
        results: List[Tuple[Optional[BaseModel], Optional[Exception]]] = [
            (None, None) for _ in contexts
        ]
        pending = list(range(len(contexts)))
        ids = {i: str(pos + 1) for pos, i in enumerate(pending)}  # stable for the conversation
        items = []
        for i in pending:
            item = contexts[i].to_dict()  # shallow copy; the context stays clean
            item[ITEM_ID_FIELD] = ids[i]
            items.append(item)
        convo = self._build_batch_messages(items)
        blind: List[int] = []
        rounds = 0
        while pending:
            expected = tuple(ids[i] for i in pending)
            try:
                parsed_items = await self._transport_call(convo, expected)
            except LengthFinishReasonError as exc:
                if self._reprompts_truncation and rounds < self.reprompt_attempts:
                    rounds += 1
                    logger.debug(
                        f"{self.name}: correction round {rounds}/{self.reprompt_attempts} "
                        f"for item(s) {list(expected)}: output truncated at the length limit"
                    )
                    convo = convo + truncation_turns(exc)
                    continue
                for i in pending:
                    results[i] = (None, exc)
                    self._record_rounds(contexts[i], rounds)
                break
            except Exception as exc:  # turn's transport budget exhausted
                for i in pending:
                    results[i] = (None, exc)
                    self._record_rounds(contexts[i], rounds)
                break
            corrections: List[Tuple[str, str]] = []
            next_pending: List[int] = []
            for i, parsed in zip(pending, self._align_items(parsed_items, expected)):
                if parsed is None:
                    exc: ResponseValidationError = BatchItemMismatchError(
                        f"no uniquely echoed {ITEM_ID_FIELD} {ids[i]!r} in the batch response",
                        feedback=f'your reply did not contain exactly one object with '
                        f'"{ITEM_ID_FIELD}": "{ids[i]}"; return exactly one corrected '
                        f"object for ITEM {ids[i]}",
                    )
                else:
                    try:
                        parsed = await self._repair(parsed, contexts[i])
                        await self._validate(parsed, contexts[i])
                        results[i] = (parsed, None)
                        # This path only runs with reprompting on, so record the
                        # round count on every accepted item (0 = passed the first
                        # batch pass) — a reliable column, not a rare needle.
                        self._record_rounds(contexts[i], rounds)
                        continue
                    except ResponseValidationError as err:
                        exc = err
                if exc.feedback is None:
                    blind.append(i)  # infra-flavored: never a correction turn
                elif rounds >= self.reprompt_attempts:
                    results[i] = (None, exc)
                    self._record_rounds(contexts[i], rounds)  # budget spent, item flagged
                else:
                    corrections.append((ids[i], exc.feedback))
                    next_pending.append(i)
            if not next_pending:
                break
            rounds += 1
            logger.debug(
                f"{self.name}: correction round {rounds}/{self.reprompt_attempts} "
                f"for item(s) {[id_ for id_, _ in corrections]}"
            )
            convo = convo + [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"items": [it.model_dump() for it in parsed_items]}, ensure_ascii=False
                    ),
                },
                {"role": "user", "content": self._batch_correction_turn(corrections)},
            ]
            pending = next_pending
        if blind:  # today's fresh blind re-batch, full retries budget, no reprompting
            for i, res in zip(blind, await self._complete_batch_blind([contexts[i] for i in blind])):
                results[i] = res
        return results

    # -- the worker-facing entry ---------------------------------------------------
    async def run_group(self, records: List[Mapping]) -> List[GroupOutcome]:
        """Run one group end to end, per-record contracts intact.

        Modifiers fire per item (``before`` ahead of the render, ``after``
        once the output is split back); any per-record failure carries its
        partial context exactly as ``Stage.run`` does, so flagged records
        keep before-phase enrichments.
        """
        outcomes: List[Optional[GroupOutcome]] = [None] * len(records)
        contexts: List[StageContext] = []
        live: List[int] = []
        for i, record in enumerate(records):
            context = StageContext(dict(record))
            try:
                await self._apply_modifiers(context, ModifierPhase.BEFORE)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                setattr(exc, _CONTEXT_ATTR, context)
                outcomes[i] = (None, exc)
                continue
            contexts.append(context)
            live.append(i)

        if contexts:
            batch_results = await self._complete_batch(contexts)
            if self.batch_on_failure == "single":
                for j, (parsed, exc) in enumerate(batch_results):
                    if exc is not None:
                        logger.debug(f"{self.name}: retrying failed batch item as a 1-item batch")
                        batch_results[j] = (await self._complete_batch([contexts[j]]))[0]
            for j, i in enumerate(live):
                context = contexts[j]
                parsed, exc = batch_results[j]
                if exc is not None and self._keepable(exc):
                    kept = {k: v for k, v in exc.output.items() if k != ITEM_ID_FIELD}  # type: ignore[attr-defined]
                    self._keep_guard_failure(context, kept, exc)  # type: ignore[arg-type]
                elif exc is not None:
                    setattr(exc, _CONTEXT_ATTR, context)
                    outcomes[i] = (None, exc)
                    continue
                else:
                    context.set(self.output_field, parsed.model_dump(exclude={ITEM_ID_FIELD}))
                try:
                    await self._apply_modifiers(context, ModifierPhase.AFTER)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    setattr(exc, _CONTEXT_ATTR, context)
                    outcomes[i] = (None, exc)
                    continue
                outcomes[i] = (context.to_dict(), None)
        return outcomes  # type: ignore[return-value]
