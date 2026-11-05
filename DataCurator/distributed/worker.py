"""The distributed worker: a pure executor.

Pull → download → run stage → upload → report done. Topology-blind: it never
decides what stage N+1 is, never reads local pipeline config (behaviour
arrives in the message; only endpoint bindings come from its environment) and
carries no rerun logic. Losing a worker mid-chunk is routine: the lease
expires, the chunk redelivers elsewhere, the recompute is equivalent.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import signal
import socket
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import hydra.utils
import nats
import nats.errors
import yaml
from loguru import logger

# Reused, never modified: the per-record execution layer.
from DataCurator.pipeline.stage import Stage, errored_context

from . import s3io, wire

# Failures that read as "the endpoint is down", not "this record is bad".
try:
    import httpx
    import openai

    CONNECTION_ERRORS: tuple = (
        ConnectionError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        httpx.TransportError,
    )
except ImportError:  # pragma: no cover - both are hard deps today
    CONNECTION_ERRORS = (ConnectionError,)


class LeaseLost(Exception):
    """Our lease is (or may be) gone — stop silently; the chunk redelivers."""


class ChunkOutage(Exception):
    """Most of a chunk failed — an endpoint outage, not data. Nak, don't upload."""


# --------------------------------------------------------------------------- #
# Bindings: endpoints/secrets from the worker's env, injected at instantiate
# --------------------------------------------------------------------------- #
def load_bindings(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the worker bindings file (see bindings.example.yaml)."""
    path = path or os.environ.get("DC_BINDINGS_FILE", "bindings.yaml")
    if not os.path.exists(path):
        logger.warning(f"bindings file {path!r} not found — cpu-only worker (no LLM stages)")
        return {"models": {}}
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    data.setdefault("models", {})
    return data


def derive_capabilities(bindings: Dict[str, Any], version: str) -> set:
    """Capabilities are derived, never hand-listed: one ``model:`` tag per
    bound model, ``cpu``, and the version fence."""
    caps = {f"model:{name}" for name in bindings.get("models", {})}
    caps.add("cpu")
    caps.add(f"ver:{version}")
    return caps


def inject_bindings(stage_spec: Dict[str, Any], bindings: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``stage_spec`` with this worker's endpoints filled in."""
    spec = copy.deepcopy(stage_spec)
    stage = spec.get("stage") or {}
    model = stage.get("model")
    if isinstance(model, dict) and model.get("name"):
        bound = bindings.get("models", {}).get(model["name"])
        if not bound:
            # Capability routing should make this impossible — we only bind
            # consumers whose requires we satisfy. Fail fast, not silently.
            raise RuntimeError(f"no binding for model {model['name']!r} (see bindings.example.yaml)")
        model["api_base"] = bound["api_base"]
        model["api_key"] = bound.get("api_key", "EMPTY")
        # Serving knobs come from this worker's hardware. A spec rendered
        # before they were stripped may still carry them; the binding wins.
        for key in wire.MODEL_SERVING_KEYS:
            if bound.get(key) is not None:
                model[key] = bound[key]
    # Google Translate's endpoint and serving knobs, like the model's, come from
    # this box; unbound knobs fall back to the spec (older jobs) or the stage defaults.
    translate = bindings.get("google_translate") or {}
    for node in wire.google_translate_nodes(spec):
        for key in wire.GOOGLE_TRANSLATE_BINDING_KEYS:
            if not translate.get(key):
                raise RuntimeError(f"stage needs Google Translate but the worker has no {key} binding for it")
            node[key] = translate[key]
        for key in wire.GOOGLE_TRANSLATE_SERVING_KEYS:
            if translate.get(key) is not None:
                node[key] = translate[key]
    # Modifiers, validators and repairs alike may call out to a hosted service;
    # patch each spec that names one with this worker's endpoint for it.
    items = (
        list(stage.get("modifiers") or [])
        + list(stage.get("validators") or [])
        + list(stage.get("repairs") or [])
    )
    for item in items:
        if not isinstance(item, dict):
            continue
        target = str(item.get("_target_", ""))
        if target.endswith((
            "languagetool.LanguageToolModifier",
            "validators.LanguageToolValidator",
            "repairs.LanguageToolRepair",
        )):
            url = (bindings.get("languagetool") or {}).get("remote_server")
            if not url:
                raise RuntimeError("stage needs LanguageTool but the worker has no binding for it")
            item["remote_server"] = url
        elif target.endswith(("validators.MetricXValidator", "metricx.MetricXScoreModifier")):
            metricx = bindings.get("metricx") or {}
            if not metricx.get("remote_server"):
                raise RuntimeError("stage needs MetricX but the worker has no binding for it")
            item["remote_server"] = metricx["remote_server"]
            if metricx.get("api_token"):
                item["api_token"] = metricx["api_token"]
    return spec


def build_stage(
    stage_spec: Dict[str, Any], stage_id: str, bindings: Dict[str, Any]
) -> Tuple[Stage, List[Tuple[Any, str]], List[str]]:
    """Instantiate a stage + filters from a message spec, as build_pipeline does."""
    spec = inject_bindings(stage_spec, bindings)
    batch_cfg = spec.get("batch")
    if batch_cfg:
        # The `batch` toggle re-targets the configured LLMStage to its batched
        # subclass; the render/split/validate machinery lives there, only the
        # grouping lives in this worker's record loop.
        from DataCurator.pipeline.stages.llm import LLMStage

        target = hydra.utils.get_class(str(spec["stage"].get("_target_", "")))
        if not issubclass(target, LLMStage):
            raise TypeError(f"stage {stage_id!r}: batch requires an LLM stage, got {target.__name__}")
        spec["stage"]["_target_"] = "DataCurator.distributed.batch.BatchedLLMStage"
        spec["stage"]["batch_size"] = int(batch_cfg.get("size", 5))
        spec["stage"]["batch_on_failure"] = str(batch_cfg.get("on_failure", "flag"))
    stage = hydra.utils.instantiate(spec["stage"])
    if not isinstance(stage, Stage):
        raise TypeError(f"stage {stage_id!r} must instantiate a Stage, got {type(stage).__name__}")
    if not stage.name or stage.name == type(stage).__name__:
        stage.name = stage_id
    filter_pairs: List[Tuple[Any, str]] = []
    for node in spec.get("filters") or []:
        node = dict(node)
        # `on_filtered` is worker semantics, not a RecordFilter kwarg.
        mode = str(node.pop("on_filtered", "drop"))
        if mode not in ("drop", "pass"):
            raise ValueError(f"stage {stage_id!r}: on_filtered must be drop|pass, got {mode!r}")
        filter_pairs.append((hydra.utils.instantiate(node), mode))
    drop_fields = [str(name) for name in (spec.get("drop_fields") or [])]
    return stage, filter_pairs, drop_fields


# --------------------------------------------------------------------------- #
# Record selection and flagging (mirrors runner._select_input / _flag_record)
# --------------------------------------------------------------------------- #
def select_input(
    records: List[dict], filter_pairs: List[Tuple[Any, str]], drop_fields: List[str]
) -> List[Tuple[dict, bool]]:
    """Return ``(record, skip)`` pairs in input order.

    A record rejected by any ``drop`` filter vanishes (drop wins over pass, so
    the outcome is filter-order independent); one rejected only by ``pass``
    filters is kept with ``skip=True`` and rides through unprocessed.
    ``drop_fields`` are popped from every kept record, exactly as the
    standalone ``_select_input`` does.
    """
    selected: List[Tuple[dict, bool]] = []
    for record in records:
        rejecting = [mode for f, mode in filter_pairs if not f.keep(record)]
        if "drop" in rejecting:
            continue
        selected.append((record, bool(rejecting)))
    for record, _ in selected:
        for name in drop_fields:
            record.pop(name, None)
    return selected


def flag_record(record: dict, stage_id: str, exc: Exception) -> dict:
    """The standalone ``_flag_record`` field shape, plus a truncated traceback —
    in a fleet you debug from the record you hold, not across boxes' logs."""
    out = dict(record)
    out["error"] = True
    out["error_message"] = str(exc)
    out["error_type"] = f"{type(exc).__module__}.{type(exc).__name__}"
    out["error_stage"] = stage_id
    tb = "".join(traceback.format_exception(exc))
    out["error_traceback"] = tb[-wire.ERROR_TRACEBACK_LIMIT :]
    return out


# --------------------------------------------------------------------------- #
# Lease: heartbeat is liveness, and our sensor for having lost the message
# --------------------------------------------------------------------------- #
class Lease:
    """Tracks the last *successful* ``InProgress`` for one held message."""

    def __init__(self) -> None:
        self.last_ok = time.monotonic()

    @property
    def age(self) -> float:
        return time.monotonic() - self.last_ok

    @property
    def expired(self) -> bool:
        # Past AckWait the server has definitively redelivered elsewhere.
        return self.age > wire.ACK_WAIT_S

    @property
    def safe_to_upload(self) -> bool:
        # Pre-upload check: shrink the lineage-skew window to one PUT.
        return self.age <= wire.ACK_WAIT_S / 2


def nak_delay(attempt: int) -> float:
    return min(
        wire.NAK_BACKOFF_BASE_S * wire.NAK_BACKOFF_FACTOR ** max(attempt - 1, 0),
        wire.NAK_BACKOFF_CAP_S,
    )


# --------------------------------------------------------------------------- #
# The worker
# --------------------------------------------------------------------------- #
class Worker:
    def __init__(
        self,
        *,
        nats_url: str,
        worker_id: str,
        concurrency: int,
        bindings: Dict[str, Any],
        version: str,
        store: s3io.DistStore,
    ) -> None:
        self.nats_url = nats_url
        self.worker_id = worker_id
        self.concurrency = concurrency
        self.bindings = bindings
        self.version = version
        self.store = store
        self.capabilities = derive_capabilities(bindings, version)

        self._nc = None
        self._js = None
        self._subs: Dict[Tuple[str, str], Any] = {}  # (stream, consumer) -> pull sub
        self._stage_cache: Dict[str, Tuple[Stage, List[Tuple[Any, str]], List[str]]] = {}
        self._records_in_flight = 0
        self._setup_holds = 0
        self._chunk_tasks: set = set()
        self._draining = asyncio.Event()
        self._rr = 0

    # -- lifecycle ----------------------------------------------------------
    async def run(self) -> None:
        self._nc = await nats.connect(self.nats_url, **wire.nats_connect_options())
        self._js = self._nc.jetstream()
        logger.info(
            f"worker {self.worker_id!r} up: version={self.version} "
            f"concurrency={self.concurrency} capabilities={sorted(self.capabilities)}"
        )
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._start_drain, signal.Signals(sig).name)
        stdin_task = asyncio.create_task(self._watch_stdin())
        discovery = asyncio.create_task(self._discovery_loop())
        heartbeat = asyncio.create_task(self._fleet_heartbeat_loop())
        try:
            await self._pull_loop()
            if self._chunk_tasks:
                logger.info(f"draining: waiting for {len(self._chunk_tasks)} in-flight chunk(s)")
                await asyncio.gather(*self._chunk_tasks, return_exceptions=True)
        finally:
            discovery.cancel()
            stdin_task.cancel()
            heartbeat.cancel()
            await self._nc.close()
        logger.info("worker drained; exiting")

    def _start_drain(self, why: str) -> None:
        if not self._draining.is_set():
            logger.info(f"drain requested ({why}): no new pulls, finishing in-flight chunks")
            self._draining.set()

    async def _watch_stdin(self) -> None:
        """Drain on an ``s`` line — the no-signal path for an attached terminal.

        The reader is a daemon thread, not an executor task: a thread blocked
        on ``readline`` must never keep a SIGTERM-drained process from
        exiting.
        """
        if not sys.stdin.isatty():
            return
        loop = asyncio.get_running_loop()

        def read_lines() -> None:
            for line in sys.stdin:
                if line.strip().lower() == "s":
                    loop.call_soon_threadsafe(self._start_drain, "keypress")
                    return

        threading.Thread(target=read_lines, name="stdin-drain", daemon=True).start()

    async def _fleet_heartbeat_loop(self) -> None:
        """Ephemeral pub/sub for observability only — never load-bearing."""
        from .observability import fleet_heartbeat_payload

        while True:
            try:
                await self._nc.publish(
                    wire.FLEET_HEARTBEAT_SUBJECT,
                    await asyncio.to_thread(fleet_heartbeat_payload, self),
                )
            except Exception as exc:
                logger.debug(f"fleet heartbeat failed: {exc!r}")
            await asyncio.sleep(10)

    # -- discovery: bind consumers whose requires we satisfy -----------------
    async def _discovery_loop(self) -> None:
        while True:
            try:
                await self._discover_once()
            except Exception as exc:
                logger.warning(f"consumer discovery failed: {exc!r}")
            await asyncio.sleep(wire.REDISCOVER_INTERVAL_S)

    async def _discover_once(self) -> None:
        streams = await self._js.streams_info()
        for stream in streams:
            name = stream.config.name
            if not name.startswith("JOB_") or name.endswith("_EV") or name.startswith("KV_"):
                continue
            for consumer in await self._js.consumers_info(name):
                key = (name, consumer.name)
                if key in self._subs:
                    continue
                requires = set(json.loads((consumer.config.metadata or {}).get("requires", "[]")))
                if not requires <= self.capabilities:
                    continue
                self._subs[key] = await self._js.pull_subscribe_bind(
                    consumer=consumer.name, stream=name
                )
                logger.info(f"bound {name}/{consumer.name} (requires={sorted(requires)})")

    # -- demand-driven pull ---------------------------------------------------
    @property
    def _in_flight(self) -> int:
        return self._records_in_flight + self._setup_holds

    async def _pull_loop(self) -> None:
        while not self._draining.is_set():
            subs = list(self._subs.items())
            if not subs or self._in_flight >= self.concurrency:
                await asyncio.sleep(0.25)
                continue
            key, sub = subs[self._rr % len(subs)]
            self._rr += 1
            try:
                msgs = await sub.fetch(1, timeout=0.5)
            except nats.errors.TimeoutError:
                continue
            except Exception as exc:
                # A deleted consumer (job cancelled/edited) is routine churn.
                logger.info(f"unbinding {key[0]}/{key[1]}: {exc!r}")
                self._subs.pop(key, None)
                continue
            for msg in msgs:
                self._setup_holds += 1
                task = asyncio.create_task(self._process(msg))
                self._chunk_tasks.add(task)
                task.add_done_callback(self._chunk_tasks.discard)

    # -- one chunk -------------------------------------------------------------
    async def _process(self, msg) -> None:
        try:
            message = wire.WorkMessage.from_bytes(msg.data)
        except Exception:
            logger.exception("undecodable work message; acking it away")
            self._setup_holds -= 1
            await msg.ack()
            return

        attempt = msg.metadata.num_delivered
        with logger.contextualize(
            worker=self.worker_id,
            job=message.job_id,
            stage=message.stage_id,
            chunk=message.chunk_id,
            sig=message.sig,
        ):
            lease = Lease()
            work = asyncio.create_task(self._run_chunk(msg, message, attempt, lease))
            beat = asyncio.create_task(self._heartbeat(msg, lease, work))
            try:
                await work
            except LeaseLost:
                logger.warning(
                    f"lease lost after {lease.age:.0f}s — chunk aborted, it is redelivering elsewhere"
                )
            except ChunkOutage as exc:
                delay = nak_delay(attempt)
                logger.warning(f"outage breaker: {exc}; nak with {delay:.0f}s backoff")
                await self._try_nak(msg, delay)
            except Exception as exc:
                delay = nak_delay(attempt)
                logger.exception(f"chunk failed (attempt {attempt}): {exc!r}; nak with {delay:.0f}s backoff")
                await self._try_nak(msg, delay)
            finally:
                beat.cancel()

    async def _heartbeat(self, msg, lease: Lease, work: asyncio.Task) -> None:
        """Keep the lease alive; abort the chunk once it is definitively gone.

        A single failed ``InProgress`` (a blip) aborts nothing — only a gap
        longer than AckWait does, at which point the server has already
        redelivered and we are a zombie burning GPU.
        """
        while True:
            await asyncio.sleep(wire.INPROGRESS_INTERVAL_S)
            try:
                await msg.in_progress()
                lease.last_ok = time.monotonic()
            except Exception as exc:
                logger.warning(f"InProgress failed (lease age {lease.age:.0f}s): {exc!r}")
            if lease.expired:
                work.cancel()
                return

    async def _try_nak(self, msg, delay: float) -> None:
        try:
            await msg.nak(delay=delay)
        except Exception as exc:
            logger.warning(f"nak failed (lease will expire on its own): {exc!r}")

    def _stage_for(self, message: wire.WorkMessage):
        """Stages are cached per sig so model-level concurrency caps and HTTP
        clients are shared across chunks, as in the standalone runner."""
        cached = self._stage_cache.get(message.sig)
        if cached is None:
            cached = build_stage(message.stage_spec, message.stage_id, self.bindings)
            self._stage_cache[message.sig] = cached
        return cached

    async def _run_chunk(self, msg, message: wire.WorkMessage, attempt: int, lease: Lease) -> None:
        try:
            # Skip-if-done: a redelivered chunk whose output landed costs one
            # HEAD, not GPU time. The done event still publishes (hints are
            # cheap; the monitor dedups/verifies).
            if await asyncio.to_thread(self.store.exists, message.output_key):
                logger.bind(lifecycle=True).info("output already exists — skip-if-done")
                await self._publish_done(message, attempt)
                await msg.ack()
                return

            records: List[dict] = []
            for key in message.input_keys:
                records.extend(await asyncio.to_thread(s3io.read_chunk, self.store, key))

            stage, filter_pairs, drop_fields = self._stage_for(message)
            selected = select_input(records, filter_pairs, drop_fields)
            outputs, flagged, connection_flags, ran, flagged_types = await self._run_records(
                stage, message, attempt, selected
            )

            # Outage breaker: a chunk mostly lost to connection-class failures
            # is an outage reading as data — nak with backoff instead of
            # uploading it. Validator rejections never count: they are the
            # records' own verdicts, and a redelivery would only reproduce them.
            if ran and connection_flags / ran > wire.OUTAGE_FLAG_RATIO:
                raise ChunkOutage(f"{flagged}/{ran} records flagged ({connection_flags} connection-class)")

            if not lease.safe_to_upload:
                raise LeaseLost()
            await asyncio.to_thread(s3io.write_chunk, self.store, message.output_key, outputs)
            await self._publish_done(
                message, attempt, records=len(outputs), flagged=flagged, flagged_types=flagged_types
            )
            await msg.ack()
            logger.bind(lifecycle=True).info(
                f"chunk done: {len(outputs)} records ({flagged} flagged), attempt {attempt}"
            )
        finally:
            self._setup_holds -= 1

    async def _run_records(
        self,
        stage: Stage,
        message: wire.WorkMessage,
        attempt: int,
        selected: List[Tuple[dict, bool]],
    ) -> Tuple[List[dict], int, int, int, Dict[str, int]]:
        """The record loop — four lines of ceremony around one reused call."""
        model = (message.stage_spec.get("stage") or {}).get("model") or {}
        prov_entry = {
            "stage": message.stage_id,
            "sig": message.sig,
            "chunk": message.chunk_id,
            "worker": self.worker_id,
            "version": message.version,
            "attempt": attempt,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        if model.get("name"):
            prov_entry["model"] = model["name"]

        outputs: List[Optional[dict]] = [None] * len(selected)
        flagged = 0
        connection_flags = 0
        flagged_types: Dict[str, int] = {}

        def flag_out(record: dict, exc: Exception) -> dict:
            nonlocal flagged, connection_flags
            context = errored_context(exc)
            base = context.to_dict() if context is not None else record
            base.pop(wire.PROV_FIELD, None)
            out = flag_record(base, message.stage_id, exc)
            flagged += 1
            flagged_types[out["error_type"]] = flagged_types.get(out["error_type"], 0) + 1
            if isinstance(exc, CONNECTION_ERRORS):
                connection_flags += 1
            return out

        async def run_one(i: int, record: dict) -> None:
            prov = record.pop(wire.PROV_FIELD, [])
            try:
                out = (await stage.run(dict(record))).to_dict()
            except asyncio.CancelledError:
                raise  # the lease sensor's abort must propagate
            except Exception as exc:
                out = flag_out(record, exc)
            out[wire.PROV_FIELD] = prov + [dict(prov_entry)]
            outputs[i] = out

        async def run_batch_group(pairs: List[Tuple[int, dict]]) -> None:
            provs = [record.pop(wire.PROV_FIELD, []) for _, record in pairs]
            results = await stage.run_group([record for _, record in pairs])
            for (i, record), prov, (out, exc) in zip(pairs, provs, results):
                if exc is not None:
                    out = flag_out(record, exc)
                out[wire.PROV_FIELD] = prov + [dict(prov_entry)]
                outputs[i] = out

        runnable: List[Tuple[int, dict]] = []
        for i, (record, skip) in enumerate(selected):
            if skip or record.get("error"):
                outputs[i] = record  # rides through unchanged, no prov entry
            else:
                runnable.append((i, record))
        ran = len(runnable)

        tasks = []
        try:
            from .batch import BatchedLLMStage  # local: workers without batch jobs never import it

            if isinstance(stage, BatchedLLMStage):
                # A group holds `size` slots for the duration of its request,
                # so the demand-driven pull rule is unchanged.
                for start in range(0, len(runnable), stage.batch_size):
                    group = runnable[start : start + stage.batch_size]
                    self._records_in_flight += len(group)
                    task = asyncio.create_task(run_batch_group(group))
                    task.add_done_callback(lambda _, n=len(group): self._decrement_records(n))
                    tasks.append(task)
            else:
                for i, record in runnable:
                    self._records_in_flight += 1
                    task = asyncio.create_task(run_one(i, record))
                    task.add_done_callback(lambda _: self._decrement_records())
                    tasks.append(task)
            if tasks:
                await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            raise
        return [out for out in outputs if out is not None], flagged, connection_flags, ran, flagged_types

    def _decrement_records(self, n: int = 1) -> None:
        self._records_in_flight -= n

    async def _publish_done(
        self,
        message: wire.WorkMessage,
        attempt: int,
        records: int = -1,
        flagged: int = 0,
        flagged_types: Optional[Dict[str, int]] = None,
    ) -> None:
        event = wire.DoneEvent(
            job_id=message.job_id,
            stage_index=message.stage_index,
            stage_id=message.stage_id,
            chunk_id=message.chunk_id,
            sig=message.sig,
            output_key=message.output_key,
            worker=self.worker_id,
            attempt=attempt,
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            records=records,
            flagged=flagged,
            flagged_types=flagged_types or {},
        )
        # Publish-then-ack: die in between and the work message redelivers,
        # recomputes and republishes — never a written-but-unannounced chunk.
        await self._js.publish(
            wire.done_subject(message.job_id),
            event.to_bytes(),
            headers={"Nats-Msg-Id": event.msg_id},
        )


def worker_from_env() -> Worker:
    return Worker(
        nats_url=wire.nats_url(),
        worker_id=os.environ.get("DC_WORKER_ID") or socket.gethostname(),
        concurrency=int(os.environ.get("DC_WORKER_CONCURRENCY", "32")),
        bindings=load_bindings(),
        version=wire.resolve_version(),
        store=s3io.store_from_env(),
    )
