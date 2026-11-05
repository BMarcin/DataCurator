"""Controller: prepare and administer distributed jobs.

Short-lived admin actions (``seed``/``status``/``rerun``/``cancel``/``gc``),
driven by ``run_controller.py``. Seeding is idempotent and resumable: every
step converges on re-run (deterministic keys overwrite identically, stream and
consumer creation is create-if-absent, publishes dedup by ``Nats-Msg-Id``) and
the durable job record is written **last**, so a job exists only once fully
seeded. All rerun/resume logic lives here — the worker is unaware of it.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import nats
from hydra import compose, initialize_config_dir
from loguru import logger
from nats.js import api as js_api
from nats.js.errors import BadRequestError, KeyNotFoundError, NotFoundError
from omegaconf import DictConfig, OmegaConf

from DataCurator.pipeline.prompts import inline_prompt_files
# Reused, never modified: the standalone runner's dataset readers.
from DataCurator.pipeline.runner import _read_records, _resolve_inputs, order_records

from . import s3io, wire

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _seed_publish_concurrency() -> int:
    """Stage-0 publishes fanned out at seed time (env ``DC_SEED_PUBLISH_CONCURRENCY``, default 32).

    Each JetStream publish awaits its own PubAck round-trip, so a bounded gather
    overlaps them without flooding the server with unbounded in-flight acks.
    """
    return int(os.environ.get("DC_SEED_PUBLISH_CONCURRENCY", "32"))


def _progress_reporter(label: str, buckets: int = 10) -> Callable[[int, int], None]:
    """A throttled ``(done, total)`` callback: logs ``label`` ~``buckets`` times plus at 100%.

    Seeding is run by hand, so this gives the operator a live sense of how a
    long upload/publish is moving without one log line per chunk.
    """

    def report(done: int, total: int) -> None:
        step = max(1, total // buckets)
        if done == total or done % step == 0:
            logger.info(f"  {label} {done}/{total} ({done * 100 // total}%)")

    return report


class SeedError(RuntimeError):
    """A job cannot be seeded as specified; nothing was published."""


# --------------------------------------------------------------------------- #
# Config composition and plan rendering
# --------------------------------------------------------------------------- #
def compose_experiment(experiment: str, overrides: Optional[List[str]] = None) -> DictConfig:
    """Compose the experiment behaviour-only: bindings never reach a spec."""
    with initialize_config_dir(config_dir=str(_REPO_ROOT / "config"), version_base=None):
        return compose(
            config_name="config",
            overrides=[f"experiment={experiment}", "bindings=none", *(overrides or [])],
        )


@dataclass
class Plan:
    """A fully rendered job plan: what seed publishes and the record stores."""

    job_id: str
    version: str
    id_field: str
    stages: List[wire.StagePlan]

    def work_message(self, prefix: str, stage_index: int, chunk_id: str) -> wire.WorkMessage:
        return wire.build_work_message(
            self.job_id, self.version, self.stages, prefix, stage_index, chunk_id
        )


def render_plan(cfg: DictConfig, version: str) -> Plan:
    """Render per-stage behaviour specs, enforce the bindings split, chain sigs."""
    job_id = str(cfg.get("experiment_name", "pipeline"))
    if not wire.JOB_ID_RE.match(job_id):
        raise SeedError(
            f"experiment_name {job_id!r} is not a valid job id ({wire.JOB_ID_RE.pattern}); "
            "it is embedded in NATS subjects and stream names"
        )
    id_field = OmegaConf.select(cfg, "id_field") or "id"

    rendered: List[tuple[str, Dict[str, Any]]] = []
    for stage_id in cfg.order:
        node = cfg.pipeline[stage_id]
        if not node.get("enabled", True):
            continue
        # Prompt files are read here, so workers get the prompt text inside the
        # spec (and the chained sig covers it) instead of reading files.
        container = inline_prompt_files(OmegaConf.to_container(node, resolve=True))
        # Runner-only orchestration keys are not behaviour and never travel.
        container.pop("enabled", None)
        container.pop("pause_before", None)
        container = wire.strip_null_bindings(container)
        # Serving knobs are per worker (bindings), never behaviour.
        container = wire.strip_serving_keys(container)
        leaks = wire.find_binding_keys(container)
        if leaks:
            raise SeedError(
                f"stage {stage_id!r} carries binding keys in its behaviour spec: "
                f"{', '.join(leaks)} — endpoints and secrets belong in worker env "
                "(compose with bindings=none, see config/bindings/)"
            )
        rendered.append((stage_id, container))
    if not rendered:
        raise SeedError("no enabled stages in the experiment")

    sigs = wire.chain_sigs(version, [wire.canonical(spec) for _, spec in rendered])
    stages = [
        wire.StagePlan(
            index=i,
            stage_id=stage_id,
            sig=sig,
            requires=wire.derive_requires(spec, version),
            stage_spec=spec,
        )
        for i, ((stage_id, spec), sig) in enumerate(zip(rendered, sigs))
    ]
    return Plan(job_id=job_id, version=version, id_field=id_field, stages=stages)


# --------------------------------------------------------------------------- #
# S3 input seeding
# --------------------------------------------------------------------------- #
def seed_input(
    store: s3io.DistStore,
    prefix: str,
    plan: Plan,
    dataset_input: Any,
    chunk_size: int,
    priority: Any = None,
) -> int:
    """Freeze record ids and chunk boundaries into ``input/``; return chunk count.

    The prefix is immutable once seeded: a rerun reuses the existing chunks and
    never re-reads the local dataset. Changed source data = a new job.
    ``priority`` (``dataset.priority``) sorts the records before they are cut,
    so higher-priority rows land in lower chunk ids — the order stage-0 is
    published in and every later stage follows. Like the ids, it is frozen
    here: changing it takes effect only for a new job.
    """
    existing = [
        k for k in store.list_keys(wire.input_prefix(prefix, plan.job_id)) if k.endswith(".jsonl")
    ]
    if existing:
        logger.info(f"input/ already seeded ({len(existing)} chunks) — reusing as-is")
        return len(existing)

    records = order_records(_read_records(_resolve_inputs(dataset_input)), priority)
    if not records:
        raise SeedError(f"dataset input {dataset_input!r} yielded no records")
    for i, record in enumerate(records):
        record.setdefault(plan.id_field, f"{i:08d}")

    items = [
        (
            wire.input_key(prefix, plan.job_id, wire.chunk_id_str(i)),
            records[start : start + chunk_size],
        )
        for i, start in enumerate(range(0, len(records), chunk_size))
    ]
    logger.info(
        f"uploading {len(records)} records as {len(items)} input chunks "
        f"(concurrency={s3io.upload_concurrency()})"
    )
    s3io.write_chunks(store, items, on_progress=_progress_reporter("uploaded chunks"))
    return len(items)


# --------------------------------------------------------------------------- #
# NATS setup (create-if-absent throughout)
# --------------------------------------------------------------------------- #
async def _connect():
    return await nats.connect(wire.nats_url(), **wire.nats_connect_options())


async def _ensure_stream(js, config: js_api.StreamConfig) -> None:
    try:
        await js.add_stream(config)
    except BadRequestError:
        await js.stream_info(config.name)  # exists — converged; re-raise if truly broken


async def ensure_job_streams(js, plan: Plan) -> None:
    await _ensure_stream(
        js,
        js_api.StreamConfig(
            name=wire.work_stream(plan.job_id),
            subjects=[wire.work_subject_wildcard(plan.job_id)],
            retention=js_api.RetentionPolicy.WORK_QUEUE,
            duplicate_window=wire.DUPLICATE_WINDOW_S,
        ),
    )
    await _ensure_stream(
        js,
        js_api.StreamConfig(
            name=wire.done_stream(plan.job_id),
            subjects=[wire.done_subject(plan.job_id)],
            retention=js_api.RetentionPolicy.LIMITS,
            max_age=wire.DONE_STREAM_MAX_AGE_S,
            duplicate_window=wire.DUPLICATE_WINDOW_S,
        ),
    )
    for stage in plan.stages:
        config = js_api.ConsumerConfig(
            durable_name=wire.consumer_name(stage.index),
            filter_subject=wire.work_subject(plan.job_id, stage.index),
            ack_policy=js_api.AckPolicy.EXPLICIT,
            ack_wait=wire.ACK_WAIT_S,
            max_deliver=wire.MAX_DELIVER,
            # Stamped at creation, never changed: workers route on this.
            metadata={
                "requires": json.dumps(stage.requires),
                "stage_id": stage.stage_id,
                "stage_index": str(stage.index),
            },
        )
        try:
            await js.add_consumer(wire.work_stream(plan.job_id), config)
        except BadRequestError:
            await js.consumer_info(wire.work_stream(plan.job_id), config.durable_name)


async def jobs_kv(js):
    try:
        return await js.create_key_value(js_api.KeyValueConfig(bucket=wire.JOBS_KV_BUCKET))
    except BadRequestError:
        return await js.key_value(wire.JOBS_KV_BUCKET)


async def publish_work(js, message: wire.WorkMessage, msg_id: Optional[str] = None) -> None:
    await js.publish(
        wire.work_subject(message.job_id, message.stage_index),
        message.to_bytes(),
        headers={"Nats-Msg-Id": msg_id or message.msg_id},
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
async def seed(
    experiment: str,
    overrides: Optional[List[str]] = None,
    max_chunks: Optional[int] = None,
    chunk_size: int = wire.CHUNK_SIZE,
) -> None:
    """Prepare a job: input chunks, streams/consumers, stage-0 messages, record.

    ``max_chunks`` seeds a canary: the full input is uploaded (it is immutable),
    but only the first N chunks enter the flow and the job record counts only
    those — re-seed without the flag to widen to the full dataset.
    """
    version = wire.resolve_version()
    cfg = compose_experiment(experiment, overrides)
    plan = render_plan(cfg, version)
    store = s3io.store_from_env()
    prefix = s3io.s3_prefix()

    total_chunks = seed_input(
        store, prefix, plan, cfg.dataset.input, chunk_size,
        priority=OmegaConf.select(cfg, "dataset.priority"),
    )
    chunk_count = total_chunks if max_chunks is None else min(max_chunks, total_chunks)

    nc = await _connect()
    try:
        js = nc.jetstream()
        kv = await jobs_kv(js)

        # Editing a running job = purge and re-seed: old-plan messages are
        # dead the moment the plan changes (their sigs are dead prefixes), so
        # nothing waits for in-flight work. Consumers are recreated because
        # their `requires` metadata is stamped at creation, never changed.
        previous = None
        try:
            entry = await kv.get(plan.job_id)
            previous = wire.JobRecord.from_bytes(entry.value)
        except (NotFoundError, KeyNotFoundError):
            pass
        if previous is not None and [s.sig for s in previous.stages] != [s.sig for s in plan.stages]:
            logger.info(f"plan changed — purging {wire.work_stream(plan.job_id)} and recreating consumers")
            prev_sigs = {s.index: s.sig for s in previous.stages}
            invalidated = [s.stage_id for s in plan.stages if prev_sigs.get(s.index) != s.sig]
            logger.info(f"stages recomputing under new sigs: {', '.join(invalidated)}")
            try:
                await js.purge_stream(wire.work_stream(plan.job_id))
                for stage in previous.stages:
                    try:
                        await js.delete_consumer(
                            wire.work_stream(plan.job_id), wire.consumer_name(stage.index)
                        )
                    except NotFoundError:
                        pass
            except NotFoundError:
                pass

        await ensure_job_streams(js, plan)

        # Publish stage-0 only for chunks not already done under the current
        # sig (one LIST, not N HEADs) — the NATS-wipe recovery path re-seeds
        # only what S3 says is missing. Chunks already past stage 0 are the
        # monitor's reconcile job, not seed's.
        stage0 = plan.stages[0]
        done0 = set(
            store.list_keys(
                wire.stage_sig_prefix(prefix, plan.job_id, stage0.index, stage0.stage_id, stage0.sig)
            )
        )
        candidates = (plan.work_message(prefix, 0, wire.chunk_id_str(i)) for i in range(chunk_count))
        pending = [message for message in candidates if message.output_key not in done0]
        sem = asyncio.Semaphore(_seed_publish_concurrency())
        report = _progress_reporter("published stage-0")
        published = 0
        logger.info(
            f"publishing {len(pending)} stage-0 messages "
            f"(concurrency={_seed_publish_concurrency()}, {chunk_count - len(pending)} already done)"
        )

        async def _publish(message: wire.WorkMessage) -> None:
            nonlocal published
            async with sem:
                await publish_work(js, message)
            # Single-threaded event loop: the counter needs no lock.
            published += 1
            report(published, len(pending))

        await asyncio.gather(*(_publish(message) for message in pending))

        done_info = await js.stream_info(wire.done_stream(plan.job_id))
        record = wire.JobRecord(
            job_id=plan.job_id,
            version=version,
            state="active",
            chunk_count=chunk_count,
            input_prefix=wire.input_prefix(prefix, plan.job_id),
            id_field=plan.id_field,
            stages=plan.stages,
            done_floor=done_info.state.last_seq,
        )
        await kv.put(plan.job_id, record.to_bytes())
        logger.info(
            f"seeded job {plan.job_id!r} version={version}: {chunk_count} chunks "
            f"({published} stage-0 messages published), "
            f"stages: {', '.join(f'{s.index:02d}_{s.stage_id}[{s.sig}]' for s in plan.stages)}"
        )
    finally:
        await nc.close()


async def _load_record(js, job_id: str) -> wire.JobRecord:
    kv = await jobs_kv(js)
    try:
        entry = await kv.get(job_id)
    except (NotFoundError, KeyNotFoundError):
        raise SeedError(f"no job record for {job_id!r} — was it seeded?")
    return wire.JobRecord.from_bytes(entry.value)


def _done_chunks(store: s3io.DistStore, prefix: str, record: wire.JobRecord, stage: wire.StagePlan) -> set:
    keys = store.list_keys(
        wire.stage_sig_prefix(prefix, record.job_id, stage.index, stage.stage_id, stage.sig)
    )
    return {key.rsplit("chunk-", 1)[1].removesuffix(".jsonl") for key in keys if "chunk-" in key}


async def status(job_id: str) -> None:
    """Per-stage done/missing under the plan's *current* sigs."""
    nc = await _connect()
    try:
        js = nc.jetstream()
        record = await _load_record(js, job_id)
        store = s3io.store_from_env()
        prefix = s3io.s3_prefix()
        expected = {wire.chunk_id_str(i) for i in range(record.chunk_count)}
        print(f"job {record.job_id}  state={record.state}  version={record.version}  chunks={record.chunk_count}")
        for stage in record.stages:
            done = _done_chunks(store, prefix, record, stage) & expected
            missing = sorted(expected - done)
            try:
                info = await js.consumer_info(wire.work_stream(job_id), wire.consumer_name(stage.index))
                queue = f"queued={info.num_pending} unacked={info.num_ack_pending} redelivered={info.num_redelivered}"
            except NotFoundError:
                queue = "consumer missing"
            line = f"  {stage.index:02d}_{stage.stage_id} [{stage.sig}]  done {len(done)}/{record.chunk_count}  {queue}"
            if missing and len(missing) <= 8:
                line += f"  missing: {','.join(missing)}"
            print(line)
    finally:
        await nc.close()


async def rerun(job_id: str, chunks: Optional[List[str]] = None, force: bool = False) -> None:
    """Enqueue only what is missing under the plan's current sigs.

    ``--force`` deletes the named chunks' outputs at every stage first, so
    skip-if-done no longer short-circuits and exactly those recompute.
    Reads the *stored* plan — rerun never re-renders config.
    """
    if force and not chunks:
        raise SeedError("--force requires --chunks: it deletes outputs")
    nc = await _connect()
    try:
        js = nc.jetstream()
        record = await _load_record(js, job_id)
        if record.state == "cancelled":
            raise SeedError(f"job {job_id!r} is cancelled; seed it again to restart it")
        store = s3io.store_from_env()
        prefix = s3io.s3_prefix()
        expected = {wire.chunk_id_str(i) for i in range(record.chunk_count)}
        targets = expected if chunks is None else (set(chunks) & expected)

        if force:
            doomed = [
                wire.stage_key(prefix, job_id, stage.index, stage.stage_id, stage.sig, chunk_id)
                for stage in record.stages
                for chunk_id in sorted(targets)
            ]
            deleted = store.delete_keys(doomed)
            logger.info(f"force: deleted outputs of chunks {sorted(targets)} at every stage ({deleted} keys)")

        done_by_stage = {s.index: _done_chunks(store, prefix, record, s) for s in record.stages}
        published = 0
        # A forced recompute is a *new* instruction: uniquify its msg-ids so
        # the dedup window (which exists to absorb duplicate seeds/repairs)
        # cannot swallow it when the original ran within the last 2 minutes.
        force_suffix = f":f{int(time.time())}" if force else ""
        for stage in record.stages:
            if stage.index == 0:
                ready = targets
            else:
                ready = done_by_stage[stage.index - 1] & targets
            for chunk_id in sorted(ready - done_by_stage[stage.index]):
                message = wire.build_work_message(
                    job_id, record.version, record.stages, prefix, stage.index, chunk_id
                )
                await publish_work(js, message, msg_id=message.msg_id + force_suffix)
                published += 1
        if published:
            # Reactivate the record and advance done_floor: the monitor treats
            # a floor bump like a re-seed and rebuilds its progress marks from
            # S3 — without this, its stale done/enqueued memory (or a
            # `complete` state) would strand the recomputed chunks.
            done_info = await js.stream_info(wire.done_stream(job_id))
            record.state = "active"
            record.done_floor = done_info.state.last_seq
            kv = await jobs_kv(js)
            await kv.put(job_id, record.to_bytes())
        logger.info(f"rerun: published {published} messages for what is missing under current sigs")
    finally:
        await nc.close()


async def cancel(job_id: str) -> None:
    """Purge the work queue and mark a surviving record cancelled.

    Robust to a half-deleted job: the stream is purged by its deterministic
    name and its consumers are enumerated directly, so leftover messages clear
    even when the S3 data or the durable job record is already gone. A monitor
    restart consults the record, so a surviving record is marked cancelled and
    can never resurrect the job."""
    nc = await _connect()
    try:
        js = nc.jetstream()
        stream = wire.work_stream(job_id)
        try:
            # stream_info first: it yields the count of stuck messages (purge
            # itself only returns success) and raises NotFound when the stream
            # is already gone, feeding the branch below.
            messages = (await js.stream_info(stream)).state.messages
            await js.purge_stream(stream)
            logger.info(f"purged work stream {stream} ({messages} messages)")
            # Deleting the consumers fences the race where a monitor holding a
            # stale record hands off a straggler done event after the purge:
            # such a message lands in the stream but nothing can ever pull it.
            # Enumerated from the stream (not the record) so a wiped job still
            # cleans up: every consumer on JOB_<job_id> belongs to this job.
            for consumer in await js.consumers_info(stream):
                try:
                    await js.delete_consumer(stream, consumer.name)
                except NotFoundError:
                    pass
        except NotFoundError:
            logger.info(f"work stream {stream} does not exist — nothing to purge")

        # Best-effort: mark a surviving record cancelled. A missing record means
        # there is nothing for a monitor to resurrect, so the purge above is all
        # cancel can (and needs to) do.
        try:
            record = await _load_record(js, job_id)
        except SeedError:
            logger.warning(f"no job record for {job_id!r}; purged the work stream only")
            return
        record.state = "cancelled"
        kv = await jobs_kv(js)
        await kv.put(job_id, record.to_bytes())
        logger.info(f"job {job_id!r} cancelled; in-flight chunks will finish into their keys unconsumed")
    finally:
        await nc.close()


async def gc(job_id: str) -> None:
    """Delete dead-sig prefixes: everything under ``stage/`` that the current
    plan does not address. Input and live outputs are never touched."""
    nc = await _connect()
    try:
        js = nc.jetstream()
        record = await _load_record(js, job_id)
    finally:
        await nc.close()
    store = s3io.store_from_env()
    prefix = s3io.s3_prefix()
    live = tuple(
        wire.stage_sig_prefix(prefix, job_id, stage.index, stage.stage_id, stage.sig)
        for stage in record.stages
    )
    all_stage_keys = store.list_keys(f"{prefix}/{job_id}/stage/")
    dead = [key for key in all_stage_keys if not key.startswith(live)]
    deleted = store.delete_keys(dead)
    logger.info(f"gc: deleted {deleted} dead-sig objects ({len(all_stage_keys) - deleted} live kept)")
