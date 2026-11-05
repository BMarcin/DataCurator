"""The monitor: the long-lived brain (one instance).

Consumes chunk-done events and enqueues the next stage for that chunk — it
owns the pipeline topology, loaded from the durable job record. Aggregates
progress, declares jobs complete, and reconciles against S3 as a backstop:
events drive latency, S3 drives correctness.

Handoff is effectively-once without any extra store: the done stream's
durable consumer acks an event only *after* stage N+1 is published, so a
crash in between merely redelivers the event, and a duplicate publish is
absorbed by msg-id dedup or, past the window, by the worker's skip-if-done.
Death of the monitor stalls handoff; nothing corrupts — restart replays.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import nats
import nats.errors
from croniter import croniter
from loguru import logger
from nats.js import api as js_api
from nats.js.errors import BadRequestError, NoKeysError, NotFoundError
from prometheus_client import Counter, Gauge

# Reused, never modified: best-effort ntfy notifications.
from DataCurator.pipeline.notifications import notify

from . import s3io, wire

MONITOR_DURABLE = "mon"
WORKER_ALIVE_WINDOW_S = 30
ALERT_COOLDOWN_S = 3600
FLAGGED_SPIKE_WINDOW_S = 600
FLAGGED_SPIKE_THRESHOLD = 25
REDELIVERY_SPIKE_THRESHOLD = 10
# Reconcile passes an active, incomplete job must sit with every queue empty
# and nothing in flight before its missing chunks are declared lost. Long
# enough for an in-transit hand-off and reconcile's own repairs to show up.
DRAINED_TICKS = 3
# Chunk ids listed per stage in the drained alert before eliding the rest.
DRAINED_ALERT_MAX_IDS = 10

# ntfy priority per alert kind; each is overridable with
# DC_NTFY_PRIORITY_<KIND> (e.g. DC_NTFY_PRIORITY_DEAD_LETTER=default).
ALERT_PRIORITIES: Dict[str, str] = {
    "dead_letter": "high",
    "ghost": "urgent",
    "stall": "high",
    "flagged": "default",
    "redelivery": "default",
    "complete": "default",
}
NTFY_PRIORITIES = frozenset({"min", "low", "default", "high", "urgent", "1", "2", "3", "4", "5"})

# The monitor is the single Prometheus scrape target — it re-exports what
# already flows through it. No exporters or agents on workers.
M_CHUNKS_DONE = Gauge("dc_chunks_done", "Chunks done under current sigs", ["job", "stage"])
M_CHUNK_COUNT = Gauge("dc_chunk_count", "Chunks in the job", ["job"])
M_QUEUE_DEPTH = Gauge("dc_queue_depth", "Undelivered work messages", ["job", "stage"])
M_UNACKED = Gauge("dc_chunks_unacked", "Delivered, unacked work messages", ["job", "stage"])
M_REDELIVERED = Gauge("dc_redeliveries", "Cumulative redeliveries", ["job", "stage"])
M_DEAD_LETTERED = Gauge("dc_chunks_deadlettered", "Dead-lettered chunks", ["job", "stage"])
M_DONE_EVENTS = Counter("dc_done_events", "Done events handled (throughput)", ["job", "stage"])
M_RECORDS_DONE = Counter("dc_records_done", "Records in completed chunks", ["job", "stage"])
M_FLAGGED = Counter("dc_records_flagged", "Flagged records", ["job", "stage", "error_type"])
M_JOB_STATE = Gauge("dc_job_state", "1 for the job's current state", ["job", "state"])
M_GHOST_EVENTS = Counter("dc_done_without_object", "Done events with no S3 object", ["job"])
M_WORKERS_ALIVE = Gauge("dc_workers_alive", "Workers heartbeating within 30s")
M_WORKER_INFLIGHT = Gauge("dc_worker_in_flight", "Records in flight per worker", ["worker"])
M_WORKER_INFO = Gauge("dc_worker_info", "1, labelled with version", ["worker", "version"])
M_WORKER_CPU = Gauge("dc_worker_cpu_pct", "Worker CPU percent", ["worker"])
M_WORKER_RAM = Gauge("dc_worker_ram_pct", "Worker RAM percent", ["worker"])
M_WORKER_GPU = Gauge("dc_worker_gpu_util_pct", "Worker GPU utilisation", ["worker", "gpu"])


def notifications_from_env() -> dict:
    server = os.environ.get("NTFY_URL")
    topic = os.environ.get("DC_NTFY_TOPIC")
    if not server or not topic:
        return {"enabled": False}
    return {
        "enabled": True,
        "backend": "ntfy",
        "server": server,
        "topic": topic,
        "auth_token": os.environ.get("NTFY_TOKEN"),
    }


def alert_priorities_from_env() -> Dict[str, str]:
    """``ALERT_PRIORITIES`` with ``DC_NTFY_PRIORITY_<KIND>`` overrides applied.

    An unknown priority value is a typo, not a reason to take the monitor
    down — warn and keep the default for that kind.
    """
    priorities = dict(ALERT_PRIORITIES)
    for kind in priorities:
        var = f"DC_NTFY_PRIORITY_{kind.upper()}"
        value = (os.environ.get(var) or "").strip().lower()
        if not value:
            continue
        if value in NTFY_PRIORITIES:
            priorities[kind] = value
        else:
            logger.warning(f"{var}={value!r} is not an ntfy priority; keeping {priorities[kind]!r}")
    return priorities


@dataclass
class JobState:
    """In-memory view of one job — rebuildable from the stream + S3."""

    record: wire.JobRecord
    done: Dict[int, Set[str]] = field(default_factory=dict)      # stage -> verified chunks
    enqueued: Dict[int, Set[str]] = field(default_factory=dict)  # stage -> published chunks
    dead_lettered: Set[str] = field(default_factory=set)         # "NN:chunk"
    flagged: Dict[str, Dict[str, int]] = field(default_factory=dict)  # stage_id -> error_type -> n
    queues: Dict[str, Dict[str, int]] = field(default_factory=dict)   # stage_id -> pending/unacked/redelivered
    queues_idle: bool = False   # last _watch_queues: nothing queued or unacked anywhere
    drained_ticks: int = 0      # consecutive idle-but-incomplete reconcile passes
    consumer_task: Optional[asyncio.Task] = None

    def mark_done(self, stage_index: int, chunk_id: str) -> None:
        self.done.setdefault(stage_index, set()).add(chunk_id)

    def mark_enqueued(self, stage_index: int, chunk_id: str) -> None:
        self.enqueued.setdefault(stage_index, set()).add(chunk_id)

    @property
    def last_stage(self) -> int:
        return len(self.record.stages) - 1

    @property
    def is_complete(self) -> bool:
        return len(self.done.get(self.last_stage, ())) >= self.record.chunk_count


class Monitor:
    def __init__(
        self,
        *,
        nats_url: str,
        store: s3io.DistStore,
        prefix: str,
        notifications: dict,
        metrics_port: Optional[int] = None,
        stall_cron: Optional[str] = None,
        priorities: Optional[Dict[str, str]] = None,
    ) -> None:
        self.nats_url = nats_url
        self.store = store
        self.prefix = prefix
        self.notifications = notifications
        self.priorities = {**ALERT_PRIORITIES, **(priorities or {})}
        self.metrics_port = metrics_port
        self.jobs: Dict[str, JobState] = {}
        self.fleet: Dict[str, dict] = {}  # worker_id -> last heartbeat payload (+_seen)
        self.listeners: Set[asyncio.Queue] = set()  # SSE subscribers (panel_api)
        self._alert_last: Dict[str, float] = {}
        self._flag_window: deque = deque()  # (ts, job, stage, error_type, count)
        self._redeliveries: Dict[tuple, int] = {}  # (job, stage) -> last cumulative
        # DC_STALL_CRON: when set, stall alerts are batched into one digest fired
        # on this schedule instead of one per-(job,stage) ntfy every hour. A
        # typo'd expression must never take the monitor down — warn and fall back.
        self._stall_cron: Optional[str] = None
        self._stall_next_fire: Optional[datetime] = None
        if stall_cron:
            if croniter.is_valid(stall_cron):
                self._stall_cron = stall_cron
                self._stall_next_fire = croniter(stall_cron, datetime.now()).get_next(datetime)
                logger.info(f"stall alerts aggregated on cron {stall_cron!r}")
            else:
                logger.warning(
                    f"DC_STALL_CRON={stall_cron!r} is not a valid cron expression; "
                    "falling back to per-stage stall alerts"
                )
        self._nc = None
        self._js = None
        self._kv = None

    # -- lifecycle ------------------------------------------------------------
    async def run(self) -> None:
        self._nc = await nats.connect(self.nats_url, **wire.nats_connect_options())
        self._js = self._nc.jetstream()
        self._kv = await self._ensure_kv()
        await self._nc.subscribe(
            "$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.>", cb=self._on_max_deliveries
        )
        await self._nc.subscribe(wire.FLEET_HEARTBEAT_SUBJECT, cb=self._on_fleet_heartbeat)
        server_task = None
        if self.metrics_port:
            import uvicorn

            from .panel_api import build_app

            config = uvicorn.Config(
                build_app(self), host="0.0.0.0", port=self.metrics_port, log_level="warning"
            )
            server_task = asyncio.create_task(uvicorn.Server(config).serve())
            logger.info(f"serving /metrics and /api on :{self.metrics_port}")
        logger.info("monitor up")
        try:
            while True:
                try:
                    await self._load_jobs()
                    await self._reconcile_all()
                except Exception as exc:
                    logger.exception(f"reconcile pass failed: {exc!r}")
                await asyncio.sleep(wire.RECONCILE_INTERVAL_S)
        finally:
            for state in self.jobs.values():
                if state.consumer_task:
                    state.consumer_task.cancel()
            if server_task:
                server_task.cancel()
            await self._nc.close()

    async def _ensure_kv(self):
        try:
            return await self._js.create_key_value(
                js_api.KeyValueConfig(bucket=wire.JOBS_KV_BUCKET)
            )
        except BadRequestError:
            return await self._js.key_value(wire.JOBS_KV_BUCKET)

    # -- job records ------------------------------------------------------------
    async def _load_jobs(self) -> None:
        try:
            keys = await self._kv.keys()
        except (NoKeysError, NotFoundError):  # empty bucket
            return
        for job_id in keys:
            entry = await self._kv.get(job_id)
            record = wire.JobRecord.from_bytes(entry.value)
            state = self.jobs.get(job_id)
            if state is None:
                state = JobState(record=record)
                self.jobs[job_id] = state
                logger.info(
                    f"job {job_id!r} loaded: state={record.state} chunks={record.chunk_count} "
                    f"stages={[s.stage_id for s in record.stages]}"
                )
                await self._replay_done_events(state)
                state.consumer_task = asyncio.create_task(self._consume_done_events(state))
            else:
                old_sigs = [s.sig for s in state.record.stages]
                old_floor = state.record.done_floor
                state.record = record  # controller may have edited/cancelled/rerun it
                if [s.sig for s in record.stages] != old_sigs or record.done_floor != old_floor:
                    # Edited plan or a rerun/force: our marks describe a flow
                    # that no longer holds (dead sigs, or outputs the
                    # controller deleted). Reconcile rebuilds `done` from S3.
                    logger.info(f"job {job_id!r} re-seeded or rerun; resetting progress state")
                    state.done.clear()
                    state.enqueued.clear()
                    state.dead_lettered.clear()
                    state.drained_ticks = 0
                    self._refresh_dead_letter_gauge(state)

    # -- done events --------------------------------------------------------------
    async def _replay_done_events(self, state: JobState) -> None:
        """Rebuild progress from the durable done stream (monitor restart).

        Every replayed event was acked by the durable consumer, and acking
        happens only after stage N+1 was published — so marking it enqueued
        here can never strand a chunk.
        """
        job_id = state.record.job_id
        try:
            sub = await self._js.pull_subscribe(
                wire.done_subject(job_id),
                stream=wire.done_stream(job_id),
                config=js_api.ConsumerConfig(deliver_policy=js_api.DeliverPolicy.ALL),
            )
        except NotFoundError:
            return
        replayed = 0
        try:
            while True:
                try:
                    msgs = await sub.fetch(64, timeout=1)
                except nats.errors.TimeoutError:
                    break
                for msg in msgs:
                    event = wire.DoneEvent.from_bytes(msg.data)
                    stage = state.record.stage(event.stage_index)
                    if stage is not None and stage.sig == event.sig:  # old-plan events are dead data
                        state.mark_done(event.stage_index, event.chunk_id)
                        # Only events newer than the record's done_floor imply
                        # a *current-plan* enqueue of stage N+1: a sig-valid
                        # stage-N event from before a re-seed handed off to a
                        # stage N+1 that no longer exists.
                        if (
                            event.stage_index < state.last_stage
                            and msg.metadata.sequence.stream > state.record.done_floor
                        ):
                            state.mark_enqueued(event.stage_index + 1, event.chunk_id)
                    await msg.ack()
                    replayed += 1
        finally:
            await sub.unsubscribe()
        if replayed:
            logger.info(f"job {job_id!r}: replayed {replayed} done events")

    async def _consume_done_events(self, state: JobState) -> None:
        """The latency path: one durable consumer per job's done stream."""
        job_id = state.record.job_id
        sub = await self._js.pull_subscribe(
            wire.done_subject(job_id),
            durable=MONITOR_DURABLE,
            stream=wire.done_stream(job_id),
        )
        while True:
            try:
                msgs = await sub.fetch(16, timeout=5)
            except nats.errors.TimeoutError:
                continue
            except Exception as exc:
                logger.warning(f"job {job_id!r}: done-event fetch failed: {exc!r}")
                await asyncio.sleep(5)
                continue
            for msg in msgs:
                try:
                    event = wire.DoneEvent.from_bytes(msg.data)
                    await self._handle_done(state, event)
                    await msg.ack()
                except Exception as exc:
                    logger.exception(f"job {job_id!r}: done event failed: {exc!r}")
                    await msg.nak(delay=10)

    async def _handle_done(self, state: JobState, event: wire.DoneEvent) -> None:
        """HEAD-verify, enqueue stage N+1 for this chunk, detect completion."""
        record = state.record
        if record.state == "cancelled":
            return
        stage = record.stage(event.stage_index)
        if stage is None or stage.sig != event.sig:
            return  # a straggler from an edited plan — dead data, never a clobber
        # Events are hints, not truth: "finished" must mean *in S3*.
        if not await asyncio.to_thread(self.store.exists, event.output_key):
            logger.error(
                f"done event without an S3 object behind it: {event.output_key} "
                f"(job={event.job_id} stage={event.stage_id} chunk={event.chunk_id} "
                f"worker={event.worker}) — upload-before-publish contract broken"
            )
            M_GHOST_EVENTS.labels(job=event.job_id).inc()
            self._alert(
                f"ghost:{event.job_id}",
                f"done event without S3 object: {event.job_id}",
                f"stage {event.stage_id} chunk {event.chunk_id} worker {event.worker} "
                f"reported {event.output_key} but it does not exist",
                priority=self.priorities["ghost"],
            )
            return
        state.mark_done(event.stage_index, event.chunk_id)
        self._emit(
            {"type": "chunk_done", "job": event.job_id, "stage": event.stage_id,
             "chunk": event.chunk_id, "worker": event.worker, "attempt": event.attempt}
        )
        M_DONE_EVENTS.labels(job=event.job_id, stage=event.stage_id).inc()
        if event.records >= 0:
            M_RECORDS_DONE.labels(job=event.job_id, stage=event.stage_id).inc(event.records)
        if event.flagged:
            self._note_flagged(event)
        next_index = event.stage_index + 1
        # Consult own state before enqueueing — but only *done* gates the
        # publish, never `enqueued`: this event may be a redelivery of a
        # crash between publish and ack, where publishing again is exactly
        # the repair (dedup/skip-if-done absorb the benign duplicate).
        if next_index <= state.last_stage and event.chunk_id not in state.done.get(next_index, ()):
            await self._enqueue(state, next_index, event.chunk_id)
        await self._check_complete(state)

    async def _enqueue(self, state: JobState, stage_index: int, chunk_id: str) -> None:
        record = state.record
        message = wire.build_work_message(
            record.job_id, record.version, record.stages, self.prefix, stage_index, chunk_id
        )
        ack = await self._js.publish(
            wire.work_subject(record.job_id, stage_index),
            message.to_bytes(),
            headers={"Nats-Msg-Id": message.msg_id},
        )
        if getattr(ack, "duplicate", False):
            # Swallowed by the dedup window: an identical msg-id was published
            # recently and may already be consumed (e.g. a forced recompute
            # soon after the original run). Do NOT mark it enqueued — the next
            # reconcile pass retries until the window expires and it lands.
            logger.info(
                f"enqueue of job {record.job_id!r} stage {stage_index} chunk {chunk_id} "
                "was deduped; a later reconcile will retry"
            )
            return
        state.mark_enqueued(stage_index, chunk_id)

    async def _check_complete(self, state: JobState) -> None:
        if state.record.state != "active" or not state.is_complete:
            return
        state.record.state = "complete"
        await self._kv.put(state.record.job_id, state.record.to_bytes())
        self._emit({"type": "job_state", "job": state.record.job_id, "state": "complete"})
        logger.info(f"job {state.record.job_id!r} complete ({state.record.chunk_count} chunks)")
        notify(
            self.notifications,
            f"job {state.record.job_id} complete",
            f"{state.record.chunk_count} chunks through {len(state.record.stages)} stages",
            priority=self.priorities["complete"],
            tags=["white_check_mark"],
        )

    def _emit(self, event: dict) -> None:
        """Push an SSE delta to panel subscribers (drop on backpressure)."""
        for queue in list(self.listeners):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    # -- fleet + alert plumbing --------------------------------------------------------
    async def _on_fleet_heartbeat(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
            worker_id = payload["worker_id"]
        except Exception:
            return
        payload["_seen"] = time.monotonic()
        if worker_id not in self.fleet:
            self._emit({"type": "fleet", "worker": worker_id, "event": "joined"})
        self.fleet[worker_id] = payload
        M_WORKER_INFLIGHT.labels(worker=worker_id).set(payload.get("in_flight", 0))
        M_WORKER_INFO.labels(worker=worker_id, version=str(payload.get("version"))).set(1)
        metrics = payload.get("metrics") or {}
        if "cpu_pct" in metrics:
            M_WORKER_CPU.labels(worker=worker_id).set(metrics["cpu_pct"])
        if "ram_pct" in metrics:
            M_WORKER_RAM.labels(worker=worker_id).set(metrics["ram_pct"])
        for gpu in metrics.get("gpu") or []:
            if "util_pct" in gpu:
                M_WORKER_GPU.labels(worker=worker_id, gpu=str(gpu.get("index"))).set(gpu["util_pct"])

    def _alive_workers(self) -> Dict[str, dict]:
        cutoff = time.monotonic() - WORKER_ALIVE_WINDOW_S
        return {wid: hb for wid, hb in self.fleet.items() if hb["_seen"] >= cutoff}

    def _alert(self, key: str, title: str, message: str, priority: Optional[str] = None) -> None:
        """Rate-limited operator alert (one per key per cooldown)."""
        now = time.monotonic()
        if now - self._alert_last.get(key, -ALERT_COOLDOWN_S) < ALERT_COOLDOWN_S:
            return
        self._alert_last[key] = now
        logger.warning(f"ALERT {title}: {message}")
        notify(self.notifications, title, message, priority=priority, tags=["warning"])

    def _note_flagged(self, event: wire.DoneEvent) -> None:
        now = time.monotonic()
        state = self.jobs.get(event.job_id)
        for error_type, count in (event.flagged_types or {}).items():
            M_FLAGGED.labels(job=event.job_id, stage=event.stage_id, error_type=error_type).inc(count)
            if state is not None:
                per_stage = state.flagged.setdefault(event.stage_id, {})
                per_stage[error_type] = per_stage.get(error_type, 0) + count
            self._flag_window.append((now, event.job_id, event.stage_id, error_type, count))
        while self._flag_window and self._flag_window[0][0] < now - FLAGGED_SPIKE_WINDOW_S:
            self._flag_window.popleft()
        totals: Dict[tuple, int] = {}
        for _, job, stage, error_type, count in self._flag_window:
            totals[(job, stage, error_type)] = totals.get((job, stage, error_type), 0) + count
        for (job, stage, error_type), total in totals.items():
            if total >= FLAGGED_SPIKE_THRESHOLD:
                self._alert(
                    f"flagged:{job}:{stage}:{error_type}",
                    f"flagged-rate spike: {job}",
                    f"{total} records flagged as {error_type} at stage {stage} "
                    f"in the last {FLAGGED_SPIKE_WINDOW_S // 60} min",
                    priority=self.priorities["flagged"],
                )

    # -- dead letters ---------------------------------------------------------------
    async def _on_max_deliveries(self, msg) -> None:
        """A chunk exhausted MaxDeliver — reserved for real process death and
        repeated systemic naks. Name the full correlation key and alert."""
        try:
            advisory = json.loads(msg.data)
            stream = advisory["stream"]
            if not stream.startswith("JOB_") or stream.endswith("_EV"):
                return
            raw = await self._js.get_msg(stream, seq=advisory["stream_seq"])
            work = wire.WorkMessage.from_bytes(raw.data)
        except Exception as exc:
            logger.warning(f"undecodable MAX_DELIVERIES advisory: {exc!r}")
            return
        key = f"{work.stage_index:02d}:{work.chunk_id}"
        state = self.jobs.get(work.job_id)
        if state is not None:
            if key in state.dead_lettered:
                return
            state.dead_lettered.add(key)
            self._refresh_dead_letter_gauge(state)
        else:
            M_DEAD_LETTERED.labels(job=work.job_id, stage=work.stage_id).inc()
        logger.error(
            f"dead-letter: job={work.job_id} stage={work.stage_id} chunk={work.chunk_id} "
            f"sig={work.sig} after {advisory.get('deliveries')} deliveries"
        )
        notify(
            self.notifications,
            f"dead-letter: {work.job_id}",
            f"stage {work.stage_id} chunk {work.chunk_id} failed {advisory.get('deliveries')} "
            f"deliveries (sig {work.sig}); worker set is in the logs: "
            f'{{job="{work.job_id}",stage="{work.stage_id}"}} |= "{work.chunk_id}"',
            priority=self.priorities["dead_letter"],
            tags=["rotating_light"],
        )

    # -- reconcile: S3 drives correctness -------------------------------------------
    async def _reconcile_all(self) -> None:
        M_WORKERS_ALIVE.set(len(self._alive_workers()))
        stalled: List[dict] = []
        for state in self.jobs.values():
            for job_state in ("active", "cancelled", "complete"):
                M_JOB_STATE.labels(job=state.record.job_id, state=job_state).set(
                    1 if state.record.state == job_state else 0
                )
            if state.record.state == "active":
                await self._reconcile_job(state)
                await self._watch_queues(state, stalled)
                self._check_drained(state)
        self._maybe_send_stall_digest(stalled)

    def _refresh_dead_letter_gauge(self, state: JobState) -> None:
        """Set the dead-letter gauge per stage from the job's dead-letter set."""
        for stage in state.record.stages:
            prefix = f"{stage.index:02d}:"
            count = sum(1 for key in state.dead_lettered if key.startswith(prefix))
            M_DEAD_LETTERED.labels(job=state.record.job_id, stage=stage.stage_id).set(count)

    @staticmethod
    def _missing_chunks(state: JobState) -> Dict[int, List[str]]:
        """Per stage, the chunks that cleared the previous stage (every chunk,
        for stage 0) but have no verified output here."""
        missing: Dict[int, List[str]] = {}
        previous = {wire.chunk_id_str(i) for i in range(state.record.chunk_count)}
        for stage in state.record.stages:
            done = state.done.get(stage.index, set())
            lost = sorted(previous - done)
            if lost:
                missing[stage.index] = lost
            previous = done
        return missing

    def _check_drained(self, state: JobState) -> None:
        """Alert on a job that drained without completing.

        Nothing queued, nothing in flight, yet chunks missing: they exhausted
        MaxDeliver (or were otherwise dropped) and nothing will ever pick them
        up again. The MAX_DELIVERIES advisory covers this only while the
        monitor is up — it is not persisted — so the lost set is rebuilt from
        the S3 done sets instead, recorded as dead-letters, and the operator
        is told to rerun. The cooldown repeats the alert while the job stays
        stuck; a rerun bumps ``done_floor``, which resets this state.
        """
        record = state.record
        if record.state != "active" or state.is_complete or not state.queues_idle:
            state.drained_ticks = 0
            return
        state.drained_ticks += 1
        if state.drained_ticks < DRAINED_TICKS:
            return
        missing = self._missing_chunks(state)
        if not missing:
            return
        for stage_index, chunks in missing.items():
            state.dead_lettered.update(f"{stage_index:02d}:{chunk}" for chunk in chunks)
        self._refresh_dead_letter_gauge(state)
        total = sum(len(chunks) for chunks in missing.values())
        lines = []
        for stage_index, chunks in missing.items():
            shown = ", ".join(chunks[:DRAINED_ALERT_MAX_IDS])
            more = f", … (+{len(chunks) - DRAINED_ALERT_MAX_IDS})" if len(chunks) > DRAINED_ALERT_MAX_IDS else ""
            lines.append(f"{record.stages[stage_index].stage_id}: {len(chunks)} ({shown}{more})")
        self._alert(
            f"drained:{record.job_id}:{record.done_floor}",
            f"drained: {record.job_id}",
            f"queues are empty and nothing is in flight, but {total} chunk(s) never "
            f"finished (dead-lettered or dropped):\n" + "\n".join(lines)
            + f"\nrecover: uv run run_controller.py rerun {record.job_id}",
            priority=self.priorities["dead_letter"],
        )

    async def _watch_queues(self, state: JobState, stalled: List[dict]) -> None:
        """Queue gauges + the stall and redelivery-spike alerts.

        Stalled stages are appended to ``stalled`` for the cron-scheduled digest
        when ``DC_STALL_CRON`` is set; otherwise each fires its own hourly alert.
        """
        record = state.record
        alive_caps = [set(hb.get("capabilities", [])) for hb in self._alive_workers().values()]
        idle = True
        for stage in record.stages:
            try:
                info = await self._js.consumer_info(
                    wire.work_stream(record.job_id), wire.consumer_name(stage.index)
                )
            except NotFoundError:
                continue  # no consumer, nothing queued: counts as idle
            if info.num_pending or info.num_ack_pending:
                idle = False
            labels = {"job": record.job_id, "stage": stage.stage_id}
            M_QUEUE_DEPTH.labels(**labels).set(info.num_pending)
            M_UNACKED.labels(**labels).set(info.num_ack_pending)
            M_REDELIVERED.labels(**labels).set(info.num_redelivered)
            state.queues[stage.stage_id] = {
                "queued": info.num_pending,
                "unacked": info.num_ack_pending,
                "redelivered": info.num_redelivered,
            }

            requires = set(stage.requires)
            if info.num_pending > 0 and not any(requires <= caps for caps in alive_caps):
                if self._stall_cron:
                    stalled.append({
                        "job": record.job_id,
                        "stage": stage.stage_id,
                        "queued": info.num_pending,
                        "requires": sorted(requires),
                    })
                else:
                    self._alert(
                        f"stall:{record.job_id}:{stage.stage_id}",
                        f"stall: {record.job_id}",
                        f"stage {stage.stage_id} has {info.num_pending} queued chunks and zero "
                        f"capable pullers (requires {sorted(requires)})",
                        priority=self.priorities["stall"],
                    )
            key = (record.job_id, stage.stage_id)
            delta = info.num_redelivered - self._redeliveries.get(key, 0)
            self._redeliveries[key] = info.num_redelivered
            if delta >= REDELIVERY_SPIKE_THRESHOLD:
                self._alert(
                    f"redeliver:{record.job_id}:{stage.stage_id}",
                    f"redelivery spike: {record.job_id}",
                    f"stage {stage.stage_id}: {delta} redeliveries in the last "
                    f"{wire.RECONCILE_INTERVAL_S}s — zombie storm? AckWait too tight, or "
                    "blocking CPU work stalling heartbeats",
                    priority=self.priorities["redelivery"],
                )
        state.queues_idle = idle

    def _maybe_send_stall_digest(self, stalled: List[dict]) -> None:
        """One batched stall notification per DC_STALL_CRON tick.

        Replaces the per-(job,stage) hourly spray: on each cron boundary, send a
        single digest listing every currently-stalled stage. Nothing stalled at
        fire time means nothing sent — the clock still advances, so we never
        double-fire, and quiet periods stay quiet.
        """
        if not self._stall_cron:
            return
        now = datetime.now()
        if now < self._stall_next_fire:
            return
        self._stall_next_fire = croniter(self._stall_cron, now).get_next(datetime)
        if not stalled:
            return
        alive = len(self._alive_workers())
        lines = [
            f"{s['job']} / {s['stage']}: {s['queued']} queued, requires {s['requires']}"
            for s in stalled
        ]
        notify(
            self.notifications,
            f"stalled: {len(stalled)} stage(s), {alive} workers alive",
            "\n".join(lines),
            priority=self.priorities["stall"],
            tags=["warning"],
        )

    async def _reconcile_job(self, state: JobState) -> None:
        """Repair from S3: verified done sets, missed handoffs, completion.

        Never touches stage 0 (the controller seeds it) and never re-enqueues
        a chunk this incarnation already published — an in-flight recompute is
        never triggered by the monitor merely re-scanning.
        """
        record = state.record
        for stage in record.stages:
            keys = await asyncio.to_thread(
                self.store.list_keys,
                wire.stage_sig_prefix(self.prefix, record.job_id, stage.index, stage.stage_id, stage.sig),
            )
            done = {key.rsplit("chunk-", 1)[1].removesuffix(".jsonl") for key in keys if "chunk-" in key}
            state.done[stage.index] = done
            M_CHUNKS_DONE.labels(job=record.job_id, stage=stage.stage_id).set(len(done))
        M_CHUNK_COUNT.labels(job=record.job_id).set(record.chunk_count)
        for stage_index in range(1, len(record.stages)):
            ready = state.done.get(stage_index - 1, set())
            have = state.done.get(stage_index, set())
            queued = state.enqueued.setdefault(stage_index, set())
            for chunk_id in sorted(ready - have - queued):
                logger.info(
                    f"reconcile: job {record.job_id!r} chunk {chunk_id} cleared stage "
                    f"{stage_index - 1} but stage {stage_index} was never enqueued — repairing"
                )
                await self._enqueue(state, stage_index, chunk_id)
        await self._check_complete(state)


def monitor_from_env() -> Monitor:
    return Monitor(
        nats_url=wire.nats_url(),
        store=s3io.store_from_env(),
        prefix=s3io.s3_prefix(),
        notifications=notifications_from_env(),
        metrics_port=int(os.environ.get("DC_MONITOR_PORT", "8080")),
        stall_cron=os.environ.get("DC_STALL_CRON"),
        priorities=alert_priorities_from_env(),
    )
