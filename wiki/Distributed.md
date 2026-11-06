# Distributed pipeline

> **Audience:** the coding agent implementing this system. This document is
> the complete specification. The prose sections define the contracts and the
> *why* behind them — read the rationale; it is what stops a plausible
> "simplification" from breaking an invariant. The **Implementation plan**
> near the end pins every remaining choice, the build order and the
> acceptance bar. Decisions in this document are final: implement, don't
> redesign. If something is genuinely ambiguous, take the simplest reading
> that preserves the bolded invariants and flag it in your summary; if a
> contract seems impossible, stop and report rather than improvising around
> it.

## What this is

The pipeline runs today as a single process on one box: `run_pipeline.py` →
`build_pipeline(cfg)` → `PipelineRunner.run()` (`DataCurator/pipeline/runner.py`),
feeding a dataset through an ordered list of stages, one stage at a time, with a
local asyncio semaphore and sharded JSONL on local disk. The distributed system
spreads that same work across an elastic fleet of VMs: prepare a job, spin up
worker nodes, add or remove nodes mid-run, and lose nodes without interrupting
the job.

This is **purely additive**. `run_pipeline.py` and `PipelineRunner` are
unchanged — standalone stays the default. Distribution is a separate, opt-in
code path via new entry scripts. What it *reuses* is the per-record execution
layer — `Stage.run()` and the modifiers — **not** the runner's orchestration
(sharding, resume, `_read_records`). Those concerns move to the controller and
the shared store.

## The work model

A single unit of work is a **`(stage, chunk)` pair**. The input dataset is split
into fixed-size **chunks**; each chunk flows through the stages independently, so
there is no global stage barrier — chunk 42 can be on stage 3 while chunk 43 is
still on stage 1. Every stage transition of every chunk is one queue message.

```
chunk 41: ████████ done
chunk 42: ██████░░ stage 3
chunk 43: ██░░░░░░ stage 1
chunk 44: ░░░░░░░░ queued
```

A snapshot mid-run is deliberately ragged: as soon as the first chunks clear a
stage, the next stage's (different) model is already busy — there is no phase
where one model's capacity idles — and a slow chunk delays only itself. Valid
because stages are per-record maps (see non-goals).

Data never travels through the queue. Intermediate outputs live in **S3**
(Garage), because the worker that runs `(stage N+1, chunk C)` may be a different
VM than the one that ran `(stage N, chunk C)`. A message is a small *pointer*:

```jsonc
// NATS subject: job.<id>.stage.<NN>    Nats-Msg-Id: <job>:<sig>:<stage>:<chunk>
{
  "schema_version": 1,
  "job_id":      "fix_documents_trivia",
  "stage_index": 2,
  "stage_id":    "pick_the_best_translation",
  "chunk_id":    "000042",
  "sig":         "9f3ab2c1",   // this stage's chained signature — see S3 layout
  "input_keys":  ["stage/01_fix_document/c0ffee12/chunk-000042.jsonl"],
  "output_key":  "stage/02_pick_the_best_translation/9f3ab2c1/chunk-000042.jsonl",
  "requires":    ["model:Qwen/Qwen3.6-35B-A3B", "ver:abc1234"],
  "version":     "git:abc1234",
  "stage_spec":  { /* rendered stage YAML — behaviour only, see below */ }
}
```

The worker downloads `input_keys`, runs the stage over those records, uploads
`output_key`, and reports the chunk done. Nothing else. It always uploads
**exactly one object** — when filters leave zero records the file is empty,
but it exists, so "the output key exists" stays the single completion signal
everywhere (skip-if-done, the monitor's checks, stage N+1's download).
Messages and the durable job record carry a `schema_version` so the wire
format can evolve without stranding an in-flight job mid-upgrade.

### S3 layout (data only)

```
s3://<bucket>/<prefix>/<job_id>/
  input/chunk-<CID>.jsonl                    # seeded by the controller (stable ids assigned)
  stage/<NN>_<stage>/<sig>/chunk-<CID>.jsonl # each stage's per-chunk output
```

`<sig>` is the stage's **chained signature** — `hash(version, sig of stage N-1,
rendered stage_spec)` — the distributed successor of `_signature`
(`runner.py:93`). Chaining means editing any stage flips the sig of that stage
*and everything downstream of it*, so outputs are spec-addressed: "missing"
always means *missing for this exact plan*, stale outputs are structurally
invisible (they sit under dead prefixes the controller GCs), and an old-plan
straggler can never clobber a new plan's output.

The `input/` prefix is **immutable once seeded**: record ids and chunk
boundaries are frozen in S3 and every sig chains off them, so a rerun reuses
the existing input chunks and never re-reads the local dataset. Changed source
data = a new job.

The input order is the processing order. Stage-0 messages are published in
chunk order, workers pull each stage's queue oldest-first, and the monitor
enqueues chunk K for the next stage once chunk K finishes the current stage,
so low chunk ids flow through the whole pipeline first. An experiment's
`dataset.priority` (`{field, order}`) stable-sorts the records before
`seed_input` cuts them, e.g. every `kind: query` row before any document. This
is a soft order, since parallel workers and redeliveries reorder a little. It
is frozen with the chunks, so a changed priority only applies to a new job.

The last stage's chunk files are the job's results. **No job metadata lives in
S3** — the plan, config and version travel in the messages and the durable job
record (below); S3 holds input, intermediate and result data only.

`helpers/pull_failed_s3.py` turns a finished job's failures into the input of
a rerun experiment. A record failed if the last stage holds it with
`error: true`, or if it never reached that stage. The script writes those
records' seeded `input/` rows unchanged, so they don't carry the `error` flag
(a flagged record skips every stage) or stale stage answers. It prints a count
per `<error_stage>:<error_type>`. `--full` writes the flagged stage records
instead, for inspection only. `--stage` and `--sig` pick another directory.
Credentials come from `.env.browse`.

```
uv run python helpers/pull_failed_s3.py gate_squadv2_qwen38_batching --out data/squadv2_segments_flagged.jsonl
```

A rerun's own failures can seed the next one the same way.
`gate_squadv2_qwen38_batching_rerun2` reads the failures of
`gate_squadv2_qwen38_batching_rerun` and differs from it only in the
`fix_document` `MetricXValidator`, whose `max_score` is 9 instead of 7.

```
uv run python helpers/pull_failed_s3.py gate_squadv2_qwen38_batching_rerun --out data/squadv2_segments_flagged2.jsonl
```

### Per-record provenance

Every record carries its own history in a reserved **`_prov` list** — one entry
appended by the worker per stage. It lives *inside* the record, so it is atomic
with the bytes by construction and travels through every copy, download and
export (S3 user-metadata was rejected: it stays behind in the bucket).

```jsonc
"_prov": [
  { "stage": "fix_document", "sig": "c0ffee12", "chunk": "000042",
    "worker": "gpu-01", "version": "git:abc1234",
    "model": "Qwen/Qwen3.6-35B-A3B-FP8", "attempt": 1,
    "ts": "2026-07-02T14:31:07Z" },
  { "stage": "pick_the_best_translation", "sig": "9f3ab2c1", "chunk": "000042",
    "worker": "gpu-02", "version": "git:abc1234",
    "model": "Qwen/Qwen3.6-35B-A3B", "attempt": 1,
    "ts": "2026-07-02T16:02:44Z" }
]
```

- **Append-only and rerun-safe.** A stage rebuilds its output from stage N-1's
  records, so a recompute re-appends exactly one fresh entry — entries never
  duplicate, and a rerun's entry *replaces* the old one along with the bytes.
- **Owned by the worker, invisible to stages.** The worker pops `_prov` before
  `Stage.run()` and re-attaches it (plus its own entry) on the way out —
  prompts, modifiers, validators and filters never see it.
- `attempt` is the message's JetStream delivery count: `attempt > 1` is the
  visible fingerprint of a redelivered/recomputed chunk (see lineage skew).
- `worker` + `ts` answer "which box, when"; a worker id maps to its full
  capability set via the fleet heartbeat log — only the stage-relevant
  `model` is denormalised into the entry.

### A stage is a rendered YAML — behaviour, not bindings

`stage_spec` is the fully-resolved OmegaConf for one `pipeline.<id>` node
(interpolations like `${_models.qwen35_stage1}` already expanded), which the
worker feeds straight to `hydra.utils.instantiate` — exactly what
`build_pipeline` does today (`runner.py:114`), just per-stage. The message is
therefore the single source of truth for *what to do*, so a worker never reads
local config and config drift across nodes is impossible.

It carries **behaviour only**, split from environment bindings:

| In the message (behaviour) | From the worker's env (bindings) |
|---|---|
| modifiers, prompt text, response schema, validators | `api_base`, `api_key` |
| sampling params, `model.name`, `batch` | S3 credentials, per-model `concurrency` / `timeout`, Google Translate `concurrency` / `timeout` / `retries` |
| filters, drop-fields | `languagetool.remote_server`, `google_translate.remote_server` / `client_name` |

This split is a **config-structure change** the distributed path requires, and
it fixes a standalone wart too: `_signature` (`runner.py:93`) hashes the whole
stage node, so today changing the LanguageTool URL flips the resume signature
and forces a full recompute of a stage whose behaviour never changed. Endpoints
and secrets belong outside the hashed behaviour config.

Prompt files are read on the controller: `render_plan` passes each stage
through `inline_prompt_files`, so the spec carries every message's `text`
instead of a `template` path. A worker never needs the prompt assets, and the
chained signature changes when the prompt text does.

The model's serving knobs follow the same rule. `concurrency` (requests one
worker keeps in flight per stage) and `timeout` describe the box's vLLM server,
not the output, so the controller strips them from every rendered spec and a
worker sets them per model in its bindings file next to `api_base`. Two
workers with different GPUs then share one job and its sigs, each sized to its
own `--max-num-seqs`. A spec rendered before the strip still carries the
preset values; the worker's binding overrides them, and an unbound knob falls
back to the spec or the stage default (8 in flight, OpenAI's 600 s timeout).
Google Translate gets the same treatment: the `concurrency`, `timeout` and
`retries` of the translate stage and of `GoogleTranslateModifier` depend on the
box's network, so they are stripped too and come from the bindings file's
`google_translate:` section (unset keys use the stage defaults 8, 30 s, 5).
That section also holds `remote_server`, the endpoint URL, and `client_name`, its
`client` query value, which a worker running a Google Translate stage must set.

The same rule flushes out the `runner:`-level knobs that change output bytes:
`id_field` moves out of `runner:` into the experiment, so it renders into
every message's behaviour spec — a distributed worker reads **nothing
output-affecting from local config**. `flag_on_errors` disappears entirely in
the distributed path, replaced by catch-all flagging (see the worker's record
loop).

## Correctness: effectively-once

The queue guarantees only **at-least-once** delivery — a task can run more than
once (a worker dies mid-chunk, an ack arrives late, a duplicate is published).
That is safe because re-running a task lands in the same place as running it
once, guaranteed by two properties:

1. **Equivalent recompute** — LLM stages *sample* (temperature > 0; vLLM output
   shifts with batching even at 0), so a rerun is **not byte-identical** — it is
   *equally valid*: same input chunk + same `stage_spec` + same version → an
   interchangeable output. (This is *why* behaviour travels in the message and
   the version is pinned.)
2. **Single-object, fixed-key writes** — chunk C's stage-N output always writes
   the same S3 key (fixed per `(stage, sig, chunk)`), as one object; no append,
   no merge, no duplicate, no half-written object. A rerun replaces it with an
   equally valid one.

At-least-once delivery + equivalent recompute + single-object writes ⇒ the
*effects* are **effectively-once**: any number of deliveries leaves exactly one
well-formed output per chunk. This single property is what lets the fleet lose
nodes freely with no special recovery code: a lost chunk simply redelivers and
recomputes. Every failure story below reduces to it.

Two consequences of "equivalent, not identical":

- **Skip-if-done.** Recompute is expensive, so the worker HEADs `output_key`
  *before* computing; if the object exists it publishes the done event and acks
  without running the stage. A redelivered chunk whose output already landed
  costs one HEAD, not GPU time.
- **Lineage skew (accepted, narrowed).** A worker that loses its lease but not
  its life (a network blip longer than AckWait) can upload *after* its
  replacement did, swapping the stored bytes for an equivalent rerun of them —
  the intermediate file then differs from what stage N+1 actually read.
  Records stay valid; only byte-level lineage suffers. A conditional PUT
  (`If-None-Match: *`) would fence this, but Garage documents concurrent-writer
  mutual exclusion as structurally unsupported (no consensus layer), so the
  skew is accepted and squeezed from both ends: the pre-compute HEAD makes it
  rare, and a **pre-upload lease check** — the worker verifies its last
  successful `InProgress` is fresh immediately before starting the PUT and
  aborts instead of uploading otherwise — shrinks the window from the chunk's
  whole tail to the duration of one PUT (seconds for a tens-of-records chunk;
  check-then-put stays non-atomic, so this narrows, never fences). The
  per-record `_prov` trail (above) makes what remains visible —
  `worker`/`attempt` in the stored chunk vs the done event betray a late
  overwrite. On a backend with conditional writes (AWS S3, MinIO)
  first-writer-wins is a one-line upgrade.

## Roles

Three roles; two entry scripts. The controller and monitor are distinct
lifecycles (short-lived admin actions vs a long-lived loop) and may share a
module.

- **Controller** — `run_controller.py` (admin CLI, short-lived). Prepares the
  job: assigns **stable record ids** and deterministic chunk boundaries over the
  sorted input, uploads the input chunks to S3, renders per-stage YAML, and
  seeds the stage-0 messages. Also drives reruns (enqueue only what is missing
  *under the plan's current sigs*), forced reruns of named chunks (delete their
  output keys and re-enqueue — skip-if-done no longer short-circuits, so
  exactly those recompute), and cancel (purge + mark the job record).
  Seeding is **idempotent and resumable**: every step converges on re-run
  (deterministic keys overwrite identically, consumers are create-if-absent,
  publishes dedup by `Nats-Msg-Id`), and the job record is written **last** —
  a job exists only once fully seeded, so a controller crash mid-seed is
  cured by running it again. All rerun/resume logic lives here — the worker
  is unaware of it.
- **Monitor** — the long-lived brain (one instance). Consumes chunk-done events
  and **enqueues the next stage** for that chunk (it owns the pipeline
  topology, loaded from the durable job record). Aggregates progress, sends **job-level** notifications, detects
  stalls, and declares the job complete when every chunk has cleared the last
  stage.
- **Worker** — `run_distributed.py` (long-lived, one per VM). A pure executor:
  pull → download → run stage → upload → report done. Advertises its
  capabilities and sends keep-alives. Topology-blind — it reports "finished
  stage N, chunk C" and the monitor decides what stage N+1 is.

### The worker's record loop

`Stage.run()` is reused as-is, but the behaviours of the standalone runner's
record loop live *around* the stage, and the worker must carry them —
forgetting any of these turns single records into dead-lettered chunks. The
whole contract is four lines of ceremony around one reused call:

```python
for record in chunk_records:              # after filters/drop_fields from the spec
    if record.get("error"):               # 1. flagged upstream → pass through
        out = record
    else:
        prov = record.pop("_prov", [])    # 2. the stage never sees _prov
        try:
            out = (await stage.run(record)).to_dict()   # ← the reused layer
        except Exception as exc:          # 3. catch-all → flag, don't kill the chunk
            out = flag(record, exc)       #    (+ outage breaker over the whole chunk)
        out["_prov"] = prov + [my_entry]  # 4. append provenance
    write(out)
```

- **Flagged records pass through.** A record carrying `error: true` bypasses
  `Stage.run()` at every later stage and rides through unchanged
  (`runner.py:573`) — otherwise a record flagged at stage N crashes stage
  N+1's modifiers and poisons its whole chunk.
- **Catch-all per-record flagging.** Any `Exception` a record raises flags it
  (`error`, `error_message`, `error_type`, `error_stage`, plus a truncated
  `error_traceback` — in a fleet you debug from the record you hold, not by
  hunting across boxes' log files) instead of failing the chunk — there is no `flag_on_errors` list to maintain, so no behaviour
  edit (and no sig flip) is ever needed to rescue a poison chunk. A failed
  record is never emptied: it keeps every field it carried (including
  `before`-phase enrichments, as `errored_context` preserves today) and merely
  lacks the stage's output field. Two guards:
  `except Exception`, never bare `except` — task cancellation (the lease
  sensor's abort) must propagate; and an **outage breaker** — when more than
  50 % of a chunk's runnable records flag with a connection-class error
  (transport errors, timeouts, and `ServiceUnavailableError` from an
  unreachable MetricX, LanguageTool or Google Translate), the worker naks with
  backoff instead of uploading, so an endpoint outage reads as an outage, not
  as data. Validator rejections and other content failures never count: they
  are the records' own verdicts, a redelivery would only reproduce them, so
  even a mostly-flagged chunk uploads. Dead-letters are thereby reserved for
  real process death and repeated systemic naks.
- **Filters and `drop_fields`** from the message spec are applied to the
  chunk's input ahead of the loop, exactly as `_select_input` does
  (`runner.py:667`). Each filter carries its own **`on_filtered: drop |
  pass`** flag (experiment-owned, inside the filter's behaviour spec): a
  record rejected by a `drop` filter vanishes from the flow (today's
  semantics, the default); a record rejected only by `pass` filters rides
  through unchanged like flagged records — present in every later output,
  never processed, never grouped into a batch. When rejecting filters
  disagree, **drop wins**, so the outcome is independent of filter order.
  One stage can then carry both kinds: "not a positive" discards the record
  for good, while "all three candidates identical" skips the stage but
  keeps the record in the final file.
- **`_prov` handling.** Pop before `Stage.run()`, append this stage's entry on
  the way out (see per-record provenance).

## Coordination

Everything runs on the **single NATS binary** (JetStream) plus S3. No separate
coordination store (KV, etcd) is needed — the candidates for shared state each
already have an owner:

| State | Lives in | Why not a KV store |
|---|---|---|
| Work items | JetStream stream (WorkQueue retention) | that *is* the stream |
| Routing | worker pull subscriptions | no central lookup happens (below) |
| Job progress | monitor memory | rebuildable from the stream + S3 |
| Fleet registry | ephemeral heartbeats | self-healing, transient |
| Job plan + cancel flag | **durable job record** — JetStream KV bucket `jobs`, key `<job_id>` (same binary; value pinned in the Implementation plan) | must survive a monitor restart, or reconciliation would re-enqueue cancelled work |

### Leases, heartbeats and worker death

A worker holds a message via JetStream's **AckWait** — the time since the last
sign of life before the task is assumed dead and redelivered. AckWait is kept
**small and fixed** (e.g. 2 min). It is *not* set to the chunk's duration:

- While processing, the worker sends `msg.InProgress()` on a background task
  every ~30–60 s; each one resets AckWait. A 3-hour chunk holds its lease across
  ~360 heartbeats while AckWait stays at 2 min.
- If the worker dies, heartbeats stop and the task redelivers **within one
  AckWait window** — fast death detection with an unbounded work duration.
- The heartbeat is also the worker's **lease sensor**: it tracks the time since
  the last *successful* `InProgress`, and once that gap exceeds AckWait the
  lease has definitively expired server-side — the chunk is being redelivered
  elsewhere — so the worker **aborts the chunk** rather than finishing work it
  no longer owns. A zombie stops burning GPU within one heartbeat interval,
  not one chunk. (A single failed heartbeat during a brief blip aborts
  nothing; only a gap longer than AckWait does.)
- `MaxDeliver` (e.g. 3) caps redeliveries so a poison chunk dead-letters instead
  of looping forever.

Because processing is asyncio and LLM calls are `await`ed, a 10-minute request
does not block the event loop, so the `InProgress` task keeps firing throughout.
**Caveat:** any blocking CPU work in a stage must run in an executor, or it will
stall the heartbeat and trigger a false death.

### Graceful shutdown (drain)

Planned shutdowns must not cost what crashes cost. Pressing **`s`** in the
runner's terminal (or sending SIGTERM — the no-TTY path, e.g. `docker stop`)
drains the worker: it stops pulling new messages, finishes its in-flight
chunks (upload → done event → ack), then exits. Nothing redelivers, nothing
recomputes. Give the container a stop timeout of at least one chunk
wall time; anything harder (SIGKILL, power loss) is simply the crash path the
leases already handle. A rolling upgrade of the fleet is drain → deploy →
restart, box by box.

### Heartbeat is liveness, not success

The heartbeat only answers "is this worker alive?" — never "did the work
succeed." Success is the **final ack**, and it comes last:

1. Process the chunk (heartbeats firing).
2. Upload results to S3 and confirm the PUT.
3. *Only then* `ack()`.

So a live worker that fails to save to S3 does **not** ack — it `nak()`s and the
task redelivers, recomputes, and retries the write (safe, per the backbone).
Layer it: the S3 client retries transient errors inline (5xx, throttling,
transport errors — exponential backoff from 2 s, `DC_S3_RETRIES` attempts,
default 6, each sleep capped at `DC_S3_RETRY_MAX_WAIT`, default 60 s; a missing
key is an answer and never retried); still failing, the
worker `nak`s **with a backoff delay** so a brief S3 outage self-heals without
burning `MaxDeliver`; past `MaxDeliver` the chunk dead-letters and the monitor
surfaces it. A *flood* of failures is an outage signal (alert + back off), not a
poison chunk. Because each output is one atomic S3 object, the write either
fully lands or does not — there is nothing partial to reconcile.

### Completion handoff

"Finished" must mean **in S3**, never "the worker said so", or a chunk could be
written but lost before the next stage is enqueued. The worker **publishes the
done event to a durable stream before acking** the work message (publish-then-
ack): die in between and the work message redelivers, recomputes, and
republishes the done event. `Nats-Msg-Id` dedup only swallows duplicates inside
the stream's duplicate window (default ~2 min) — shorter than a recompute — so
the monitor treats done events as **hints, not truth**: before enqueueing stage
N+1 for a chunk it consults its own state and HEADs the chunk's output key in
S3. This is per chunk — never "all chunks cleared stage N"; there is no stage
barrier. A duplicate enqueue that still slips through is harmless
(skip-if-done). As a backstop the monitor reconciles from S3 wholesale (on
restart, or periodically) — events drive latency, S3 drives correctness.

### Cancel and editing a running job

Cancel = **purge the WorkQueue stream**, so nothing more is pulled, and mark
the job cancelled in its **durable job record** — the record, not monitor
memory, is what reconciliation consults, so a monitor restart cannot resurrect
a cancelled job. In-flight chunks finish into their keys and are simply never
consumed.

Editing a running job = **purge and re-seed** with the new plan. Safety comes
from sig-addressed keys, not from draining the fleet: edited stages (and, via
the sig chain, everything downstream) get new sigs — new messages, new
`Nats-Msg-Id`s, new output prefixes. A straggler still finishing an old-plan
chunk writes into an old-sig prefix: dead data, never a clobber, so the admin
re-seeds immediately and never waits for in-flight work. Unchanged upstream
stages keep their sigs and reuse their outputs as-is — and because outputs are
sig-addressed, the controller can tell the admin exactly which chunks a
proposed edit invalidates before committing to the recompute. The controller
GCs abandoned prefixes.

### Single points of failure (accepted)

Both coordination pieces are deliberately unreplicated. Neither can lose *data*
— input, intermediates and results are all in S3 — only progress:

- **Monitor** (one instance) — death stalls stage handoff; nothing corrupts.
  Restart it: it reloads the job records, replays done events and reconciles
  against S3.
- **NATS box** — losing its disk loses queue state (pending messages, acks, job
  records). Recovery is the rerun path that already exists: start a fresh NATS,
  re-run the controller for each active job — it re-seeds only what S3 says is
  missing, and skip-if-done keeps finished chunks from recomputing. Worst case:
  in-flight chunks rerun.

### Versioning

Code drift ("node 1 on an old library, node 2 on the latest") would break the
equivalence the backbone relies on. The version is pinned in a **root file** in
the repo (git sha / package version); the controller stamps it onto each job
and folds it into every stage's `requires` as a `ver:` tag. A mismatched worker
therefore **never binds the consumer** — the message is never pulled, so there
is no refuse/nak loop and no `MaxDeliver` burned on version mismatches. Workers
report their version in their heartbeat so the operator sees drift. (Config
drift is already impossible — behaviour travels in the message.)

Bumping the version is a **deliberate admin act with a precise meaning**: "the
outputs already produced are no longer equivalent to what this code writes."
A crash-fix does *not* require it — a bug that crashed chunks or flagged
records never contaminated a successful output (its victims are dead-lettered
or flagged, both visible), so the fix redeploys under the same version and
reprocesses only what is missing or flagged; nothing completed is lost. Bump
only when a fix changes what *already-successful* outputs should contain (a
silent-wrong-output bug) — then the sig chain flips and the full recompute is
the point, not a cost to engineer around.

### Worker image

The `Build images` workflow runs when a GitHub release is published and pushes
the worker and monitor images to GitHub Container Registry, tagged with the release
tag and, unless it is a pre-release, `latest`. The worker image has no `.git`, so the
build bakes the release tag in as `DC_VERSION`. Set the same value in the
controller's env file, otherwise it stamps jobs with the git SHA and no released
worker matches them. Connection settings and bindings are provided at runtime. A
`DC_VERSION` in the worker's env file overrides the built-in value, so a crash-fix
release can run under the version it fixes.

### Monitor image

The monitor image needs no `DC_VERSION`, because the monitor reads each job's
version from its record and never calls `resolve_version`. SIGTERM stops it
straight away, and its state rebuilds from the done stream, the job records and S3
on the next start. `DC_STALL_CRON` may be quoted, because `docker run --env-file`
keeps the quotes that `uv run --env-file` needs around a value with spaces.

## Capability routing

Workers and stages meet through free-form **tags**, matched as a set:

- A **stage** carries `requires: [<tag>, …]` (the controller derives it when
  rendering — e.g. an LLM stage → `model:Qwen/Qwen3.6-35B-A3B-FP8`, a CPU-only
  stage → `cpu`).
- A **worker** is launched advertising `capabilities: [<tag>, …]` — e.g.
  `[model:Qwen/…, model:bielik, cpu]` for a multi-model box, `[cpu]` for a
  runner with no GPU.

**Routing is pull-based — there is no dispatcher.** The stream has one durable
consumer per stage (a JetStream *consumer* is a named server-side cursor over
the stream, **not a process** — many workers share one); a worker binds only to
the consumers whose `requires` its tags satisfy, and pulls from them. Workers
discover them by periodically listing the stream's consumers (so jobs started
mid-run appear); the controller stamps `requires` into each consumer's
**metadata** at creation, and it never changes — an edited job creates new
consumers. A task for a `model:X` stage can only be pulled by a worker
advertising `model:X`; incapable workers never ask, so no per-task lookup and
no registry is needed. This gives **capability = capacity** for free:
if no worker serves stage X, its messages simply wait in the queue until a
capable node comes online. Adding a capability = binding another consumer, live;
losing a node = its unacked tasks redeliver elsewhere. No membership protocol.

Workers *announce* capabilities only for **observability** — a periodic heartbeat
to a `fleet.heartbeat` subject (`{worker_id, capabilities, in_flight,
version}`) that the monitor consumes to show the fleet and fire **stall alerts**
(queue depth with zero capable pullers). This is ephemeral pub/sub; nothing
persists.

The heartbeat also carries the box's CPU/RAM/GPU usage. A worker that only
relays to a remote LLM endpoint can turn this off with `system_metrics: false`
at the top level of its bindings file (see `bindings.example.yaml`). Its
heartbeat still reports liveness, in-flight and version, but the monitor never
creates CPU/RAM/GPU gauges for it.

### LLM endpoints: local or shared

A tag like `model:X` means "I can serve stages needing model X" and is decoupled
from *where* X runs. A GPU box advertises it because it hosts vLLM locally
(`api_base: http://localhost:8000/v1`); a runner-only box could advertise it too
if pointed at a shared endpoint, or advertise only `cpu` and never touch LLM
stages. The worker supplies the matching `api_base`/`api_key` from its env
(behaviour/bindings split), so the same task runs correctly whether the model is
local or shared — the design does not need to know which.

## Batched LLM requests

A stage-level **`batch` toggle** packs N records into one completion. The
record loop aggregates the group, a **batch-aware prompt** renders all N
items into one request, the model returns a **list of response models**, and
each output is matched back to its record by an echoed item id. No `batch`
key is today's one-record-one-request path.

```yaml
pick_the_best_translation:
  batch:
    size: 5            # records per request
    on_failure: flag   # a dead batch flags all its records; `single`
                       # retries each record as a 1-item batch first
```

The win is prompt economy, not server parallelism — vLLM already batches
across requests server-side. One shared prefix is processed once instead of
N times and per-request overhead is divided by N. It is also a **quality
trade**: sibling items sit in each other's context and per-item attention
thins as N grows — start small (~5) and judge with a canary, like any
behaviour change.

The LLM stage's per-record options carry over item by item: `repairs` run on
each item before its validators, and `on_guard_failure: keep` keeps an item's
rejected output (never a misaligned item's). `reprompt_on_truncation` answers a
truncated batch with one correction turn for every pending item.

### The prompt is written for batching — not wrapped into it

Batching is opt-in per stage because the prompt must be authored for it: a
generic wrapper that numbers N single-record prompts was rejected — how the
items are laid out and instructed over is prompt engineering, and it belongs
in the template, not in code. With the toggle on:

- **Templates receive `items`** — the list of per-record contexts (each the
  same variables a single-record render sees today, post-modifier, plus its
  `item_id`) — and iterate over it. One template handles every group size,
  so chunk tails and the single-record fallback render from the same file;
  `StrictUndefined` still catches missing fields at render.
- **The response schema is the list.** The stage generates a wrapper —
  `{items: [{item_id, <response_model>}] × N}` — and the model must return
  exactly one item per input, each opening with the `item_id` it answers. A
  count mismatch is an `UnparseableOutputError` for the whole batch, retried
  like any parse failure. Outputs are matched to inputs **by the echoed id,
  not by position**: a truthfully misordered batch is realigned, while a
  missing or duplicated id fails only the affected items
  (`BatchItemMismatchError`) — positional matching let a model that answered
  a batchmate twice corrupt records silently. The id is an enum the model
  chooses freely; forcing the right id per position in the grammar would
  stamp correct ids over wrong content and hide the swap. The echo is
  stripped from the output before it reaches the record.
- **`max_tokens` is per request** — scale it with `size`, or truncation
  reads as parse failures.

### Per-record semantics are preserved around the group

The group is an execution detail between the record loop and the request;
every contract stays per record:

- **Grouping.** The worker slices a chunk's runnable records (post-filter,
  not flagged) into groups of `size`. A group never spans chunks — so it can
  never mix jobs, plans or specs.
- **Modifiers** run per item, automatically: `before` on each record ahead
  of the render, `after` on each record once its output is split back.
  Existing modifiers work unchanged; batching is invisible to them.
- **Validators** run per item, each against its own record's context. Only
  the failing items are re-sent — re-batched together into a smaller group —
  while passing items keep their outputs; nothing valid is resampled.
- **Reprompting** (`stage.reprompt_attempts > 0`) upgrades the re-send to a
  correction round in the *same conversation*: the failed batch answer is
  replayed as an assistant turn, a user turn lists each rejected item's
  validator feedback under its **original id**, and the follow-up schema is a
  wrapper over exactly those ids. Transport/parse failures blindly re-send
  the same conversation (fresh `retries` budget per turn); items failing
  *without* feedback (unreachable checker) leave the conversation and go
  through the blind re-batch instead. The `single` fallback opens a fresh
  conversation with fresh budgets. Every record of a reprompt-enabled stage
  carries `<output_field>_reprompt_rounds` — the correction-round count, `0`
  when accepted on the first batch pass — flagged records included.
- **Failure** exhausts retries per `on_failure`: `flag` marks every record
  of the dead batch (simple; batchmates pay for a poison item), `single`
  first retries each record as a 1-item batch, so only records that fail on
  their own merits flag. Flagged records carry the same error fields either
  way; `_prov` stays one entry per record.

### What it changes elsewhere: almost nothing

- **`batch` is behaviour.** The prompt and schema differ under it, so it
  renders into `stage_spec` and is hashed into the sig: changing `size`
  recomputes the stage and everything downstream. Tune it before seeding,
  not mid-job.
- **Per-record maps still hold.** Each output is produced from its own
  record's fields under the same instruction; batchmates shifting the
  sampling is the same accepted effect as vLLM's server-side batching (see
  equivalent recompute). A rerun that groups differently is equally valid,
  so group boundaries carry no contract and are never recorded.
- **Concurrency.** Worker concurrency still counts records — a group holds
  `size` slots for the duration of its request — so requests in flight ≈
  concurrency / size and the demand-driven pull rule is unchanged. A batched
  request runs ~N× longer (N outputs, one stream): the granularity table's
  chunk wall time stays the same aggregate, lumpier per request.
- **Scope.** The render/split/validate machinery lives in `LLMStage`; only
  the grouping lives in the worker's record loop. `PipelineRunner` stays
  unchanged (non-goal) — standalone batching would be the same grouping
  added there, nothing more.

## Granularity

Two sizes are kept independent so they do not fight:

1. **Chunk size** — records per message/file (global; deterministic keys).
2. **Worker concurrency** — in-flight records per VM; the knob that makes a
   multi-GPU box faster.

There is deliberately **no prefetch knob**. The worker pulls demand-driven:
whenever in-flight records < concurrency, it pulls one more message and starts
its records immediately. This one rule covers everything prefetch would have
done: a 32-slot box naturally runs two 20-record chunks at once; chunk
executions **overlap**, so the end-of-chunk straggler tail never idles the GPU
(slots freed by chunk A's last slow records are already running chunk B); and
a worker never holds a lease on work it is not actively executing, so nothing
is hoarded from the rest of the fleet.

Chunk size is the lever for **wasted work on death**. With ~10-min LLM requests,
chunk wall time ≈ `(chunk_size / concurrency) × 10 min`, and a node dying near
the end reruns the whole chunk:

| chunk size | concurrency | ~wall time | rework if it dies at 90% |
|---|---|---|---|
| 500 | 8  | ~10 h   | ~9 h |
| 500 | 32 | ~2.6 h  | ~2.3 h |
| 50  | 32 | ~20 min | ~18 min |
| 20  | 20 | ~10 min | ~9 min |

So **size chunks small — tens of records** for an expensive stage. The message
count is a non-issue (1M records at 20/chunk ≈ 50k messages, trivial for NATS),
and smaller chunks give the monitor finer progress. Use **one global chunk size
tuned to the most expensive stage** (the LLM one); cheap stages ride along.
Per-stage re-chunking was rejected — it breaks clean deterministic keys across
stages for no real gain.

Held in reserve: if small chunks still waste too much GPU time, add **intra-chunk
checkpointing** — the worker flushes completed records as it goes (as
`_ShardWriter` already does, `runner.py:270`) and skips done ids on redelivery,
bounding rework to the in-flight records. It slightly muddies the pure-executor
model, so it is opt-in, not the default.

## Observability

**Errors are data; logs answer "why".** The primary failure signals are durable
and queryable with no log infrastructure at all: flagged records carry their
error fields (and traceback) in the dataset itself, dead-letters name their
`(job, stage, chunk)`, done events and `_prov` narrate what ran where and with
what code. Logs exist only to explain a failure those signals already located.

### Alerts (monitor → ntfy)

Each alert kind has an ntfy priority, overridable per kind with
`DC_NTFY_PRIORITY_<KIND>` (`min`, `low`, `default`, `high`, `urgent`, or
`1`–`5`; an invalid value logs a warning and keeps the default):

| Kind | Variable | Default |
|---|---|---|
| Dead-letter (and drained job) | `DC_NTFY_PRIORITY_DEAD_LETTER` | `high` |
| Done event without an S3 object | `DC_NTFY_PRIORITY_GHOST` | `urgent` |
| Stall (and the `DC_STALL_CRON` digest) | `DC_NTFY_PRIORITY_STALL` | `high` |
| Flagged-rate spike | `DC_NTFY_PRIORITY_FLAGGED` | `default` |
| Redelivery-rate spike | `DC_NTFY_PRIORITY_REDELIVERY` | `default` |
| Job complete | `DC_NTFY_PRIORITY_COMPLETE` | `default` |

- **Dead-letter** — the monitor consumes the `MAX_DELIVERIES`
  advisory and alerts with the full correlation key: job, stage, chunk,
  attempts, and which workers failed it. The worker set is the first
  diagnostic: a chunk that failed on three different boxes is a code/data
  bug; one box failing chunks that others handle fine is a sick node.
- **Drained job** — an active job with every stage queue empty and nothing
  unacked, yet incomplete, for `DRAINED_TICKS` (3) consecutive reconcile
  passes. The advisory above is not persisted, so a monitor that was down when
  a chunk dead-lettered never hears of it. This check rebuilds the lost chunks
  from the S3 done sets instead: every chunk missing from stage 0, and per
  later stage the chunks done at the previous stage but not at this one. It
  records them as dead-letters (`dc_chunks_deadlettered`, `/api`) and alerts
  with the chunk ids and the `run_controller.py rerun <job>` command. The alert
  repeats hourly while the job stays stuck.
- **Stall** — queue depth > 0 with zero capable pullers (capability routing).
  One ntfy per `(job, stage)` per hour by default; set `DC_STALL_CRON` (a cron
  expression) to batch every currently-stalled stage into a single digest fired
  on that schedule instead.
- **Flagged-rate spike** — the monitor aggregates flagged records by
  `(stage, error_type)`; a spike in one signature reads as a code bug, a
  trickle of validator errors as model noise.
- **Redelivery-rate spike** — zombie storm: AckWait too tight, or an event
  loop blocked by un-executored CPU work.
- **Done event without an S3 object behind it** — should never fire; it means
  the upload-before-publish contract broke.

### Logging

- **Correlation key on every line.** The worker binds
  `{worker, job, stage, chunk, sig}` into loguru context for the duration of
  a chunk (record id inside the loop), so the jump from any alert to the
  exact lines is a filter, not an investigation.
- **Full logs stay local** — rotating files per worker, all levels; the only
  sink that survives a network incident, which is when logs matter most.
- **WARNING+ (and chunk lifecycle at INFO) push straight to Loki** from a
  loguru sink over Loki's HTTP push API (`POST /loki/api/v1/push`; the
  `python-logging-loki` package or a ~30-line hand-rolled batching sink,
  attached with `logger.add(..., level="WARNING")`) — **fail-open**: logging
  must never block or crash the pipeline; if Loki is down, drop-and-count and
  rely on the local files. No per-VM agent. Labels stay low-cardinality
  (`worker`, `job`, `stage`); chunk and record ids live in the log line, not
  in labels. Search is LogQL in Grafana's Explore tab.
- Loki + Grafana (+ Prometheus) run **once**, on the observability box
  alongside the monitor — never on workers.

### Metrics

The **monitor is the single Prometheus scrape target** — no exporters or
agents on workers; it re-exports what already flows through it:

- per job/stage: chunks done / in-flight / dead-lettered, queue depth and
  redelivery counts (from JetStream consumer info), records flagged by
  `error_type`, throughput;
- fleet: workers alive, versions, in-flight per worker, plus the cheap gauges
  workers already carry in their heartbeats (GPU util, RAM), relayed per
  worker.

Grafana over Prometheus is the operator view — the standard dashboards: a
per-job **chunk burndown** per stage, flagged records by `error_type`, queue
depth / redeliveries per consumer, and a **fleet table** (worker, version,
in-flight, GPU/RAM).

### Monitor HTTP API

The monitor serves a read-only HTTP API next to `/metrics`, re-exporting what
already flows through it, so no job metadata re-enters S3.

- `GET /api/jobs` — summaries. `GET /api/jobs/{id}` — the full view:
  per-stage chunk progress, fleet table, flagged records by `error_type`,
  dead-lettered chunks, queue depth.
- `GET /api/jobs/{id}/artifacts` — the final stage's **live-sig chunk keys**,
  straight from the plan. The monitor never lists a dead-sig prefix, and a
  re-seeded job's artifact view flips atomically to the new plan.
- `GET /api/stream` — SSE deltas (chunk done, fleet and state changes).
- Static bearer token (`DC_PANEL_TOKEN`). Without one the API stays closed;
  `/metrics` stays unauthenticated for Prometheus.

### Canary

Before seeding a full job, seed a limited one (a few chunks) and inspect its
output with `helpers/browse_jsonl_s3.py`. Cheap insurance against the one bug
class nothing above catches automatically: code that runs fine and writes
wrong output.

## Non-goals

- No change to `run_pipeline.py` or `PipelineRunner` behaviour.
- Stages are pure **per-record maps** — no cross-record/global/reduce stages
  (dedup, global sort, corpus statistics). This is what makes chunking valid.
  (Batched LLM requests stay inside this: an execution grouping, not a
  cross-record stage.)
- No fan-in DAG (the pipeline is linear); no per-record queuing.
- No per-VM file sizing, no chunk→file index, no read-time dedup — not needed
  with deterministic keys.
- No auto-provisioning of VMs or model deployment (operator concern).
- No fleet-wide pause between stages. The standalone `runner.pause_between_stages`
  (`runner.py:762`) is a manual model-swap gate for the single-GPU box;
  heterogeneous models are handled by capability routing instead — there is no
  point at which the whole fleet stops to swap a model.

## Implementation plan

Everything above is the contract; this section pins the remaining choices so
the build is assembly, not design. Never invent new architecture (a new
store, a new daemon, a new message type) to get unstuck.

### Ground rules

- `run_pipeline.py`, `PipelineRunner` and everything under
  `DataCurator/pipeline/` stay untouched (non-goal #1). All new code lives in
  `DataCurator/distributed/` plus three entry scripts at the repo root.
- **Reuse, don't fork:** `Stage.run()` (`pipeline/stage.py:205`) and
  `hydra.utils.instantiate` of a stage node (as `build_pipeline` does,
  `runner.py:100`); the flag-field shape of `_flag_record` (`runner.py:690`);
  the filter/drop-fields semantics of `_select_input` (`runner.py:667`);
  `S3Sink` (`reporting/s3.py`) for all S3 IO; `pipeline/notifications.py` for
  ntfy. Copy a small private helper if importing it is awkward — never modify
  the originals.
- The worker and monitor are **asyncio**; boto3 calls run via
  `asyncio.to_thread`. `except Exception` only, never bare `except` — the
  lease-sensor abort is a task cancellation and must propagate.
- New dependencies: `nats-py`, `prometheus-client`. Nothing else until the
  monitor API phase (which may add `starlette`/`uvicorn`).

### Module layout

```
DataCurator/distributed/
  wire.py        # message + job-record dataclasses, schema_version,
                 # subject/stream/key name builders, the sig chain
  s3io.py        # chunk JSONL read/write + HEAD, over S3Sink
  controller.py  # seed / status / rerun / cancel / gc
  monitor.py     # done-event loop, handoff, reconcile, alerts, metrics
  worker.py      # consumer discovery, pull loop, record loop, leases, drain
run_controller.py   # CLI over controller.py
run_monitor.py
run_distributed.py  # the worker
```

### Pinned names

| Object | Name / subject | Notes |
|---|---|---|
| Work stream (per job) | `JOB_<job_id>`, subjects `job.<job_id>.stage.*` | **WorkQueue retention.** JetStream allows multiple consumers on a WorkQueue stream only with non-overlapping filters — one consumer per stage satisfies that by construction. |
| Stage consumer | durable `s<NN>`, filter `job.<job_id>.stage.<NN>` | `requires` + `stage_id` in consumer **metadata**, set at creation, never changed. |
| Work message id | `Nats-Msg-Id: <job>:<sig>:<chunk>` | dedup across re-seeds/re-publishes. |
| Done stream (per job) | `JOB_<job_id>_EV`, subject `job.<job_id>.done` | limits retention, `max_age` 30 d; msg-id `<job>:<sig>:<chunk>:done`. |
| Job records | JetStream **KV bucket `jobs`**, key `<job_id>` | value below. |
| Fleet heartbeats | plain subject `fleet.heartbeat` | core pub/sub, no stream. |
| Dead letters | advisory `$JS.EVENT.ADVISORY.CONSUMER.MAX_DELIVERIES.JOB_<job_id>.*` | consumed by the monitor. |

`<job_id>` is the experiment name (the standalone reporter's convention);
validate it against `[A-Za-z0-9_-]+` at seed time — it is embedded in
subjects and stream names. `<NN>` is the zero-padded stage index (`00`…);
chunk ids are `chunk-%06d`.

**The job record** (KV value, JSON): `schema_version`, `job_id`, `version`,
`state` (`active | cancelled | complete`), `chunk_count`, `input_prefix`, and
`stages` — an ordered list of `{index, stage_id, sig, requires, stage_spec}`.
It is the monitor's single source of topology and, with S3, everything a
monitor restart needs.

### Pinned parameters

| Knob | Default |
|---|---|
| `chunk_size` | 20 records |
| AckWait | 120 s |
| `InProgress` interval | 30 s |
| Lease-sensor abort / pre-upload check | last successful `InProgress` older than AckWait → abort chunk; older than AckWait ÷ 2 → skip the upload |
| MaxDeliver | 3 |
| Duplicate window | 120 s |
| Nak backoff | 60 s, ×2 per redelivery, cap 15 min |
| Outage breaker | > 50 % of a chunk's runnable records flagged with a connection-class error |
| Consumer re-discovery | every 30 s |
| Worker concurrency | env; counts records in flight; pulls round-robin across bound consumers |

### The sig chain, exactly

```python
canonical = json.dumps(OmegaConf.to_container(stage_node, resolve=True),
                       sort_keys=True, separators=(",", ":"))
sig_n = sha256(f"{version}|{sig_prev}|{canonical}".encode()).hexdigest()[:8]
# stage 0 uses the literal "input" as sig_prev
```

The controller **refuses to seed** a spec whose canonical form contains a
binding key (`api_base`, `api_key`, `remote_server`, anything matching
`*_key` / `*_secret` / `*_token`) — the behaviour/bindings split is enforced,
not hoped for. The worker injects bindings at instantiate time from its env:
a `{model name → {api_base, api_key}}` map, the LanguageTool URL, S3
credentials. Its capabilities are **derived**, not hand-listed: `model:<name>`
per entry of that map, plus `cpu`, plus `ver:<version>`.

`version` is the `DC_VERSION` env var when set, which is the release tag on the
worker image, otherwise `git rev-parse --short=7 HEAD` at process start. Record ids: the controller keeps dataset
order, uses the experiment's `id_field` when present, otherwise assigns
`<row_index:08d>` — either way ids are frozen into `input/` before any sig is
computed.

### Controller CLI

```
run_controller.py seed   <experiment> [--max-chunks N]   # canary = small N
run_controller.py status <job_id>    # per-stage done/missing under current sigs
run_controller.py rerun  <job_id> [--chunks 000042,000043 --force]
run_controller.py cancel <job_id>
run_controller.py gc     <job_id>    # delete dead-sig prefixes
```

Seed order (idempotent; each step converges on re-run): read dataset → assign
ids → upload `input/` chunks → render specs, compute sigs → create stream +
consumers (create-if-absent) → publish stage-0 messages (msg-id dedup) →
**write the job record last**.

### Build phases

Each phase ends runnable and verified before the next starts. Phases 1–7 use
a test pipeline of two or three trivial CPU stages (an uppercase stage, a
sleep stage) — no GPU, no LLM. The dev harness is `nats-server -js` + MinIO
(or Garage) in docker-compose, two worker processes, and the real
controller/monitor entries.

1. **`wire.py` + controller seed.** Sigs, schemas, name builders; `seed`
   produces input chunks, consumers, stage-0 messages and the job record.
   Unit tests: editing stage N flips sigs N…end and only those;
   canonicalisation is stable across key order.
2. **Minimal worker.** Discover consumers, pull, skip-if-done HEAD, download,
   run the stage per record, upload, publish done, ack. A one-stage test job
   completes on two workers.
3. **Monitor handoff.** Done-event loop, HEAD-verified enqueue of stage N+1,
   completion detection, reconcile from S3 on start and every 60 s. A
   multi-stage test job completes; killing and restarting the monitor mid-job
   duplicates no compute.
4. **Failure semantics.** `InProgress` heartbeats, lease-sensor abort,
   pre-upload lease check, nak-with-backoff, outage breaker, dead-letter
   advisory consumption, drain on SIGTERM/`s`. Chaos tests: `kill -9` a
   worker at 90 % of a chunk; wipe NATS and re-seed; cancel mid-run.
5. **Record-loop parity.** Flagged pass-through, catch-all flag fields,
   `on_filtered: drop | pass`, `drop_fields`, `_prov`. Parity test: the same
   test dataset through the standalone runner and the distributed path yields
   the same records (modulo `_prov` and ordering).
6. **Controller ops.** `rerun`, `--force`, `cancel`, `gc`, `status`,
   `--max-chunks` canaries; edit-and-re-seed reuses unchanged upstream sigs.
7. **Observability.** Prometheus metrics from the monitor (the gauges and
   counters listed under Metrics), the five ntfy alerts, the Loki sink and
   correlation-key logging on the worker.
8. **Real fleet.** An LLM experiment with bindings split out of the config;
   one GPU box + one cpu-only box; verify capability routing (the cpu worker
   never pulls a model stage) and version fencing (a mismatched worker pulls
   nothing).
9. **Batched LLM requests** (per the Batched section) — only after 1–8 hold.
10. **HTTP API** on the monitor (per the Monitor HTTP API section).

### Acceptance checklist

- [ ] Standalone path untouched: no diff under `DataCurator/pipeline/` or in
      `run_pipeline.py`.
- [ ] A 3-stage × 50-chunk test job completes on 2 workers; a mid-run snapshot
      of chunk×stage state is ragged (no stage barrier).
- [ ] `kill -9` a worker mid-chunk → the job completes; the rerun chunk's
      records carry `attempt: 2` in `_prov`.
- [ ] Monitor restart mid-job → the job completes with no chunk recomputed.
- [ ] NATS state wiped → `seed` again → only unfinished chunks recompute.
- [ ] Editing stage 2 of 3 → stage-1 outputs reused, stages 2–3 recomputed,
      `gc` removes the dead prefixes.
- [ ] `cancel` → no new pulls, in-flight chunks finish, a monitor restart
      does not resurrect the job.
- [ ] A record flagged at stage 0 appears in the final output unprocessed,
      its error fields intact.
- [ ] A chunk fully emptied by `drop` filters still uploads an (empty) object
      and the job completes.
- [ ] Drain (`s`/SIGTERM) exits after in-flight chunks finish; nothing
      redelivers.

## Deferred / open

- **S3 security.** Data may be publicly readable, but access is still gated by
  credentials: least-privilege, per-bucket Garage keys (read input, read/write
  `jobs/<id>/`), TLS to the endpoint, secrets via each role's env file — never in a message or
  committed.
- **NATS security.** The internet-facing server runs with `authorization` on:
  one user/password per node, listed in the gitignored
  `distributed-infra/nats/nats-users.conf` (SIGHUP reloads). Nodes authenticate via
  `NATS_USER`/`NATS_PASSWORD` in their role's env file (`NATS_TOKEN`/`NATS_CREDS` also
  supported). TLS terminates at the proxy in front of `:4222`; the
  unauthenticated `:8222` monitoring port is loopback-only.
