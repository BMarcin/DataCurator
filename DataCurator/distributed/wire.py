"""Wire format and naming contract for the distributed pipeline.

Everything two processes must agree on lives here: message and job-record
schemas, NATS subject/stream/consumer names, S3 key layout, the chained
stage signature, the behaviour/bindings policy and the pinned parameters
from wiki/Distributed.md. Nothing in this module talks to the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- #
# Pinned parameters (wiki/Distributed.md "Pinned parameters")
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 20                 # records per chunk
ACK_WAIT_S = 120                # lease: time since last sign of life before redelivery
INPROGRESS_INTERVAL_S = 30      # worker heartbeat on in-flight messages
MAX_DELIVER = 3                 # redeliveries before a chunk dead-letters
DUPLICATE_WINDOW_S = 120        # JetStream msg-id dedup window
NAK_BACKOFF_BASE_S = 60         # nak delay: base, doubled per redelivery, capped
NAK_BACKOFF_FACTOR = 2
NAK_BACKOFF_CAP_S = 900
OUTAGE_FLAG_RATIO = 0.5         # > this fraction of a chunk connection-flagged -> outage, nak
REDISCOVER_INTERVAL_S = 30      # worker re-lists consumers (jobs started mid-run)
RECONCILE_INTERVAL_S = 60       # monitor reconciles against S3
DONE_STREAM_MAX_AGE_S = 30 * 24 * 3600
JOBS_KV_BUCKET = "jobs"
FLEET_HEARTBEAT_SUBJECT = "fleet.heartbeat"
ERROR_TRACEBACK_LIMIT = 4000    # chars kept of a flagged record's traceback

# Fields the worker owns on a record; stages never see them.
PROV_FIELD = "_prov"

JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# --------------------------------------------------------------------------- #
# Names: NATS subjects/streams/consumers and S3 keys
# --------------------------------------------------------------------------- #
def work_stream(job_id: str) -> str:
    return f"JOB_{job_id}"


def work_subject(job_id: str, stage_index: int) -> str:
    return f"job.{job_id}.stage.{stage_index:02d}"


def work_subject_wildcard(job_id: str) -> str:
    return f"job.{job_id}.stage.*"


def consumer_name(stage_index: int) -> str:
    return f"s{stage_index:02d}"


def done_stream(job_id: str) -> str:
    return f"JOB_{job_id}_EV"


def done_subject(job_id: str) -> str:
    return f"job.{job_id}.done"


def work_msg_id(job_id: str, sig: str, chunk_id: str) -> str:
    """Dedup id for a work message — stable across re-seeds and re-publishes."""
    return f"{job_id}:{sig}:{chunk_id}"


def done_msg_id(job_id: str, sig: str, chunk_id: str) -> str:
    return f"{job_id}:{sig}:{chunk_id}:done"


def chunk_id_str(index: int) -> str:
    return f"{index:06d}"


def job_prefix(prefix: str, job_id: str) -> str:
    return f"{prefix}/{job_id}"


def input_prefix(prefix: str, job_id: str) -> str:
    return f"{prefix}/{job_id}/input/"


def input_key(prefix: str, job_id: str, chunk_id: str) -> str:
    return f"{prefix}/{job_id}/input/chunk-{chunk_id}.jsonl"


def stage_dir_prefix(prefix: str, job_id: str, stage_index: int, stage_id: str) -> str:
    """The per-stage prefix holding one subtree per sig (live and dead)."""
    return f"{prefix}/{job_id}/stage/{stage_index:02d}_{stage_id}/"


def stage_sig_prefix(prefix: str, job_id: str, stage_index: int, stage_id: str, sig: str) -> str:
    return f"{prefix}/{job_id}/stage/{stage_index:02d}_{stage_id}/{sig}/"


def stage_key(prefix: str, job_id: str, stage_index: int, stage_id: str, sig: str, chunk_id: str) -> str:
    return f"{prefix}/{job_id}/stage/{stage_index:02d}_{stage_id}/{sig}/chunk-{chunk_id}.jsonl"


# --------------------------------------------------------------------------- #
# NATS connection
# --------------------------------------------------------------------------- #
def nats_url() -> str:
    return os.environ.get("NATS_URL", "nats://localhost:4222")


def nats_connect_options() -> Dict[str, Any]:
    """Authentication kwargs for ``nats.connect``, from the environment.

    An internet-facing NATS runs with authorization on; every node (worker,
    controller, monitor) authenticates the same way: ``NATS_USER`` +
    ``NATS_PASSWORD``, or ``NATS_TOKEN``, or an ``NATS_CREDS`` file
    (JWT/nkeys). Credentials embedded in ``NATS_URL`` itself
    (``nats://user:pass@host``) also work without any of these.

    ``NATS_TLS_FIRST=1`` makes the client start with the TLS handshake instead
    of waiting for INFO — required when a proxy terminates TLS in front of a
    plaintext NATS (the proxy waits for a ClientHello, the standard flow would
    deadlock until timeout).
    """
    options: Dict[str, Any] = {}
    if os.environ.get("NATS_TLS_FIRST", "").lower() in ("1", "true", "yes"):
        options["tls_handshake_first"] = True
    user = os.environ.get("NATS_USER")
    password = os.environ.get("NATS_PASSWORD")
    token = os.environ.get("NATS_TOKEN")
    creds = os.environ.get("NATS_CREDS")
    if user and password:
        options["user"] = user
        options["password"] = password
    elif token:
        options["token"] = token
    if creds:
        options["user_credentials"] = creds
    return options


# --------------------------------------------------------------------------- #
# Version pinning
# --------------------------------------------------------------------------- #
def resolve_version() -> str:
    """The code version stamped onto jobs and advertised by workers.

    ``DC_VERSION`` overrides (non-git deploys); otherwise the short git sha of
    the checkout. One opaque token used everywhere — the sig chain, the
    ``ver:`` capability tag, messages and ``_prov`` — so equality is the only
    operation ever performed on it.
    """
    env = os.environ.get("DC_VERSION")
    if env:
        return env
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


# --------------------------------------------------------------------------- #
# Behaviour/bindings split enforcement
# --------------------------------------------------------------------------- #
BINDING_KEYS = {"api_base", "api_key", "remote_server", "client_name"}
BINDING_KEY_SUFFIXES = ("_key", "_secret", "_token")


# Model keys that describe the serving hardware, not the output: how many
# requests one worker keeps in flight and how long it waits for one. They are
# stripped from rendered specs (so they never enter a sig) and each worker
# supplies them per model from its bindings file.
MODEL_SERVING_KEYS = ("concurrency", "timeout")

# The same for Google Translate: how hard one box may hit the service depends
# on its network, not the job. A worker supplies them from the
# ``google_translate:`` section of its bindings file.
GOOGLE_TRANSLATE_SERVING_KEYS = ("concurrency", "timeout", "retries")
# Where the service lives and how to call it: required from the same section.
GOOGLE_TRANSLATE_BINDING_KEYS = ("remote_server", "client_name")
GOOGLE_TRANSLATE_TARGETS = ("stages.google_translate.GoogleTranslateStage", "google_translate.GoogleTranslateModifier")


def is_google_translate(node: Any) -> bool:
    """True for a stage or modifier spec that instantiates a Google Translate client."""
    return isinstance(node, dict) and str(node.get("_target_", "")).endswith(GOOGLE_TRANSLATE_TARGETS)


def google_translate_nodes(container: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The Google Translate specs in a rendered stage node: the stage itself and its modifiers."""
    stage = container.get("stage") or {}
    return [node for node in [stage, *(stage.get("modifiers") or [])] if is_google_translate(node)]


def strip_serving_keys(container: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the per-worker serving knobs from a rendered stage node: the model's
    :data:`MODEL_SERVING_KEYS` and every Google Translate spec's
    :data:`GOOGLE_TRANSLATE_SERVING_KEYS`."""
    model = (container.get("stage") or {}).get("model")
    if isinstance(model, dict):
        for key in MODEL_SERVING_KEYS:
            model.pop(key, None)
    for node in google_translate_nodes(container):
        for key in GOOGLE_TRANSLATE_SERVING_KEYS:
            node.pop(key, None)
    return container


def is_binding_key(key: str) -> bool:
    return key in BINDING_KEYS or key.endswith(BINDING_KEY_SUFFIXES)


def strip_null_bindings(node: Any) -> Any:
    """Recursively drop binding keys whose value is ``None``.

    A ``None`` binding is an artifact of composing with ``bindings=none``
    (OmegaConf cannot *omit* an interpolated key, only null it) — provably
    unbound, safe to remove. Non-null binding values are left in place for
    :func:`find_binding_keys` to refuse.
    """
    if isinstance(node, dict):
        return {
            key: strip_null_bindings(value)
            for key, value in node.items()
            if not (is_binding_key(key) and value is None)
        }
    if isinstance(node, list):
        return [strip_null_bindings(item) for item in node]
    return node


def find_binding_keys(node: Any, path: str = "") -> List[str]:
    """Dotted paths of binding keys present in ``node`` (after stripping)."""
    found: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if is_binding_key(key):
                found.append(here)
            found.extend(find_binding_keys(value, here))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            found.extend(find_binding_keys(item, f"{path}[{i}]"))
    return found


# --------------------------------------------------------------------------- #
# The sig chain
# --------------------------------------------------------------------------- #
def canonical(container: Any) -> str:
    """Stable serialisation of a rendered stage node for hashing."""
    return json.dumps(container, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def chain_sigs(version: str, canonicals: List[str]) -> List[str]:
    """Chained signatures: editing stage N flips sigs N..end and only those."""
    sigs: List[str] = []
    prev = "input"
    for canon in canonicals:
        sig = hashlib.sha256(f"{version}|{prev}|{canon}".encode()).hexdigest()[:8]
        sigs.append(sig)
        prev = sig
    return sigs


def derive_requires(stage_spec: Dict[str, Any], version: str) -> List[str]:
    """Capability tags a worker must advertise to pull this stage.

    An LLM stage (a ``model`` map with a ``name``) requires that model; any
    other stage runs anywhere (``cpu``). The ``ver:`` tag fences code drift —
    a mismatched worker never binds the consumer.
    """
    model = (stage_spec.get("stage") or {}).get("model")
    if isinstance(model, dict) and model.get("name"):
        base = f"model:{model['name']}"
    else:
        base = "cpu"
    return [base, f"ver:{version}"]


# --------------------------------------------------------------------------- #
# Messages and the durable job record
# --------------------------------------------------------------------------- #
@dataclass
class WorkMessage:
    """One ``(stage, chunk)`` task — a pointer; data lives in S3."""

    job_id: str
    stage_index: int
    stage_id: str
    chunk_id: str
    sig: str
    input_keys: List[str]
    output_key: str
    requires: List[str]
    version: str
    stage_spec: Dict[str, Any]
    schema_version: int = SCHEMA_VERSION

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "WorkMessage":
        return cls(**json.loads(data))

    @property
    def msg_id(self) -> str:
        return work_msg_id(self.job_id, self.sig, self.chunk_id)


@dataclass
class DoneEvent:
    """Published to the durable done stream *before* acking the work message.

    A hint, not truth — the monitor HEADs ``output_key`` before acting on it.
    ``worker``/``attempt`` make late overwrites (lineage skew) visible.
    """

    job_id: str
    stage_index: int
    stage_id: str
    chunk_id: str
    sig: str
    output_key: str
    worker: str
    attempt: int
    ts: str
    # Observability denormalisation (monitor aggregates without reading S3);
    # records == -1 means unknown (e.g. a skip-if-done republish).
    records: int = -1
    flagged: int = 0
    flagged_types: Dict[str, int] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "DoneEvent":
        return cls(**json.loads(data))

    @property
    def msg_id(self) -> str:
        return done_msg_id(self.job_id, self.sig, self.chunk_id)


@dataclass
class StagePlan:
    """One stage of a job's plan, as stored in the durable job record."""

    index: int
    stage_id: str
    sig: str
    requires: List[str]
    stage_spec: Dict[str, Any]


def build_work_message(
    job_id: str,
    version: str,
    stages: List["StagePlan"],
    prefix: str,
    stage_index: int,
    chunk_id: str,
) -> WorkMessage:
    """The one way a ``(stage, chunk)`` task is ever constructed — shared by
    the controller (seed/rerun) and the monitor (handoff), so keys, sigs and
    msg-ids can never diverge between them."""
    stage = stages[stage_index]
    if stage_index == 0:
        input_keys = [input_key(prefix, job_id, chunk_id)]
    else:
        prev = stages[stage_index - 1]
        input_keys = [stage_key(prefix, job_id, prev.index, prev.stage_id, prev.sig, chunk_id)]
    return WorkMessage(
        job_id=job_id,
        stage_index=stage.index,
        stage_id=stage.stage_id,
        chunk_id=chunk_id,
        sig=stage.sig,
        input_keys=input_keys,
        output_key=stage_key(prefix, job_id, stage.index, stage.stage_id, stage.sig, chunk_id),
        requires=stage.requires,
        version=version,
        stage_spec=stage.stage_spec,
    )


@dataclass
class JobRecord:
    """The durable job record (JetStream KV bucket ``jobs``, key ``<job_id>``).

    The monitor's single source of topology; with S3 it is everything a
    monitor restart needs. Written *last* during seeding, so a job exists
    only once fully seeded.
    """

    job_id: str
    version: str
    state: str  # active | cancelled | complete
    chunk_count: int
    input_prefix: str
    id_field: str
    stages: List[StagePlan] = field(default_factory=list)
    # The done stream's last sequence at (re-)seed time. Done events at or
    # below this floor predate the current plan: their *handoff* implication
    # ("stage N+1 was published for this chunk") may refer to dead sigs, so a
    # restarted monitor must not rebuild `enqueued` marks from them. Server-
    # side ordering — no clocks involved.
    done_floor: int = 0
    schema_version: int = SCHEMA_VERSION

    def to_bytes(self) -> bytes:
        return json.dumps(asdict(self), separators=(",", ":")).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "JobRecord":
        raw = json.loads(data)
        raw["stages"] = [StagePlan(**s) for s in raw.get("stages", [])]
        return cls(**raw)

    def stage(self, index: int) -> Optional[StagePlan]:
        return self.stages[index] if 0 <= index < len(self.stages) else None
