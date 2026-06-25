"""Hydra-driven runner that executes an ordered pipeline of stages.

The runner turns a Hydra config into a sequence of
:class:`~DataCurator.pipeline.stage.Stage` instances and feeds a JSONL
dataset through them. The *flow* is declared entirely in config: an
``order`` list names the stages to run, and a ``pipeline`` mapping holds
one entry per stage. Each entry carries runner metadata (``enabled``,
``model``) alongside the ``stage`` block that Hydra instantiates.

Behaviour, following the project's pipeline-design rules:

* **Sharded output (core).** Stage ``N`` reads the output of stage ``N-1``
  (or the dataset input — a single file, a glob, or a list — for the first
  stage) and writes its own output into a *directory*
  ``<output_dir>/<NN>_<stage>/`` as fixed-size shards ``step-0000.jsonl``,
  ``step-0001.jsonl``, … Each record is flushed as soon as it is produced,
  so a crash never loses completed work. When a shard fills (``shard_size``
  records) it is closed — becoming immutable — and, *if reporting is
  enabled*, uploaded to S3 as a downloadable artifact the instant it
  closes. The next stage reads every shard in order.
* **Resumable.** A sidecar ``<NN>_<stage>/meta.json`` stores a signature
  derived from the stage's config. On rerun, if the signature is
  unchanged, already-processed records (matched by ``id_field``, across all
  shards) are skipped and the last partial shard is continued; if the
  config/logic changed, the signature differs and the stage is recomputed
  from scratch.
* **Selectable.** Stages with ``enabled: false`` are skipped, and
  ``runner.start_from`` resumes the pipeline at a named stage by reading
  the previous stage's output.
* **Interruptible.** When ``runner.pause_between_stages`` is set the runner
  pauses for operator confirmation before each stage so manual steps (a
  model swap, a data check, a hardware change) can happen in between. A
  stage may override this with its own ``pause_before`` flag — set it on
  one stage to pause only there, or unset the global default to pause
  everywhere but a chosen stage.
* **Observable.** ntfy notifications fire at the configured points.
* **Fault-tolerant (configurable).** ``runner.flag_on_errors`` lists exception
  types that, when raised by a stage after its retries, flag the offending
  record with ``error: true`` (plus an explanation) instead of aborting the
  run; later stages pass flagged records through untouched. Any exception not
  listed still fails the pipeline.
"""
from __future__ import annotations

import asyncio
import glob
import hashlib
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import hydra
import orjson
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress as RichProgress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from DataCurator.pipeline.filters import RecordFilter
from DataCurator.pipeline.notifications import notify
from DataCurator.pipeline.stage import ModifierPhase, Stage, StageContext, errored_context
from DataCurator.reporting import JobState, NullReporter, Progress, Reporter, StageInfo


@dataclass
class StageSpec:
    """A built stage plus the runner-level metadata read from its config."""

    id: str
    stage: Stage
    enabled: bool
    signature: str
    pause_before: Optional[bool] = None  # per-stage override of runner.pause_between_stages
    # Input record selectors (all must keep a record) and bookkeeping keys to
    # strip from each input record before processing — applied to this stage's
    # input ahead of resume/limit. See `_select_input`.
    filters: List[RecordFilter] = field(default_factory=list)
    drop_fields: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Building the pipeline from config
# --------------------------------------------------------------------------- #
def _signature(stage_id: str, node: Any) -> str:
    """Hash a stage's resolved config so config/logic changes invalidate resume."""
    payload = OmegaConf.to_container(node, resolve=True) if isinstance(node, DictConfig) else node
    blob = orjson.dumps({"id": stage_id, "cfg": payload}, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(blob).hexdigest()[:16]


def build_pipeline(cfg: DictConfig) -> List[StageSpec]:
    """Instantiate the ordered list of stages declared by ``cfg``.

    ``cfg.order`` lists the stage ids to run; ``cfg.pipeline[id]`` holds
    each stage's config. The ``stage`` block of every entry is passed to
    :func:`hydra.utils.instantiate`; the rest is runner metadata.
    """
    pipeline = cfg.get("pipeline", {})
    specs: List[StageSpec] = []
    for stage_id in cfg.order:
        if stage_id not in pipeline:
            raise KeyError(f"stage {stage_id!r} listed in `order` but missing from `pipeline`")
        node = pipeline[stage_id]
        pause_before = node.get("pause_before")
        stage = hydra.utils.instantiate(node.stage)
        if not isinstance(stage, Stage):
            raise TypeError(
                f"stage {stage_id!r} must instantiate a Stage, got {type(stage).__name__}"
            )
        if not stage.name or stage.name == type(stage).__name__:
            stage.name = stage_id
        # Input filters are runner metadata (a sibling of `stage:`), instantiated
        # from their `_target_` like modifiers; filtering decides *which* records
        # run, which is the runner's concern, not the Stage's.
        filters = list(hydra.utils.instantiate(node.get("filters")) or [])
        for f in filters:
            if not isinstance(f, RecordFilter):
                raise TypeError(
                    f"stage {stage_id!r} filter must be a RecordFilter, got {type(f).__name__}"
                )
        drop_fields = [str(name) for name in (node.get("drop_fields") or [])]
        specs.append(
            StageSpec(
                id=stage_id,
                stage=stage,
                enabled=bool(node.get("enabled", True)),
                signature=_signature(stage_id, node),
                pause_before=None if pause_before is None else bool(pause_before),
                filters=filters,
                drop_fields=drop_fields,
            )
        )
    return specs


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file into a list of dicts, tolerating blank lines."""
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(orjson.loads(line))
    return rows


def _shards(stage_dir: Path) -> List[Path]:
    """Return a stage directory's shard files (``step-NNNN.jsonl``), ordered."""
    return sorted(stage_dir.glob("step-*.jsonl"))


def _read_records(src: Union[Path, List[Path]]) -> List[dict]:
    """Read all records from a stage input.

    ``src`` is either a list of files (the resolved dataset input for the first
    stage), a stage *directory* whose ``step-*.jsonl`` shards are read in order
    (every later stage), or a single file. Shards/files are concatenated in
    order, so the record sequence a downstream stage sees is stable.
    """
    if isinstance(src, list):
        rows: List[dict] = []
        for path in src:
            rows.extend(_read_jsonl(Path(path)))
        return rows
    src = Path(src)
    if src.is_dir():
        rows = []
        for shard in _shards(src):
            rows.extend(_read_jsonl(shard))
        return rows
    return _read_jsonl(src)


def _resolve_inputs(spec: Any) -> List[Path]:
    """Resolve the dataset ``input`` into a sorted, de-duped list of files.

    ``spec`` may be a single path, a glob pattern, or a list mixing the two.
    Glob entries (containing ``*``, ``?`` or ``[``) are expanded; literal paths
    are taken as-is. Used only for the first stage's input.
    """
    entries = [spec] if isinstance(spec, str) else list(spec)
    paths: List[Path] = []
    for entry in entries:
        text = str(entry)
        if any(ch in text for ch in "*?["):
            paths.extend(Path(match) for match in glob.glob(text))
        else:
            paths.append(Path(text))
    unique: List[Path] = []
    seen: set = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    unique.sort()
    if not unique:
        raise FileNotFoundError(f"dataset input {spec!r} matched no files")
    return unique


def _read_signature(meta_path: Path) -> Optional[str]:
    """Return the stored stage signature, or ``None`` if absent/unreadable."""
    if not meta_path.exists():
        return None
    try:
        return orjson.loads(meta_path.read_bytes()).get("signature")
    except (orjson.JSONDecodeError, OSError):
        return None


class _ShardWriter:
    """Appends records to rotating, fixed-size JSONL shards in a directory.

    Each shard (``step-NNNN.jsonl``) holds up to ``shard_size`` records. When a
    shard fills, :meth:`write` closes it — it is now immutable — and returns its
    path so the caller can publish it; the next ``write`` lazily opens the next
    shard. :meth:`scan` prepares the writer for resume by continuing the last
    partial shard (or starting fresh after a full one). The handle is opened
    lazily, so a stage with nothing left to do never creates an empty shard.
    """

    def __init__(self, stage_dir: Path, shard_size: int) -> None:
        self.stage_dir = stage_dir
        self.shard_size = shard_size
        self.seq = 0
        self.count = 0  # records in the current (open) shard
        self._handle = None

    @property
    def path(self) -> Path:
        """Path of the current shard."""
        return self.stage_dir / f"step-{self.seq:04d}.jsonl"

    def scan(self) -> None:
        """Position the writer to continue an interrupted stage.

        Continue the last shard if it is partial; if it is already full, the
        next ``write`` starts a new one. Called after ``done_ids`` is computed.
        """
        existing = _shards(self.stage_dir)
        if not existing:
            return
        last = existing[-1]
        self.seq = int(last.stem.split("-")[1])
        with last.open("rb") as handle:
            self.count = sum(1 for line in handle if line.strip())
        if self.count >= self.shard_size:
            self.seq += 1
            self.count = 0

    def write(self, row: bytes) -> Optional[Path]:
        """Append one serialized row; return the shard path if it just filled."""
        if self._handle is None:
            self._handle = self.path.open("ab")
        self._handle.write(row)
        self._handle.flush()
        self.count += 1
        if self.count >= self.shard_size:
            self._handle.close()
            self._handle = None
            full = self.path
            self.seq += 1
            self.count = 0
            return full
        return None

    def close(self) -> Optional[Path]:
        """Close the open shard; return its path if it holds unpublished records."""
        if self._handle is None:
            return None
        self._handle.close()
        self._handle = None
        return self.path if self.count > 0 else None


# --------------------------------------------------------------------------- #
# Progress display
# --------------------------------------------------------------------------- #
def _modifier_label(modifier: Any) -> str:
    """Build a bar label for ``modifier``, disambiguating duplicate names.

    Modifiers default ``.name`` to their class name, so a stage with two
    GoogleTranslate / two LanguageTool modifiers would otherwise show four
    identically-named bars. Field modifiers expose ``source``/``target``;
    fold those into the label so each bar is distinct and readable.
    """
    label = getattr(modifier, "name", type(modifier).__name__)
    source = getattr(modifier, "source", None)
    target = getattr(modifier, "target", None)
    if source:
        span = f"{source}→{target}" if target and target != source else str(source)
        label = f"{label} [{span}]"
    return label


def _make_stage_progress(stage: Stage, total: int) -> Tuple[RichProgress, TaskID, List[TaskID]]:
    """Build a live display: one main bar for the stage plus one per modifier.

    Renders to ``stderr`` (matching loguru) and returns the ``Progress`` along
    with the main task id and the per-modifier task ids, indexed identically to
    ``stage.modifiers`` so a callback can map ``index -> TaskID``.
    """
    progress = RichProgress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=Console(stderr=True),
    )
    main_task = progress.add_task(f"stage {stage.name}", total=total)
    modifier_tasks = [
        progress.add_task(f"  └ {_modifier_label(m)}", total=total) for m in stage.modifiers
    ]
    return progress, main_task, modifier_tasks


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
class PipelineRunner:
    """Executes :class:`StageSpec` objects over a JSONL dataset."""

    def __init__(
        self,
        specs: List[StageSpec],
        *,
        name: str,
        dataset_cfg: Any,
        runner_cfg: Any,
        notifications_cfg: Any = None,
        reporter: Optional[Reporter] = None,
    ) -> None:
        """Capture the stages plus dataset/runner/notification configuration."""
        self.specs = specs
        self.name = name
        # The first stage's input: a single path, a glob, or a list of either,
        # resolved to a file set at run time. Later stages read shard directories.
        self.input_spec = dataset_cfg.input
        self.output_dir = Path(dataset_cfg.output_dir)
        self.concurrency = int(_get(runner_cfg, "concurrency", 8))
        self.resume = bool(_get(runner_cfg, "resume", True))
        self.debug = bool(_get(runner_cfg, "debug", False))
        self.log_level = "DEBUG" if self.debug else "INFO"
        self.id_field = str(_get(runner_cfg, "id_field", "id"))
        # Records per persisted output shard. A shard is uploaded as a
        # downloadable artifact (when reporting is on) the moment it fills.
        self.shard_size = max(1, int(_get(runner_cfg, "shard_size", 1000)))
        # Optional cap: process only the first N input rows (for quick smoke
        # runs over a slice of the dataset). ``None`` means "all rows". Owned by
        # the experiment's `dataset:` block since it slices that dataset.
        limit = _get(dataset_cfg, "limit", None)
        self.limit: Optional[int] = int(limit) if limit is not None else None
        self.pause_between_stages = bool(_get(runner_cfg, "pause_between_stages", False))
        self.start_from = _get(runner_cfg, "start_from", None)
        # Live progress bars: on by config (default true), but only when attached
        # to a TTY so piped/unattended runs keep plain loguru output.
        self.progress = bool(_get(runner_cfg, "progress", True)) and sys.stderr.isatty()
        # Exception types that flag the offending record instead of aborting the
        # run. Resolved once from dotted paths; an empty tuple restores strict
        # abort-on-any-error behaviour (``except ()`` catches nothing).
        self.flag_on_errors: Tuple[type, ...] = _resolve_exception_types(
            _get(runner_cfg, "flag_on_errors", [])
        )
        self.notifications_cfg = notifications_cfg
        # Best-effort dashboard reporting; NullReporter when disabled/unconfigured.
        self.reporter: Reporter = reporter or NullReporter()
        # Live counters the heartbeat loop reads while a stage runs.
        self._processed = 0
        self._stage_total = 0
        # Records flagged (not aborted) in the current stage; reset per stage.
        self._errored = 0
        # Bounds how many closed shards upload concurrently so a burst of
        # rotations never saturates the link or starves the workers.
        self._upload_sem = asyncio.Semaphore(4)
        # Per-stage state for the reported `detail.stages`, keyed by stage id.
        self._stage_state: dict[str, dict] = {}

    def _stage_dir(self, index: int, spec: StageSpec) -> Path:
        """Directory holding the shard output for the stage at position ``index``."""
        return self.output_dir / f"{index:02d}_{spec.id}"

    async def run(self) -> Path:
        """Run every enabled stage in order; return the final output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # The first stage reads the resolved dataset files; each later stage
        # reads the previous stage's shard directory.
        current_input: Union[Path, List[Path]] = _resolve_inputs(self.input_spec)

        start_index = self._resolve_start_index()
        logger.info(
            f"pipeline {self.name!r}: {sum(s.enabled for s in self.specs)} enabled stage(s), "
            f"input={[str(p) for p in current_input]}"
        )

        # Seed reported stage states: disabled => skipped, before start_index =>
        # already succeeded, everything else queued. Then announce the job.
        self._stage_state = {
            spec.id: {
                "state": (
                    "skipped" if not spec.enabled
                    else "succeeded" if index < start_index
                    else "queued"
                )
            }
            for index, spec in enumerate(self.specs)
        }
        if self.reporter.enabled:
            self.reporter.start(self._build_detail(min(start_index, len(self.specs) - 1)))

        try:
            for index, spec in enumerate(self.specs):
                stage_dir = self._stage_dir(index, spec)
                if index < start_index:
                    # Reconstruct the chain so `start_from` reads the right input.
                    if stage_dir.is_dir() and _shards(stage_dir):
                        current_input = stage_dir
                    continue
                if not spec.enabled:
                    logger.info(f"stage {spec.id!r} disabled — passing input through unchanged")
                    continue
                pause = self.pause_between_stages if spec.pause_before is None else spec.pause_before
                if pause:
                    self.reporter.update(state=JobState.PAUSED, force=True)
                    await self._pause_before_stage(spec)

                await self._run_stage(index, spec, current_input, stage_dir)
                current_input = stage_dir
        except Exception:  # noqa: BLE001 - mark the job failed, then let it propagate.
            self.reporter.update(state=JobState.FAILED, force=True)
            raise

        # Normally the last stage's directory; falls back to the output dir when
        # no stage ran (e.g. every stage disabled), so callers always get a Path.
        final_output = current_input if isinstance(current_input, Path) else self.output_dir
        total_errored = self._total_errored()
        if total_errored:
            logger.info(f"pipeline {self.name!r}: {total_errored} record(s) flagged across all stages")
        self._notify_pipeline_end(final_output, total_errored)
        self.reporter.finish(JobState.SUCCEEDED)
        return final_output

    def _resolve_start_index(self) -> int:
        """Translate ``start_from`` into an index, validating the prior output."""
        if not self.start_from:
            return 0
        ids = [spec.id for spec in self.specs]
        if self.start_from not in ids:
            raise KeyError(f"start_from={self.start_from!r} is not a known stage ({ids})")
        index = ids.index(self.start_from)
        if index > 0:
            prior = self._stage_dir(index - 1, self.specs[index - 1])
            if not (prior.is_dir() and _shards(prior)):
                raise FileNotFoundError(
                    f"cannot start_from {self.start_from!r}: prior output {prior} is missing"
                )
        return index

    async def _run_stage(
        self,
        index: int,
        spec: StageSpec,
        input_src: Union[Path, List[Path]],
        stage_dir: Path,
    ) -> None:
        """Process one stage over its input, persisting results as shards."""
        records = _read_records(input_src)
        for i, record in enumerate(records):
            record.setdefault(self.id_field, i)
        # Strip configured bookkeeping fields and apply input filters before the
        # limit, so `limit` caps the *selected* rows (e.g. the first N failures
        # of a rerun), not the raw input.
        records = self._select_input(spec, records)
        if self.limit is not None and len(records) > self.limit:
            logger.info(f"stage {spec.id!r}: limiting input to first {self.limit} of {len(records)} rows")
            records = records[: self.limit]

        meta_path = stage_dir / "meta.json"
        fresh = not (
            self.resume
            and stage_dir.is_dir()
            and _read_signature(meta_path) == spec.signature
        )
        done_ids: set = set()
        if fresh:
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_dir.mkdir(parents=True, exist_ok=True)
        else:
            done_ids = {r.get(self.id_field) for r in _read_records(stage_dir)}

        # Record the signature up front, not at stage end, so an interrupted run
        # leaves behind a matching meta: on rerun the signature still matches and
        # the completed shards are resumed instead of recomputed from scratch. A
        # config/logic change flips the signature, making the stage `fresh` above.
        meta_path.write_bytes(orjson.dumps({"signature": spec.signature, "stage": spec.id}))

        todo = [r for r in records if r.get(self.id_field) not in done_ids]
        logger.info(
            f"stage {spec.id!r} [{index:02d}]: {len(todo)} to process, "
            f"{len(done_ids)} already done, fresh={fresh}"
        )
        self._notify_stage(spec, "on_stage_start", f"{len(todo)} records to process")

        # Report stage start and prime the live counters the heartbeat loop reads.
        self._processed = len(done_ids)
        self._stage_total = len(todo) + len(done_ids)
        self._errored = 0
        heartbeat_task: Optional[asyncio.Task] = None
        if self.reporter.enabled:
            self._stage_state[spec.id] = {
                "state": "running",
                "done": self._processed,
                "total": self._stage_total,
                "errored": 0,
            }
            self.reporter.update(
                progress=Progress(self._processed, self._stage_total, "records"),
                detail=self._build_detail(index),
                state=JobState.RUNNING,
                force=True,
            )
            heartbeat_task = asyncio.create_task(self._heartbeat_loop(index))

        semaphore = asyncio.Semaphore(self.concurrency)
        write_lock = asyncio.Lock()
        # Continue the last partial shard on resume, else start a fresh one.
        writer = _ShardWriter(stage_dir, self.shard_size)
        writer.scan()

        use_progress = self.progress and bool(todo)
        progress: Optional[Progress] = None
        main_task: Optional[TaskID] = None
        log_sink: Optional[int] = None
        tasks: List[asyncio.Task] = []
        if use_progress:
            progress, main_task, modifier_tasks = _make_stage_progress(spec.stage, len(todo))
            spec.stage.set_progress_callback(
                lambda i, _phase: progress.advance(modifier_tasks[i])
            )

        try:
            if use_progress:
                assert progress is not None
                progress.start()
                # Route logs through the live console so they print above the
                # bars instead of corrupting them. The default stderr sink holds
                # the original stderr (bypassing rich's redirect), so swap it out
                # for the duration and restore it in ``finally``.
                logger.remove()
                log_sink = logger.add(
                    lambda m: progress.console.print(m.rstrip("\n"), markup=False, highlight=False),
                    colorize=False,
                    level=self.log_level,
                )

            async def worker(record: dict) -> None:
                """Run the stage on one record (or pass/flag it) and append to a shard."""
                async with semaphore:
                    if record.get("error"):  # flagged upstream → pass through untouched
                        out = dict(record)
                    else:
                        try:
                            context = await spec.stage.run(dict(record))
                            out = context.to_dict()
                        except self.flag_on_errors as exc:  # configured per-record failures
                            # Flag the modifier-enriched context the stage stashed
                            # on the exception (so fields like a Google translation
                            # survive), falling back to the raw record if absent.
                            enriched = errored_context(exc)
                            base = enriched.to_dict() if enriched is not None else record
                            out = self._flag_record(base, spec.id, exc)
                    row = orjson.dumps(out, option=orjson.OPT_APPEND_NEWLINE)
                    async with write_lock:
                        # Exactly one worker — the one whose write fills a shard —
                        # gets its path back, so each closed shard is uploaded once.
                        full_shard = writer.write(row)
                        self._processed += 1
                    if use_progress:
                        progress.advance(main_task)
                    if full_shard is not None:
                        await self._upload_shard(spec, stage_dir, full_shard)
                    if self.debug:
                        logger.debug(f"stage {spec.id!r} done id={record.get(self.id_field)}")

            tasks = [asyncio.create_task(worker(record)) for record in todo]
            await asyncio.gather(*tasks)
        except Exception as exc:  # noqa: BLE001 — surface, notify, re-raise
            # gather() surfaces the first failure but leaves siblings running;
            # cancel and drain them before closing the writer, so no in-flight
            # worker writes to a shard we are about to close and upload.
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            partial = writer.close()
            logger.opt(exception=exc).error(f"stage {spec.id!r} failed")
            self._notify_stage(spec, "on_error", f"stage failed: {exc}", priority="urgent")
            if self.reporter.enabled:
                self._stage_state[spec.id] = {
                    "state": "failed",
                    "done": self._processed,
                    "total": self._stage_total,
                    "errored": self._errored,
                }
                # Earlier shards already uploaded as complete; publish the final
                # partial shard (incomplete) so its finished records aren't lost.
                if partial is not None:
                    await self._upload_shard(spec, stage_dir, partial, complete=False)
                self.reporter.update(detail=self._build_detail(index), force=True)
            raise
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
            spec.stage.set_progress_callback(None)
            if use_progress:
                if log_sink is not None:
                    logger.remove(log_sink)
                logger.add(sys.stderr, level=self.log_level)  # restore the default loguru sink
                assert progress is not None
                progress.stop()

        # Success: close and publish the final shard. The signature was already
        # written at stage start (so crashes resume), so there is nothing to
        # record here beyond the final shard.
        final_shard = writer.close()
        flagged_note = f", {self._errored} flagged" if self._errored else ""
        logger.info(f"stage {spec.id!r} [{index:02d}]: {len(todo)} records processed{flagged_note}")
        self._notify_stage(spec, "on_stage_end", f"{len(todo)} records processed{flagged_note}")
        if final_shard is not None:
            await self._upload_shard(spec, stage_dir, final_shard)

        # On resume, shards completed by an earlier process were not re-uploaded
        # above (the worker loop only publishes shards it fills this run), so they
        # are missing from this process's in-memory manifest. Re-list every shard
        # so manifest.json reflects the stage's full output, not just this run's.
        if not fresh:
            await self._reconcile_shards(spec, stage_dir)

        # Mark the stage done and push a final status for it.
        if self.reporter.enabled:
            self._stage_state[spec.id] = {
                "state": "succeeded",
                "done": self._stage_total,
                "total": self._stage_total,
                "errored": self._errored,
            }
            self.reporter.update(
                progress=Progress(self._stage_total, self._stage_total, "records"),
                detail=self._build_detail(index),
                force=True,
            )

    def _select_input(self, spec: StageSpec, records: List[dict]) -> List[dict]:
        """Keep only records passing every filter, then drop configured fields.

        ``spec.filters`` runs first and selects which records run; all filters
        must keep a record (AND), and each is evaluated against the record's
        fields as-is — including any ``error*`` flags a previous run wrote, so a
        rerun can select on ``error == True``. ``spec.drop_fields`` then strips
        the named keys from the survivors — notably those ``error*`` flags, so a
        previously-flagged record is reprocessed instead of hitting the
        upstream-error pass-through in ``worker``. Returns the surviving records
        (mutated in place).
        """
        if spec.filters:
            kept = [r for r in records if all(f.keep(r) for f in spec.filters)]
            logger.info(
                f"stage {spec.id!r}: filters kept {len(kept)} of {len(records)} input record(s)"
            )
            records = kept
        for record in records:
            for name in spec.drop_fields:
                record.pop(name, None)
        return records

    def _flag_record(self, record: dict, stage_id: str, exc: BaseException) -> dict:
        """Return a copy of ``record`` marked errored instead of aborting the run."""
        flagged = dict(record)
        flagged["error"] = True
        flagged["error_message"] = str(exc)
        flagged["error_type"] = f"{type(exc).__module__}.{type(exc).__name__}"
        flagged["error_stage"] = stage_id
        self._errored += 1
        # Surface the running count in the reported stage state so the dashboard
        # sees it on the next push (every status push reads ``_stage_state``).
        if stage_id in self._stage_state:
            self._stage_state[stage_id]["errored"] = self._errored
        logger.warning(
            f"stage {stage_id!r}: record id={record.get(self.id_field)} flagged "
            f"({flagged['error_type']}): {exc}"
        )
        return flagged

    async def _upload_shard(
        self,
        spec: StageSpec,
        stage_dir: Path,
        shard_path: Path,
        *,
        complete: bool = True,
    ) -> None:
        """Publish one closed shard to the dashboard as a downloadable artifact.

        A closed shard is static, so this is a plain off-thread ``upload_file``
        — no temp copy, no read/write race. It is a no-op when reporting (or
        artifact upload) is off, keeping the call site unconditional. The upload
        runs in a thread, bounded by ``_upload_sem``, so a burst of rotations
        never stalls the workers or saturates the link. ``dest`` groups shards by
        stage (``artifacts/<NN>_<stage>/step-NNNN.jsonl``).
        """
        if not (self.reporter.enabled and self.reporter.upload_artifacts):
            return
        dest = f"artifacts/{stage_dir.name}/{shard_path.name}"
        label = f"{spec.id} {shard_path.name}"
        async with self._upload_sem:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self.reporter.upload_artifact(
                    shard_path, label=label, complete=complete, dest=dest
                ),
            )

    async def _reconcile_shards(self, spec: StageSpec, stage_dir: Path) -> None:
        """Re-list a resumed stage's shards in the manifest, uploading any S3 lacks.

        Mirrors :meth:`_upload_shard`'s keying but routes through the reporter's
        HEAD-gated :meth:`~DataCurator.reporting.reporter.JobReporter.ensure_artifact`,
        so shards already on S3 (from the process that first wrote them) are
        re-registered without re-sending their bytes. At successful stage end
        every shard is closed and complete, so each is published ``complete=True``
        — which also repairs a stale incomplete flag left by an earlier crash.
        """
        if not (self.reporter.enabled and self.reporter.upload_artifacts):
            return
        for shard in _shards(stage_dir):
            dest = f"artifacts/{stage_dir.name}/{shard.name}"
            label = f"{spec.id} {shard.name}"
            async with self._upload_sem:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda s=shard, d=dest, l=label: self.reporter.ensure_artifact(
                        s, label=l, complete=True, dest=d
                    ),
                )

    async def _pause_before_stage(self, spec: StageSpec) -> None:
        """Block for operator confirmation before running ``spec``.

        Lets any manual step happen between stages — swapping the deployed
        model, inspecting intermediate output, reconfiguring hardware. Falls
        back to continuing when no TTY is attached so unattended runs proceed.
        """
        message = f"About to run stage {spec.id!r}. Complete any manual step before continuing."
        logger.warning(message)
        self._notify_raw("Stage confirmation required", message, priority="urgent")
        if not sys.stdin.isatty():
            logger.warning("a stage pause is requested but no TTY is attached; continuing")
            return
        await asyncio.get_running_loop().run_in_executor(
            None, input, f"{message}\nPress Enter to continue..."
        )

    # -- dashboard reporting ------------------------------------------------
    def _build_detail(self, current_index: int) -> dict:
        """Assemble the ``data-pipeline`` ``detail`` payload for the reporter.

        Maps every stage to its current state (``queued``/``running``/
        ``succeeded``/``failed``/``skipped``) so the dashboard can render the
        stage list, alongside which stage is active. Each stage also carries the
        number of records it flagged (``errored``), and ``total_errored`` sums
        those across the pipeline.
        """
        stages = [
            StageInfo(
                id=spec.id,
                state=self._stage_state.get(spec.id, {}).get("state", "queued"),
                done=self._stage_state.get(spec.id, {}).get("done"),
                total=self._stage_state.get(spec.id, {}).get("total"),
                errored=self._stage_state.get(spec.id, {}).get("errored"),
            ).to_dict()
            for spec in self.specs
        ]
        return {
            "current_stage": self.specs[current_index].id,
            "stage_index": current_index,
            "total_stages": len(self.specs),
            "stages": stages,
            "total_errored": self._total_errored(),
        }

    async def _heartbeat_loop(self, index: int) -> None:
        """Refresh liveness, progress and metrics every heartbeat interval.

        Runs for the duration of one stage so the dashboard keeps moving (and
        a missed heartbeat reliably flags a dead run) even when a single record
        takes longer than the interval to process.
        """
        try:
            while True:
                await asyncio.sleep(self.reporter.heartbeat_interval_s)
                self._stage_state[self.specs[index].id].update(
                    done=self._processed, total=self._stage_total
                )
                self.reporter.update(
                    progress=Progress(self._processed, self._stage_total, "records"),
                    detail=self._build_detail(index),
                    force=True,
                )
        except asyncio.CancelledError:
            pass

    # -- notifications ------------------------------------------------------
    def _notify_stage(
        self, spec: StageSpec, flag: str, message: str, priority: Optional[str] = None
    ) -> None:
        """Send a stage notification if ``flag`` is enabled in the config."""
        if self.notifications_cfg is None or not _get(self.notifications_cfg, flag, False):
            return
        self._notify_raw(f"[{self.name}] {spec.id}", message, priority=priority)

    def _total_errored(self) -> int:
        """Sum records flagged across every stage (disjoint, so a true total)."""
        return sum(state.get("errored") or 0 for state in self._stage_state.values())

    def _notify_pipeline_end(self, final_output: Path, total_errored: int = 0) -> None:
        """Send the pipeline-completion notification if enabled."""
        if self.notifications_cfg is None or not _get(self.notifications_cfg, "on_pipeline_end", False):
            return
        flagged_note = f" ({total_errored} flagged)" if total_errored else ""
        self._notify_raw(f"[{self.name}] finished", f"final output: {final_output}{flagged_note}")

    def _notify_raw(self, title: str, message: str, priority: Optional[str] = None) -> None:
        """Forward to the ntfy notifier; never let a failure abort the run."""
        notify(self.notifications_cfg, title, message, priority=priority)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a DictConfig/dict, returning ``default`` when absent."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def _resolve_exception_types(paths: Any) -> Tuple[type, ...]:
    """Import dotted exception paths into a tuple of exception classes."""
    resolved: List[type] = []
    for path in paths or []:
        cls = hydra.utils.get_class(str(path))
        if not (isinstance(cls, type) and issubclass(cls, BaseException)):
            raise TypeError(f"flag_on_errors entry {path!r} is not an exception type")
        resolved.append(cls)
    return tuple(resolved)
