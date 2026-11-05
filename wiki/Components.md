# Components

This page documents the modules that currently live in the repo and
how they fit together. The stage/modifier framework, the Hydra-driven
runner, a generic LLM stage and the Google Translate / LanguageTool
modifiers are all in place; further task-specific stages will plug into
the same framework as they are ported in. Each section maps directly to
a file or directory and only describes behavior that is visible in the
source. For the config keys each component reads, see
[Configuration](Configuration).

## Stages and modifiers

`DataCurator.pipeline` defines the generic building blocks for a stage.
A `Stage` is one self-contained processing step — running an LLM prompt,
applying regex fixes, calling a translation API. Subclasses implement
`process`, which reads inputs from and writes outputs back into a shared
`StageContext`; callers invoke `run`, which fires every `before` modifier,
calls `process`, then fires every `after` modifier and returns the
context. The API is async so IO-bound stages (LLM, translation) can await
their work; purely synchronous stages simply do not await.

A `StageModifier` mutates the variables a stage operates on without the
stage knowing about it — the motivating case is normalising a field
before the LLM sees it. Each modifier declares its `phases` (`BEFORE` to
adjust inputs, `AFTER` to adjust outputs) and implements `modify`.
`FieldModifier` is a ready-made subclass for the common "rewrite one
field" case: implement `transform`, and it reads `source`, transforms the
value, and writes it back to `source` or a separate `target`. This is the
pre-stage enhancer hook described in the pipeline design notes — an
attached filter that replaces a data field's value or adds new ones.

`StageContext` is the mutable mapping passed through both sides. Beyond
the usual mapping operations it exposes intent-revealing mutators:
`add` (new variable, errors if it exists), `replace` (overwrite existing,
errors if absent), and `set` (add-or-overwrite).

## Pipeline runner

`DataCurator.pipeline.runner` turns a Hydra config into an ordered run of
stages over a JSONL dataset. `build_pipeline(cfg)` reads the `order` list
and the `pipeline` mapping, instantiating each stage's `stage` block via
`hydra.utils.instantiate` (modifiers nested under it are built too);
`PipelineRunner.run` then feeds the dataset through them. The first
stage's input may be a single file, a glob, or a list of either; stage `N`
reads the output of stage `N-1`. Output is **sharded**: each stage writes
into a directory `<output_dir>/<NN>_<stage>/` as fixed-size shards
`step-NNNN.jsonl` (`runner.shard_size` records each). Records are processed
concurrently up to `runner.concurrency` and each result is flushed as soon
as it is produced, so an interrupted run never loses completed work. When a
shard fills it is closed (immutable) and — if reporting is on — uploaded as
a downloadable artifact the instant it closes; the next stage reads every
shard in order.

The runner implements the pipeline-design rules directly. Resume is
driven by a sidecar `<NN>_<stage>/meta.json` holding a signature hashed
from the stage's resolved config, with prompt files inlined so the signature
covers the prompt text rather than its path (a `prompts.json` beside it keeps
each message's role, source file and text): when the signature is unchanged and
`runner.resume` is set, records already present across the shards (matched
by `runner.id_field`) are skipped and the last partial shard is continued;
when the config or logic changes the signature differs and the stage is
recomputed from scratch. Stages with `enabled: false` are passed through,
and `runner.start_from` resumes at a named stage by reading the previous
stage's shard directory. When
`runner.pause_between_stages` is set the runner logs a warning, sends an
urgent notification, and — unless no TTY is attached — blocks for operator
confirmation before each stage, so any manual step (swapping the deployed
GPU model, inspecting intermediate output, reconfiguring hardware) can
happen in between. Individual stages override this with a `pause_before`
flag in their `pipeline.<id>` entry, so you can leave the global default
off and pause before just one stage (or invert it). A record that fails with
an exception type listed in `runner.flag_on_errors` is written with
`error: true` (plus `error_message`/`error_type`/`error_stage`) and the run
continues; later stages pass any `error: true` record through unchanged. The
flagged count is reported per stage (`detail.stages[].errored`) and job-wide
(`detail.total_errored`). Those flagged records are reprocessed by running a
separate experiment that filters the original output down to the failures —
see [Record filters](#record-filters).

Each stage can also carry `filters` and `drop_fields` (runner metadata
alongside `enabled`/`pause_before`, instantiated by `build_pipeline`). Before
resume and `limit`, the runner keeps only the input records passing every
filter, then strips `drop_fields` from the survivors — see
[Record filters](#record-filters).
Notifications fire through
[`notify`](#notifications) at the points enabled in the notifications
config.

When `runner.progress` is set and stderr is a TTY, the runner renders a live
display (a bar for the stage plus one per modifier) and routes log output
through it so the bars are not corrupted; on a non-TTY (piped or unattended)
run it falls back to plain loguru output automatically.

The Hydra entrypoint is `run_pipeline.py` at the repo root:

```bash
uv run python run_pipeline.py
```

## LLM stage

`DataCurator.pipeline.stages.llm.LLMStage` runs one structured-output chat
completion per record against any OpenAI-compatible endpoint (vLLM is the
target). It is built from a model config (see [Model config](Configuration#model-config-configmodel))
and a `response_model` — a dotted path to a Pydantic class, e.g.
`DataCurator.pipeline.schemas.TranslationReview` — and parses each reply
into that model via the OpenAI SDK's `chat.completions.parse`, storing the
result as a dict under `output_field`. The model config's
[`structured_output`](Configuration#model-config-configmodel) picks how the
schema is delivered: `json` (the default) sends it as `response_format`, `tool`
sends the identical strict schema as a forced function call for a backend that
ignores `response_format` (Muse-Glimmer applies no grammar until a tool channel
opens). Only the delivery differs — a reply that satisfies neither is an
`UnparseableOutputError` either way.

The conversation comes from one of two places. With `prompt` set, the
stage renders a list of messages against the record's fields, each either
`{role, template}` — a path to a Jinja2 file — or `{role, text}` with the
template inline; this is the path that makes modifiers useful, since a
`before`-phase modifier can fix or add a field that the template then
renders. Prompt files are experiment assets kept outside git (see
[Configuration](Configuration#prompt-sources)). Templates are compiled when
the pipeline is built, so a missing file fails fast. With `prompt: []` the
stage reads a pre-built conversation from `input_field` instead. Native OpenAI
sampling fields are sent
top-level while vLLM-only controls (`top_k`, `min_p`, `repetition_penalty`)
and the Qwen3 `enable_thinking` toggle are routed into `extra_body`,
matching the model config. In-flight requests are capped by an internal
semaphore sized to `model.concurrency`, each request honours `model.timeout`
(seconds), and each call is retried with exponential backoff via tenacity. A
reply the SDK cannot parse into the schema (`parsed is None`, e.g. a refusal)
raises `UnparseableOutputError` so it is resampled — and, if listed in
[`runner.flag_on_errors`](Configuration#flagging-unprocessable-records),
flags the record rather than aborting the run.

Optional `validators` run **inside** the retry loop, just after a reply
parses, so a guard rejecting an otherwise-valid parse triggers a fresh sample;
see [Response validators](#response-validators). When a guard still fails after
every retry and `keep_rejected_output` is set, the parsed output that failed is
stashed under `rejected_output_field` so the generated text survives on the
flagged record.

`reprompt_attempts` (default 0) turns blind resampling into **correction
rounds**: the rejected answer is replayed as an assistant turn plus a user turn
carrying the validator's `feedback` (what character is missing, which
LanguageTool rule fired, …), and the model is re-asked in the same
conversation. Connection/parse failures and feedback-less guard failures (an
unreachable checker is not the model's fault) never see a correction turn —
they keep the blind `model.retries` budget, so the worst case is
`model.retries × (reprompt_attempts + 1)` requests. While reprompting is
enabled, every record carries `<output_field>_reprompt_rounds` — the number of
correction rounds the output cost, `0` when it was accepted on the first try (the
field is absent only when `reprompt_attempts` is 0 for the stage). Flagged
records carry it too: a record that burned the whole budget and still failed
reports the rounds it spent.

`reprompt_on_truncation` (default `false`, needs `reprompt_attempts > 0`) gives
an answer cut off at `max_tokens` (`openai.LengthFinishReasonError`) a
correction round from the same budget instead of a blind resample. The clipped
partial answer is replayed, and the correction names the repeated phrase when
the answer looped. The batched stage applies it to the whole group.

`on_guard_failure` decides what a guard failure does once every round and retry
is spent. `flag` (default) raises, so the record is flagged. `keep` stores the
rejected output under `output_field` and the error text under
`guard_failure_field` (default `<output_field>_guard_failure`), and the stage
continues into its `after` modifiers, which can route the record. Outages,
parse and length errors, and batch misalignments still raise.
`gate_squadv2_qwen38_batching_rerun.yaml` routes a kept gate answer to fix with
a `SetWhenModifier` on `gate_answer_guard_failure`.

Optional `repairs` (`DataCurator.pipeline.repairs`, each an `OutputRepair`
with a `_target_`) rewrite the parsed output **before** the validators see it,
for defects a rule can undo. `LanguageToolRepair` applies LanguageTool's
suggestion for an allow-list of `rules` only (default `WORD_REPEAT_RULE`, which
drops a doubled word). An unreachable server raises `ServiceUnavailableError`.

## Stage modifiers

The modifiers in `DataCurator.pipeline.modifiers` are all attachable to
any stage as a pre- or post-pass (see [Stages and modifiers](#stages-and-modifiers)).
Two wrap the rule-based components as `FieldModifier`s (one `source` field
in, one `target` field out); the others are structural helpers that
reshape the context.

**Field modifiers** (subclass `FieldModifier`, share `source`/`target`/
`phase`/`required`):

- `google_translate.GoogleTranslateModifier` translates `source` with
  [GoogleTranslator](#googletranslator) and writes the result to `target`
  (added if new; defaults to `source` for an in-place rewrite), leaving the
  original value in place if a translation fails. Its `source_language_code`/
  `target_language_code` and the `retries`/`timeout`/`concurrency` HTTP knobs
  configure the underlying translator.
- `languagetool.LanguageToolModifier` runs [LanguageTool](#languagetool-service)
  over `source`, writing the auto-fixed text back (to `target`, or in place)
  and optionally stashing the remaining unresolved issues in `issues_field`
  for the prompt to show the model. `allowed_fixes` restricts which rule
  categories are auto-applied and `max_passes` caps the re-check loop. The
  synchronous checker runs in a worker thread and is created lazily, so
  building a pipeline does not require the LanguageTool server to be up.

**Structural modifiers** (subclass `StageModifier` directly):

- `rename.RenameModifier` renames or copies several fields at once per a
  `fields` mapping of `input -> output`. With `keep=False` (default) each
  input is dropped after the copy — a true rename — while `keep=True` leaves
  it in place. Inputs are read from an up-front snapshot, so a chained
  mapping like `{a: b, b: c}` never cascades, and a field that is itself
  another rename's output is never deleted. A missing input is skipped unless
  `required=True`. The handy first `before`-pass for aligning a raw dataset's
  column names with what the prompt and downstream modifiers expect.
- `constants.ConstantsModifier` injects fixed `name -> value` literals from
  config (`values`) — fields that are the same for every record and so do not
  live in the dataset, such as the source/target language *names* a prompt
  needs (`"English"`, `"Polish"`). Unlike `RenameModifier` it reads nothing,
  so it is the way to introduce a brand-new field with no source column. By
  default a constant that would clobber an existing field raises; set
  `overwrite=True` (or `exists_ok=True`) to replace instead.
- `candidates.CandidatesModifier` folds several context values into a single
  list of `{id, text}` objects under one `target` field — the shape a prompt
  iterates over to present N labelled candidates (`[Q1] … [Q2] … [Q3] …`).
  Each entry in `sources` is a field name or a dotted path into nested dicts
  (`review_question.improved_primary`), and ids are generated as
  `f"{id_prefix}{n}"` from 1, so `id_prefix: Q` yields `Q1`, `Q2`, `Q3`.
  Neither `RenameModifier` (which moves one whole field) nor
  `ConstantsModifier` (which writes literals) can build a list out of nested
  sibling fields, so this is what lets a final stage assemble the rewrites an
  earlier review stage produced into candidate lists. A missing source raises
  unless `required=False`; `exists_ok` guards an existing `target`.
- `transfer.TransferCheckModifier` runs one deterministic transfer check from
  `DataCurator.pipeline.transfer` over a `(source_field, hypothesis_field)`
  pair and writes the findings mapping to `target` — signal acquisition only,
  no decisions. `check: script` reports guarded-script leakage (Cyrillic/CJK
  characters in the hypothesis with no counterpart in the source); `check:
  patterns` reports dropped numbers/URLs/emails as `{kind: count}`. `{}`
  means checked-and-clean; an absent column means the pair was never checked,
  so a downstream truthiness read fails open. Pair it with a
  `conditional.SetWhenModifier` to act on the findings — the gate override
  that routes a defective Google-Translate draft (which cannot be resampled)
  through the fix path.
- `edits.ApplyEditsModifier` applies an LLM's span edits
  (`[{segment_id, span, replacement}]`, read from `edits`) to a
  `[{id, translation}]` list read from `segments`. Each edit replaces every
  exact occurrence of its span. The modifier writes one
  `{id, source_id, text, edited}` entry per segment to `target`, including
  when there are no edits. An edit that names an unknown segment, or whose span
  no longer occurs, raises, so the record is flagged instead of losing part of
  its adjustment. It is the actuation half of `EditSpanValidator` (see
  [Pair adjustment](#pair-adjustment)).

## Record filters

`DataCurator.pipeline.filters` selects **which input records a stage runs**.
A `RecordFilter` implements `keep(record) -> bool`; filters are declared per
stage as a `filters` list (each with a `_target_`, like modifiers) and are
runner metadata, not part of the `Stage` — deciding which records run is the
runner's job. `build_pipeline` instantiates them, and before resume/`limit`
the runner keeps only records passing **every** filter, then strips the
stage's `drop_fields` from the survivors.

The in-tree filter is `ExpressionFilter`, which keeps a record when its
`expression` is truthy over the record's fields. Expressions use familiar
syntax — bare names are field lookups, `a.b`/`a[b]` index nested mappings, and
`and`/`or`/`not`, comparisons (`== != < <= > >= in not in`) and the helpers
`len`/`abs`/`str`/`int`/`float`/`bool` are allowed — but are evaluated by a
small allow-listed walk over Python's `ast`, never `eval`: no builtins, no
attribute access (so nothing can reach a Python object). It is parsed and
validated when the pipeline is built, so a malformed or disallowed expression
(`__import__(...)`, `open(...)`, a syntax error) fails fast. A field absent
from the record reads as a falsy sentinel unequal to every value, so
`error == True` simply drops records without an `error` key rather than
raising; `true`/`false`/`null` alias `True`/`False`/`None`.

The motivating use is **rerunning failed records as a separate experiment**:
read the original run's output, keep `error == True`, and `drop_fields` the
`error*` flags so the failures reprocess (otherwise the runner would pass them
through). `config/experiment/queries_fiqa_no_thinking_rerun.yaml` is the worked
example — see [Filtering and rerunning records](Configuration#filtering-and-rerunning-records).

## Response validators

`DataCurator.pipeline.validators` guards **what an LLM stage produced**. A
`ResponseValidator` implements `validate(output, context) -> None` and raises
`ResponseValidationError` when the parsed output is implausible; like filters
and modifiers, validators are config-driven (declared under a stage's
`validators` list, each with a `_target_`, and built by Hydra). The
[LLM stage](#llm-stage) runs them inside its retry loop, so a guard rejecting
an output that parsed cleanly still triggers a fresh sample; once retries are
exhausted the error re-raises and — when `ResponseValidationError` is listed in
[`runner.flag_on_errors`](Configuration#flagging-unprocessable-records) — the
record is flagged rather than aborting the run. They catch a different class of
failure than `response_model`: the JSON parses and satisfies the schema, but
the *content* is wrong.

Content failures attach an LLM-facing `feedback` string (the concrete
diagnostics: per-character counts, leaked scripts, missing tokens, LT rule +
span + suggestions) that a reprompting stage shows the model verbatim; a
missing field carries none and is only ever blind-retried. An unreachable
LT/MetricX server is not a content failure at all — it raises
`ServiceUnavailableError`, which skips both the correction loop and the blind
retries (the model was never at fault, so another generation is pure waste)
and reaches the worker as connection-class. The wording is overridable per validator
with `feedback_template`, a `str.format` template checked at construction
against the validator's placeholders — `LengthRatio` `{field, length,
ref_length, ratio, min_ratio, max_ratio, unit}`, `CharacterCount` `{field,
mismatches}`, `ScriptConsistency` `{field, leaks}`, `PatternPreservation`
`{field, missing, extra}`, `Similarity` `{field, ratio, min_similarity, dice,
min_dice}`, `Repetition` `{field, phrase, count, ref_count, max_excess}`,
`LanguageTool` `{field, findings}`, `MetricX` `{field, score, max_score}`,
`EditSpan` `{problems}`.

The in-tree guards read their reference from the context (so the prompt's
own inputs bound the output) and skip the check when that reference is
missing or empty unless `required=True` — a statement about absent *fields*
only, never about a service being down:

- `LengthRatioValidator` bounds `len(field) / len(reference)` for each output
  field against `[min_ratio, max_ratio]` (either optional), in characters
  (default) or whitespace `word`s. It catches a **truncated rewrite** that
  breaks off mid-sentence yet still parses (`finish_reason: stop`, so the SDK
  never raises `LengthFinishReasonError`): far shorter than the reference it
  mirrors, it trips `min_ratio`.
- `CharacterCountValidator` requires translation-invariant characters
  (`characters`, counted with `str.count` — multi-char substrings like `\n\n`
  work) to keep their reference count within `max_diff` (default 0, exact). It
  catches a model that **escapes its own output** (prefixing words with `\`, JSON
  still valid) or drops/swaps marks like `!`/`?`/brackets that pass through
  EN→PL translation 1:1. `skip_empty_fields` skips an output field whose value
  is empty — the gate's `""` "no fix" sentinel.
- `ScriptConsistencyValidator` rejects an output using a **guarded Unicode
  script** (Cyrillic, Han, Katakana, fullwidth CJK punctuation, … — see
  `GUARDED_SCRIPT_RANGES` in `DataCurator.pipeline.transfer`) that the
  reference does not use — the leakage artifact where a CJK character or
  `，`-style punctuation lands in a Polish rewrite. Latin, digits, emoji and
  unlisted scripts never count; a script the reference itself uses is always
  allowed, and `allow_scripts` whitelists more.
- `PatternPreservationValidator` requires numbers, URLs and emails of the
  reference to survive translation (`kinds` selects which). URLs/emails must
  appear verbatim (a miss always fails); numbers compare as multisets of
  canonical digit strings — grouping/decimal restyling (`1,000.5` → `1000,5`)
  matches, English ordinal/decade tokens (`20th`, `1980s`) are exempt because
  Polish legitimately renumbers them — and fail only past `max_diff`.
  `mode: both` also counts numbers the output invented. Thresholds are
  measured with `scripts/probe_transfer_checks.py`.
- `SimilarityValidator` requires each output field to stay close to a
  same-language reference (`difflib.SequenceMatcher` ratio ≥
  `min_similarity`); its target is a batchmate's answer landing on the wrong
  record. The ratio punishes word reordering, so `min_dice` adds an
  order-insensitive Sørensen–Dice score over word tokens cut to `stem_prefix`
  characters (default 5, tolerating inflection). With it set, a field fails only
  when **both** scores are under their minimum.
- `RepetitionValidator` catches generation loops that still close their JSON.
  For each output field it finds the word `ngram` (default 3) repeated most
  beyond its count in the optional `reference`, and fails past `max_excess`
  (default 5).
- `EditSpanValidator` checks a list of span edits (`edits`, in the output)
  against the record's `segments` instead of a reference text. Each edit's
  `segment_id` must exist, and its `span` must occur in that segment's
  translation and differ from its `replacement`. Its `source_term` must occur,
  casefolded, in the English of the edited segment and of at least
  `min_segments_with_term` segments in all. The span and the replacement are
  each capped at `max_words`, and the record at `max_edits`. It reports every
  broken edit in one error, and an empty edit list always passes.

Both `fields` and `reference` accept dotted paths into nested mappings. The
worked example is the `fix_*` stage of `documents_fiqa_no_thinking.yaml`, which
pairs a `LengthRatioValidator` against `target_text` (the Google-Translate
draft) with a `CharacterCountValidator` against `source_text` (the English
source) — see [Validator configs](Configuration#validator-configs).

## GoogleTranslator

An async client around a Google Translate web endpoint, whose URL and `client`
query value are the `remote_server` and `client_name` bindings. It is intended to run as a
pre-pass that produces a reference translation alongside the original
target, so downstream LLM review can compare forms.

`GoogleTranslator` is constructed with a source/target language pair,
those two bindings and three HTTP knobs. An unset binding raises
`GoogleTranslateUnavailable` on first use, for the modifier too:

- `retries` — tenacity attempt budget for `translate_with_retries`
  (exponential backoff, 2–30 s). A blank answer to a non-blank text raises
  `EmptyTranslationError` inside the loop, so it is retried like any failure
  rather than returned.
- `timeout` — per-request timeout, applied via `httpx.Timeout`.
- `concurrency` — both `max_connections` and `max_keepalive_connections`
  on the underlying `httpx.AsyncClient` built by `_client()`.

`translate(client, text)` issues a single request and normalises the
response, which Google returns in three different shapes (flat string,
list of texts, or list of `[text, source]` pairs).
`translate_with_retries(client, text)` wraps that call in an
exponential-backoff retry loop and returns `None` after the budget is
exhausted, logging the failure via loguru. Blank input short-circuits
and is returned unchanged.

The two callers differ on `None`: `GoogleTranslateModifier` keeps the
original value, while the standalone `GoogleTranslateStage` raises
`GoogleTranslateUnavailable`. That is a `ServiceUnavailableError`, so on the
fleet a chunk Google keeps throttling trips the outage breaker and is nak'd
for redelivery instead of uploading empty or untranslated drafts.

## Notifications

`DataCurator.pipeline.notifications.notify` posts a message to ntfy.sh
using the stdlib `urllib`, so no extra dependency is pulled in. The
function takes a notifications sub-config (an OmegaConf `DictConfig` or
a plain dict), a title, a message body, and optional `priority` and
`tags` overrides.

Behavior is deliberately best-effort:

- If `enabled` is false, `notify()` returns immediately.
- If `backend` is anything other than `ntfy`, the call is skipped with
  a warning — only ntfy is implemented.
- If `topic` is missing, the call is skipped with a warning.
- Any `URLError`, `TimeoutError`, or `OSError` during the POST is
  caught and logged at warning level. A failed notification never
  aborts the caller.

Authentication is opt-in via `auth_token`, forwarded as a `Bearer`
header. Tags are joined into a comma-separated `Tags` header. See
[Configuration](Configuration#notification-configs-confignotifications)
for the full key set.

## Job reporting

`DataCurator.reporting.JobReporter` publishes a job's live status, host
metrics and output artifacts to an S3-compatible bucket (Garage, MinIO,
AWS) so an external dashboard can show progress without ever connecting to
the machine running the job — the producer only ever PUTs to S3. It is
job-agnostic: it knows the *envelope* defined in
`DataCurator.reporting.schema`, not what the pipeline does, so a future
training run can publish through the same client. Objects land under
`s3://<bucket>/<prefix>/<job_id>/` as `status.json` (the current envelope),
`metrics.jsonl` (an optional time series via `log_metric`), `manifest.json`
and the uploaded files under `artifacts/`.

The envelope carries a `schema_version`, identity (`job_id`, `type`,
`name`), a `state` (the `JobState` lifecycle — `queued`/`running`/`paused`/
`succeeded`/`failed`), `progress`, `host`, a `metrics` snapshot, timestamps
with `heartbeat_interval_s`, optional `links`, a monotonic `artifacts_rev`
(bumped on each artifact upload so the dashboard knows to refresh its
artifact list), and a free-form `detail` block the dashboard renders per
`type`. Like
[`notify`](#notifications), every call is best-effort: a failed upload logs
a warning and never aborts the job, and status pushes are throttled to the
heartbeat cadence so calling `update` per record does not hammer S3.
`build_reporter` returns a no-op `NullReporter` whenever reporting is
disabled, uses an unknown backend, or has no bucket configured, so call
sites stay unconditional. Metrics come from `psutil` (CPU/RAM) and NVML
(per-GPU utilisation, memory, temperature, power), each sampled
best-effort.

The [pipeline runner](#pipeline-runner) is the first consumer.
`PipelineRunner` takes a reporter, announces the job at start, marks each
stage `running`/`succeeded`/`failed` in the reported `detail`, runs a
heartbeat loop that refreshes progress and metrics every
`heartbeat_interval_s` for the duration of a stage, and calls `finish` at
the end. Each artifact upload re-publishes `manifest.json`, the index the
dashboard reads to list and serve downloads, and bumps `artifacts_rev`.

Because output is sharded (above), mid-stage downloads need no special
machinery: a shard is uploaded the instant it closes, so it is already a
static, immutable object — a plain `upload_file`, no temp copy and no
read/write race. The runner uploads each closed shard off-thread (bounded
by a small semaphore) under `artifacts/<NN>_<stage>/step-NNNN.jsonl`; the
upload call is unconditional and simply no-ops when reporting (or
`upload_artifacts`) is off. A shard left behind by a *failed* stage is
uploaded `complete: false`. Reporting is configured under the `reporting` key — see
[Configuration](Configuration#reporting-configs-configreporting) — and the
dashboard that consumes the contract is specified in `job-panel/README.md`.

## JSONL browser

`helpers/browse_jsonl.py` is a Textual TUI for stepping through a
JSONL file one record at a time. Each top-level field of the current
record is rendered as a collapsible Rich section, with nested dicts
laid out as sub-fields and lists rendered inline or as bullets
depending on whether they contain scalars. The viewer makes no
assumptions about field names — it is a general JSONL viewer.

Run it with:

```bash
uv run python helpers/browse_jsonl.py path/to/file.jsonl
```

Keybindings:

| Key                  | Action                                  |
|----------------------|-----------------------------------------|
| `n` / `→` / `space`  | Next sample                             |
| `p` / `←`            | Previous sample                         |
| `home` / `end`       | Jump to first / last sample             |
| `g`                  | Jump to a specific 1-based index        |
| `c`                  | Collapse / expand all sections          |
| `s`                  | Pick which top-level fields are shown   |
| `ctrl+r`             | Reload the file from disk               |
| `q`                  | Quit                                    |

On startup the column picker opens automatically when the file has any
fields, so the first interaction is choosing what to display.

## Improve-translations prompt

`improve-translations.j2` is the Jinja2 template used to ask
the LLM to review and rewrite a translation. It is loaded by the
[LLM stage](#llm-stage) (referenced as `improve-translations.j2` in the
example experiment) and paired with the `TranslationReview` schema. It
instructs the model to act as a {source_language}-to-{target_language}
reviewer following the MQM framework: classify issues by category and
severity, then produce three rewrites — a conservative `improved_primary`,
a polished `improved_polished`, and a meaningfully different
`improved_alternative`.

The template expects the following variables, which the example
experiment's modifiers populate (`google_translation` from Google
Translate, `target_language_tool_fixes` from LanguageTool):

- `source_language`, `target_language` — language names used throughout
  the instructions.
- `source_text`, `target_text` — the texts under review.
- `google_translation` — the reference translation produced by the
  Google Translate pre-pass (see [GoogleTranslator](#googletranslator)).
- `target_language_tool_fixes` — an optional list of LanguageTool
  findings; the LanguageTool-specific section of the prompt is only
  rendered when this is truthy.

## Pick-translations prompt

`qa-pick-translations.j2` is the template for the final stage of a
question-answer pipeline: given the candidate translations earlier stages
produced, it asks the model to **select and harmonize** one Q&A pair rather
than translate from scratch. It is loaded by the [LLM stage](#llm-stage) and
paired with the `QAPairSelection` schema (in
`DataCurator.pipeline.schemas`). Following the MQM framework, the model
evaluates each question candidate against the source question and each answer
candidate against the source answer, picks the single best *combination*
(optimising the pair, not the two highest-scoring texts in isolation),
checks the pair's internal consistency and Q-A relevance, and applies the
smallest harmonization edits needed to align terminology and register. Its
`QAPairSelection` output carries the per-candidate evaluations, the selected
ids and rationale, a `consistency_check`, the `harmonization_edits`, and the
`final_question`/`final_answer`.

The template expects these variables:

- `source_language`, `target_language` — language names, injected with
  [`ConstantsModifier`](#stage-modifiers).
- `source_question`, `source_answer` — the original-language pair.
- `question_candidates`, `answer_candidates` — lists of `{id, text}`, one
  entry per candidate. These are assembled by
  [`CandidatesModifier`](#stage-modifiers) from the three rewrites
  (`improved_primary`/`improved_polished`/`improved_alternative`) that each
  upstream `improve-translations` review stored in its
  `TranslationReview` output.
- `reference_question`, `reference_answer` — the Google-Translate references,
  used only for word-form checking.

This is the case the cross-stage field flow enables: because each record
carries its full context to the next stage, a final stage can read **both**
earlier reviews (`review_question`, `review_answer`) at once. For that to
work the upstream stages must not consume the raw fields the pick stage still
needs — so their [`RenameModifier`](#stage-modifiers)s copy with `keep: true`
instead of renaming, leaving the original source pair and Google-Translate
references intact for the final stage to reference.

## Context prompts

`gate-translation-rewrite-batch-context.j2`,
`fix-translation-context.j2` and
`pick-translation-id-batch-context.j2` are the gated pipeline's
templates with one addition: a record may carry `context_positives`, the
passages its query is meant to retrieve, as `[{id, text, translation}]`
(`translation` is `null` until a documents job has produced one), and the
prompts show them as a terminology anchor — a term the query shares with a
passage is rendered the way the passage's fixed translation renders it, never
judged for accuracy against it. The list comes from the `positives` column
`helpers/HF_mined_ds_to_jsonl_context.py` writes (best-first, capped), cut to
the experiment's `context_positives` by a `ListLimitModifier` in every LLM
stage — see `config/experiment/gate_queries_trivia_muse_batching_context.yaml`.

## Joint query and document runs

`gate_squadv2_muse_batching` translates the SQuAD v2 queries and documents in
one job, and `gate_squadv2_qwen38_batching` is the same experiment on
Qwen3.8-27B. The gated query and document experiments share every stage setting,
so both kinds run through the same stages. The input comes from
`helpers/HF_mined_ds_to_jsonl_segments.py`, which writes one
`{id, source_id, kind, source, text}` row per query and per unique document.
`kind` is `query`, or `positive`/`negative` for a document (positive if any
query lists it as one). `id` is prefixed `q-`/`d-` because the dataset's query
and document ids collide, and `source_id` keeps the original. The experiment's
`dataset.priority` orders the rows by `kind`.
`helpers/jsonl_to_HF_mined_ds.py --segments` splits the result by `kind` and
joins on `source_id`.

```bash
uv run python helpers/HF_mined_ds_to_jsonl_segments.py --out-prefix squadv2
```

## Pair adjustment

`adjust_squadv2_qwen38_batching` harmonizes the terms a SQuAD v2 query shares
with its positives after the joint run translated them independently. One
batched LLM stage (`adjust-pair-terminology-batch.j2`, response model
`PairAdjustment`) returns span edits, never reprinted text. `EditSpanValidator`
grounds every edit in an English term that both sources contain, and
`ApplyEditsModifier` writes `adjusted_segments`. The query and its positives
may be edited. The negatives (`context_negatives`, a `ListLimitModifier` cut)
are read-only context.

The input comes from `helpers/HF_mined_ds_to_jsonl_adjust.py`, one row per
query. Each row has `segments` (`Q`, then `P1`, `P2`, ... in English and
Polish) and best-first `negatives`. The helper reads the joint run's finalize
output from S3 with `--translations-s3`, a key or prefix taken from the bucket
root, or from disk with `--translations`. It skips a query when its own
translation or any of its positives' is missing.

`helpers/jsonl_to_HF_mined_ds.py --adjusted-s3` (or `--adjusted`) overlays
the result. A row the adjustment covered takes its query and positives from
`adjusted_segments`, so the row carries its own copy of each positive. Negatives
and the remaining rows keep the plain translations.

```bash
uv run python helpers/HF_mined_ds_to_jsonl_adjust.py --out-prefix squadv2 \
    --translations-s3 datacurator/gate_squadv2_qwen38_batching/stage/04_finalize/157f9f9a/
```

## Filling translation gaps

`helpers/fill_translations_from_HF.py` rejoins joint-run outputs onto the English
rows and fills items they lack from an already translated HuggingFace dataset.
Runs are merged by `id`, base first, and flagged records are ignored. Each item
then takes the experiment's translation, or otherwise the fallback's text for the
same id. A blank fallback text, or one identical to the English source, does not
count. A row with an unresolved item is emptied as in
`helpers/jsonl_to_HF_mined_ds.py`. Other rows gain `translation_source`, which
names each item's origin as `experiment` or `huggingface`. The fallback is pinned
by `--fallback-revision` and `--fallback-config`.

```bash
uv run python helpers/fill_translations_from_HF.py \
    --segments-s3 datacurator/gate_squadv2_qwen38_batching/stage/04_finalize/157f9f9a/ \
    datacurator/gate_squadv2_qwen38_batching_rerun/stage/04_finalize/35fb967e/
```

## LanguageTool service

`languagetool/` packages a self-hosted LanguageTool deployment as a
Docker Compose stack. The Compose file brings up Traefik as a reverse
proxy and runs the LanguageTool service in 16 replicas behind it; the
custom `Dockerfile` layers fasttext and the `lid.176.bin` model on top
of the upstream `erikvl87/languagetool:6.7` image so language
identification works without an external download at runtime.

Bring it up with:

```bash
cd languagetool
cp .env.example .env
docker compose up -d --build
```

The Traefik router exposes the service at `http://languagetool.loc` and at
the hostname in `LANGUAGETOOL_PUBLIC_HOST`, which the stack requires in `.env`;
the upstream container listens on port 8010 and is health-checked
against `/v2/languages`. Tuning is environment-driven inside the
Compose file — notable settings include `langtool_maxCheckThreads`,
`langtool_pipelineCaching`, `langtool_cacheSize`, and `Java_Xmx`.
