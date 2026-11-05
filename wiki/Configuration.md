# Configuration

DataCurator is configured through Hydra. Configs live under `config/`
and are grouped by concern — `config/config.yaml` is the entrypoint,
`config/experiment/` declares the pipeline flow, `config/stages/` holds
reusable stage templates, `config/model/` defines LLM endpoints, and
`config/notifications/` configures the notifier. Secrets are kept out of
the config files and read from the process environment (`.env.pipeline`
for standalone runs) via OmegaConf's `oc.env` resolver.
This page lists the keys that exist today; refer to
[Components](Components) for how each one is consumed.

## Top-level config (`config/config.yaml`)

`config.yaml` is the entrypoint composed by `run_pipeline.py`. Its
`defaults` pick an `experiment` (the flow of stages) and a
`notifications` backend, and it sets `hydra.job.chdir: false` so dataset
paths resolve from the directory you launch from. The `runner` block holds
the execution knobs read by `PipelineRunner`.

| Key                            | Meaning                                                                 |
|--------------------------------|-------------------------------------------------------------------------|
| `runner.concurrency`           | Max records processed in parallel within a stage.                       |
| `runner.resume`                | Skip already-processed records when the stage signature is unchanged.   |
| `runner.debug`                 | Extra per-record logging (every stage output is persisted regardless).  |
| `runner.id_field`              | Record key used to match and skip work across reruns.                   |
| `runner.shard_size`            | Records per output shard; each full shard is persisted (and uploaded as an artifact when reporting is on). |
| `runner.pause_between_stages`  | Block for operator confirmation before each stage (for manual steps in between). |
| `runner.start_from`            | Resume the pipeline at this stage id, reading the prior stage's output. |
| `runner.progress`              | Live stage + per-modifier progress bars; auto-disabled when not attached to a TTY. |
| `runner.flag_on_errors`        | Exception types (dotted import paths) that flag the offending record instead of aborting the run; any exception not listed still fails the pipeline. Defaults to `openai.LengthFinishReasonError`, `openai.BadRequestError`, `openai.APITimeoutError`, `DataCurator.pipeline.stages.llm.UnparseableOutputError`, and `DataCurator.pipeline.validators.ResponseValidationError`. Set to `[]` to abort on every error. |

### Flagging unprocessable records

Some failures are inherent to a single record — the model truncates its
output at `max_tokens` (`openai.LengthFinishReasonError`) or the prompt
overflows the served context window (`openai.BadRequestError`, an HTTP 400
from vLLM); others are transient but persistent, like a request that keeps
timing out (`openai.APITimeoutError`), a reply that never parses into the
response schema such as a refusal
(`DataCurator.pipeline.stages.llm.UnparseableOutputError`), or output that
parses but fails a stage [validator](#validator-configs)
(`DataCurator.pipeline.validators.ResponseValidationError`). Rather than abort
the whole run on one such record, `runner.flag_on_errors` lets the runner flag
it and continue.
When a stage raises a listed exception (after its `model.retries` attempts),
the runner writes the record to the stage output with `error: true`,
`error_message`, `error_type`, and `error_stage`, and later stages pass any
record carrying `error: true` through unchanged rather than reprocessing it.
The per-stage flagged count is logged and, when reporting is on, published in
the dashboard `detail` as `stages[].errored` plus a job-wide `total_errored`;
the count also appears in the pipeline-end notification.

## Experiment configs (`config/experiment/`)

An experiment declares the pipeline flow. `example.yaml` composes stage
configs into `pipeline.<id>` via the defaults list, lists the execution
`order`, and points at the dataset. Each stage's output shard directory
becomes the next stage's input.

Model presets are loaded under the `_models` namespace in the defaults
list (e.g. `- /model/qwen35@_models.qwen35`), and each stage selects one
with `pipeline.<id>.stage.model: ${_models.<name>}`. This is how
different stages run on different LLMs — load as many presets as you like
and point each stage at whichever it needs. When stages use different
models, set `runner.pause_between_stages: true` so the runner waits while
you swap the deployed model on the GPU between stages.

| Key                          | Meaning                                                  |
|------------------------------|----------------------------------------------------------|
| `experiment_name`            | Name used in output paths, notifications, and as the reporting `job_id`. |
| `prompt_dir`                 | This experiment's prompt-file folder, by convention `assets/prompts/<experiment config name>`; stages reference files as `${prompt_dir}/<name>.j2`. See [Prompt sources](#prompt-sources). |
| `dataset.input`              | Input JSONL dataset: a single file, a glob, or a list of either. |
| `dataset.output_dir`         | Directory holding each stage's shard subdir `<NN>_<stage>/step-NNNN.jsonl`. |
| `dataset.limit`              | Process only the first N input rows (`null` = all); handy for smoke runs. |
| `dataset.priority`           | Optional processing order: `{field, order}` stable-sorts the input rows by the position of `field`'s value in `order`; unlisted values go last, and `null` keeps file order. It applies before `limit`. The distributed seed cuts chunks in this order (see [Distributed](Distributed)); the standalone runner starts stage-0 records in it. |
| `job.type`                   | Renderer dispatch key written into the reporting envelope (e.g. `data-pipeline`). |
| `job.links`                  | Deep-links shown on the dashboard, each `{label, url}`.  |
| `order`                      | Ordered list of stage ids to run.                        |
| `_models.<name>`             | A model preset loaded from `config/model/`, referenced per stage as `${_models.<name>}`. |
| `pipeline.<id>`              | Per-stage config; overrides the composed stage template. |
| `pipeline.<id>.enabled`      | When false, the stage is skipped (also overridable on the CLI, e.g. `pipeline.review.enabled=false`). |
| `pipeline.<id>.stage.model`  | The model preset this stage runs on, e.g. `${_models.qwen35}`. |

## Stage configs (`config/stages/`)

`llm.yaml` is the in-tree stage template. The `stage` block is what
`build_pipeline` instantiates; the sibling keys are runner metadata. Its
`stage.model` is a required field (`???`), so each stage that uses the
template must supply a model — point it at a preset loaded under
`_models` in the experiment, e.g. `${_models.qwen35}`.

| Key                    | Meaning                                                                       |
|------------------------|-------------------------------------------------------------------------------|
| `enabled`              | When false, the stage is skipped and its input passes through unchanged.      |
| `pause_before`         | Optional per-stage override of `runner.pause_between_stages` (unset = inherit). |
| `filters`              | Optional list of input filters (each with `_target_`); only records kept by **every** filter are processed. See [Filtering and rerunning records](#filtering-and-rerunning-records). |
| `drop_fields`          | Optional list of field names stripped from each input record (after filtering), e.g. clearing stale `error*` flags so a record reprocesses. |
| `stage._target_`       | Import path of the `Stage` subclass to instantiate.                           |
| `stage.model`          | Required model preset for this stage, e.g. `${_models.qwen35}` (no default).  |
| `stage.response_model` | Dotted path to the Pydantic structured-output schema.                         |
| `stage.output_field`   | Context field the parsed result is stored under.                              |
| `stage.input_field`    | Field holding a pre-built conversation (used when `prompt` is `[]`).          |
| `stage.prompt`         | Required templated conversation: `{role, template}` (path to a `.j2` file) or `{role, text}` (inline template) entries; `[]` reads `input_field` instead. See [Prompt sources](#prompt-sources). |
| `stage.modifiers`      | List of modifier configs (each with `_target_`) attached to the stage.        |
| `stage.validators`     | Optional list of output guards (each with `_target_`), run inside the retry loop. See [Validator configs](#validator-configs). |
| `stage.keep_rejected_output` | When a validator rejects an output across all retries (flagging the record), keep the parsed output that failed under `rejected_output_field` instead of discarding it. Off by default. |
| `stage.rejected_output_field` | Field a rejected output is stashed under when `keep_rejected_output` is on (default `rejected_output`). |

### Prompt sources

Prompts belong to an experiment, not to the code, so prompt files are kept
outside git: `assets/` is gitignored, and each experiment sets `prompt_dir`
to its own folder. Every message in `stage.prompt` sets exactly one of
`template` (a path, resolved against the working directory) or `text` (the
Jinja2 template inline); setting both or neither fails the build, as does a
missing file. Before a stage is built and signed, `inline_prompt_files`
replaces each `template` with the file's `text`. The resume signature
therefore follows the prompt text: editing a prompt recomputes the stage,
moving the file does not. The runner writes the stage's prompts to
`<NN>_<stage>/prompts.json` and, with reporting on, uploads it under
`prompts/`.

```yaml
prompt_dir: assets/prompts/gate_squadv2_qwen38_batching
pipeline:
  fix_document:
    stage:
      prompt:
        - role: system
          text: "You are a careful Polish translator."
        - role: user
          template: ${prompt_dir}/fix-translation.j2
```

Tests that need an experiment's prompt files are marked
`prompt_assets(<experiment>)` and are skipped when its folder is absent.

### Modifier configs

Modifiers are declared inline in a stage's `modifiers` list, each with a
`_target_` and its constructor arguments, and run in listed order. The five
in-tree modifiers fall into two groups.

The two **field modifiers** share four knobs: `source` (field to read,
required), `target` (field to write — defaults to `source` for an in-place
rewrite; naming a new field adds it), `phase` (`before` (default) to enrich
the prompt's inputs, or `after` to post-process output), and `required`
(`false` (default) skips the modifier when `source` is absent; `true`
raises). Their extra arguments:

| `_target_`                                                              | Extra arguments                                                                            |
|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| `DataCurator.pipeline.modifiers.google_translate.GoogleTranslateModifier` | `source_language_code`, `target_language_code` (required); `retries`, `timeout`, `concurrency`; `remote_server` and `client_name` come from the bindings. |
| `DataCurator.pipeline.modifiers.languagetool.LanguageToolModifier`        | `language` (e.g. `pl-PL`), `remote_server`, `issues_field` (where unresolved issues are stashed), `allowed_fixes`, `retries`, `max_passes`. |

The two **structural modifiers** reshape the context rather than transform a
single field:

| `_target_`                                                  | Key arguments                                                                                       |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `DataCurator.pipeline.modifiers.rename.RenameModifier`     | `fields` (an `input -> output` mapping); `keep` (false = rename/drop input, true = copy — use this when a later stage still needs the original); `required`; `exists_ok` (false = raise if an output field already exists, true = overwrite); `phase`. |
| `DataCurator.pipeline.modifiers.constants.ConstantsModifier` | `values` (a `name -> literal` mapping injected into the context); `overwrite`/`exists_ok` (false = raise on an existing field, true = replace); `phase`. |
| `DataCurator.pipeline.modifiers.candidates.CandidatesModifier` | `target` (field to write); `sources` (ordered field names or dotted paths to collect into a `[{id, text}, …]` list); `id_prefix` (ids become `{prefix}{n}` from 1, e.g. `Q` → `Q1`/`Q2`/`Q3`); `required` (false = drop missing sources); `exists_ok`; `phase`. |
| `DataCurator.pipeline.modifiers.transfer.TransferCheckModifier` | `check` (`script` or `patterns`, required); `target` (findings column, required); `source_field`/`hypothesis_field` (default `source_text`/`target_text`); `allow_scripts` (script check); `kinds`/`mode` (patterns check); `phase` (default `after`). Writes `{}` when clean, nothing when an input is missing; pair with a `SetWhenModifier` to act on the findings. |

### Validator configs

An [LLM stage](Components#llm-stage) can carry `validators` — output guards
declared like `modifiers`, each with a `_target_` — that inspect the parsed
output and raise to reject it. They run **inside** the stage's retry loop, so a
rejection resamples; once retries are exhausted the error re-raises and, if
`DataCurator.pipeline.validators.ResponseValidationError` is in
[`runner.flag_on_errors`](#flagging-unprocessable-records), the record is
flagged rather than aborting the run (see [Response
validators](Components#response-validators)). The in-tree validators read a
`reference` field from the context and check one or more output `fields`
(both accept dotted paths); a reference missing from the context skips the
check unless `required=True`, and `skip_empty_fields` skips an output field
whose value is empty (the gate's `""` "no fix" sentinel). `required` is only
ever about a missing *field*: the guards backed by a service (LanguageTool,
MetricX) always run, and an unreachable server raises
`ServiceUnavailableError` rather than skipping — a record must never pass a
check that did not happen. On the fleet that error is connection-class, so the
chunk is nak'd and redelivered instead of uploaded with its records flagged.

| `_target_`                                                  | Key arguments                                                                                       |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `DataCurator.pipeline.validators.LengthRatioValidator`     | `fields`, `reference` (required); at least one of `min_ratio`/`max_ratio` bounding `len(field)/len(reference)`; `unit` (`char` (default) or `word`); `required`; `skip_empty_fields`; `name`. Catches truncated rewrites that still parse. |
| `DataCurator.pipeline.validators.CharacterCountValidator`  | `fields`, `reference`, `characters` (required, each counted with `str.count`); `max_diff` (default 0 = exact match); `required`; `skip_empty_fields`; `name`. Catches stray escapes/punctuation drift vs the source. |
| `DataCurator.pipeline.validators.ScriptConsistencyValidator` | `fields`, `reference` (required); `allow_scripts` (extra allowed scripts beyond the reference's own); `required`; `skip_empty_fields`; `name`. Catches foreign-script leakage (CJK/Cyrillic artifacts in a Polish rewrite). |
| `DataCurator.pipeline.validators.PatternPreservationValidator` | `fields`, `reference` (required); `kinds` (subset of `number`/`url`/`email`, default all); `mode` (`missing` (default) or `both`); `max_diff` (number misses tolerated, default 0 — URL/email misses always fail); `required`; `skip_empty_fields`; `name`. Catches dropped numbers, URLs and emails; probe thresholds with `scripts/probe_transfer_checks.py`. |
| `DataCurator.pipeline.validators.EditSpanValidator` | `edits` (output path, default `edits`), `segments` (context path, default `segments`); `text_key`/`source_key` (default `translation`/`text`); `min_segments_with_term` (default 2); `max_words` (default 6); `max_edits` (default 6); `name`. Grounds and bounds span edits; see [Pair adjustment](Components#pair-adjustment). |

### Filtering and rerunning records

A stage can carry `filters` — a list declared like `modifiers`, each with a
`_target_` — that select **which input records the stage processes**. The
runner applies them to the stage's input before resume and `limit`, so a
record dropped by a filter is never processed, and totals/progress/dashboard
reflect the filtered set. Several filters combine with AND (a record must be
kept by every one).

The in-tree filter is `DataCurator.pipeline.filters.ExpressionFilter`, which
keeps a record when its `expression` evaluates truthy against the record's
fields. The expression is ordinary Python-flavoured syntax — bare names are
field lookups, `a.b`/`a["b"]` index nested mappings, and `and`/`or`/`not`,
comparisons (`== != < <= > >= in not in`) and the helpers
`len`/`abs`/`str`/`int`/`float`/`bool` are supported (it is parsed and
allow-list-validated at build time, never `eval`'d). A field absent from the
record reads as falsy and unequal to everything, so `error == True` simply
drops records without an `error` key; `true`/`false`/`null` are accepted as
aliases for `True`/`False`/`None`.

**Rerunning failed records.** Flagged records (`error: true`, see [Flagging
unprocessable records](#flagging-unprocessable-records)) are skipped on resume,
so reruns are done as a **separate experiment** that reads the original run's
output and keeps only the failures — it never touches the original's shards.
`config/experiment/queries_fiqa_no_thinking_rerun.yaml` is the worked example:
it points `dataset.input` at the original's first-stage output
(`dataset/fix_queries_fiqa_no_thinking/00_fix_queries/step-*.jsonl`), and the
first stage adds

```yaml
drop_fields: [error, error_message, error_type, error_stage]
filters:
  - _target_: DataCurator.pipeline.filters.ExpressionFilter
    expression: "error == True"
```

The filter keeps only the flagged records; `drop_fields` then clears the
`error*` flags so they are reprocessed (otherwise the runner would pass them
through as already-errored). Each record keeps its original `id`, so the fixed
rows can be merged back by `id`. Run it with
`uv run --env-file .env.pipeline python run_pipeline.py experiment=queries_fiqa_no_thinking_rerun`
(`documents_fiqa_no_thinking_rerun.yaml` is the documents twin). To rerun a
failure from a *later* stage instead, point `dataset.input` at that stage's
output and filter on e.g. `error == True and error_stage == "<stage_id>"`.

Once the rerun finishes, `helpers/combine_experiments.py` merges the base run
and one or more reruns by `id`, keeping the best record for each (a successful
fix replaces the base run's failure, and a success is never regressed by a
later error). Pass the matching stage directory from each run, base first:

```
uv run python helpers/combine_experiments.py \
    --out data/queries_fiqa_combined.jsonl \
    dataset/fix_queries_fiqa_no_thinking/01_pick_the_best_translation \
    dataset/fix_queries_fiqa_no_thinking_rerun/01_pick_the_best_translation
```

Add `--drop-still-errored` to omit records that every rerun still failed.

## Environment variables

These are read by config files via `${oc.env:VAR,null}`. Missing values
resolve to `null` rather than raising.

| Variable     | Used by                                | Purpose                              |
|--------------|----------------------------------------|--------------------------------------|
| `GOOGLE_TRANSLATE_URL` | `config/bindings/local.yaml` | Full URL of the Google Translate endpoint for standalone runs. |
| `GOOGLE_TRANSLATE_CLIENT` | `config/bindings/local.yaml` | Value of that endpoint's `client` query parameter for standalone runs. |
| `NTFY_URL`   | `config/notifications/ntfy.yaml`       | ntfy server base URL.                |
| `NTFY_TOKEN` | `config/notifications/ntfy.yaml`       | Bearer token for private ntfy topics. |
| `JOB_S3_ENDPOINT` | `config/reporting/s3.yaml`        | S3 endpoint URL for job reporting.   |
| `JOB_S3_REGION` | `config/reporting/s3.yaml`          | S3 region (defaults to `garage`).    |
| `JOB_S3_BUCKET` | `config/reporting/s3.yaml`          | Bucket holding the `<prefix>/<job_id>/` objects. |
| `JOB_S3_ACCESS_KEY_ID` | `config/reporting/s3.yaml`   | S3 access key id.                    |
| `JOB_S3_SECRET_ACCESS_KEY` | `config/reporting/s3.yaml` | S3 secret access key.                |

## Model config (`config/model/`)

Each file under `config/model/` is a model preset. Several ship in-tree. The
`qwen35_*.yaml` set all point at the same vLLM-served Qwen3.6-35B endpoint but
are tuned per pipeline stage — `qwen35_stage1.yaml`/`qwen35_stage1_no_thinking.yaml`
(the review pass, the `_no_thinking` variant disabling Qwen3 thinking blocks),
`qwen35_stage2.yaml` (the lower-temperature pick pass) and `qwen35_gate.yaml`
(the near-greedy gate). The `muse_*.yaml` set mirrors them on Muse-Glimmer-30B;
Muse has no `enable_thinking` toggle and no way to switch reasoning off — its
chat template appends `Reasoning strength: low|medium|high|xhigh` *after* the
system prompt, so the strength is set by the preset's
`extra_body.chat_template_kwargs.reasoning_strength` (unset defaults to `high`)
and a system-prompt line saying otherwise is overridden. They also set
`structured_output: tool`: Muse leaves its reasoning phase only when a tool
channel opens, so a `response_format` grammar is never applied to a plain
answer — the schema is ignored, nothing errors, and the reply arrives as prose.
The same schema sent as a forced tool call is enforced, reasoning intact.
`bielik.yaml` points at a Bielik-11B endpoint. `index_translate.yaml` points at
Index-Translate-9B, a dedicated translation model served as `index-translate`,
with greedy decoding and `enable_thinking: false` as its model card specifies.
`gate_queries_squadv2_index_translate_batching` uses it to write the draft
translation that the muse gate, fix and pick stages then check. An experiment loads the presets it needs under
`_models.<name>` and each stage selects one; add more files here to give
stages more choices. The same keys apply to any OpenAI-compatible
endpoint. Top-level keys are forwarded as OpenAI fields; vLLM-only
sampling controls (`top_k`, `min_p`, `repetition_penalty`) and the
Qwen3 `enable_thinking` toggle are routed into `extra_body`. The
trailing `extra_body` map is merged last and overrides anything above
it.

| Key                  | Meaning                                                  |
|----------------------|----------------------------------------------------------|
| `name`               | Model identifier sent to the endpoint.                   |
| `api_base`           | Base URL of the OpenAI-compatible endpoint.              |
| `api_key`            | API key (use `EMPTY` for unauthenticated local servers). |
| `temperature`        | Sampling temperature.                                    |
| `max_tokens`         | Per-request output cap.                                  |
| `concurrency`        | Max concurrent in-flight requests. Standalone only: on the fleet each worker sets it per model in its bindings file. |
| `retries`            | Retry budget per request.                                |
| `timeout`            | Per-request timeout in seconds (`null` = OpenAI default of 600s). Standalone only, like `concurrency`. |
| `top_p`              | Nucleus sampling threshold.                              |
| `top_k`              | Top-k sampling (vLLM, via `extra_body`).                 |
| `min_p`              | Minimum-probability filter (vLLM, via `extra_body`).     |
| `presence_penalty`   | OpenAI presence penalty.                                 |
| `repetition_penalty` | Repetition penalty (vLLM, via `extra_body`).             |
| `enable_thinking`    | Qwen3 thinking mode toggle (via `extra_body`).           |
| `structured_output`  | How the response schema reaches the model: `json` (default, `response_format`) or `tool` (a forced function call, for a backend whose grammar ignores `response_format`). |
| `extra_body`         | Free-form passthrough merged last into `extra_body`.     |

## Notification configs (`config/notifications/`)

Three variants ship: `disabled.yaml` (sets `enabled: false` and nothing else),
`ntfy.yaml`, and `ntfy_pub.yaml` (a ready-to-use public ntfy.sh topic, the
default in `config.yaml`). The notifier reads its sub-config through
`notify(notifications_cfg, ...)` and ignores everything when `enabled` is
false.

| Key                 | Meaning                                                                  |
|---------------------|--------------------------------------------------------------------------|
| `enabled`           | Master switch; when false, `notify()` is a no-op.                        |
| `backend`           | Notifier backend; only `ntfy` is implemented.                            |
| `server`            | ntfy server URL. Reads `$NTFY_URL`.                                      |
| `topic`             | ntfy topic to publish to. Required when enabled.                         |
| `priority`          | One of `min`, `low`, `default`, `high`, `urgent`.                        |
| `tags`              | List of ntfy emoji-style tags, joined with commas in the `Tags` header.  |
| `auth_token`        | Optional bearer token. Reads `$NTFY_TOKEN`.                              |
| `timeout`           | HTTP timeout in seconds.                                                 |
| `on_stage_start`    | Whether to emit a notification at stage start.                           |
| `on_stage_end`      | Whether to emit a notification at stage end.                             |
| `on_pipeline_end`   | Whether to emit a notification at pipeline completion.                   |
| `on_error`          | Whether to emit a notification on error.                                 |

## Reporting configs (`config/reporting/`)

Two variants ship: `disabled.yaml` (the default — sets `enabled: false`)
and `s3.yaml`, which publishes live job status, host metrics and stage
artifacts to an S3-compatible bucket for an external dashboard. The runner
builds a reporter with `build_reporter(cfg.reporting, ...)`; when reporting
is off, misconfigured, or missing a bucket it falls back to a no-op
reporter and the job runs unchanged. Connection secrets come from `.env.pipeline`
via `${oc.env:...}` (see [Environment variables](#environment-variables)).
Enable with `uv run --env-file .env.pipeline python run_pipeline.py reporting=s3`.

| Key                     | Meaning                                                                  |
|-------------------------|--------------------------------------------------------------------------|
| `enabled`               | Master switch; when false, reporting is a no-op.                         |
| `backend`               | Reporting backend; only `s3` is implemented.                            |
| `endpoint_url`          | S3 endpoint URL. Reads `$JOB_S3_ENDPOINT`.                              |
| `region`                | S3 region. Reads `$JOB_S3_REGION` (defaults to `garage`).               |
| `bucket`                | Target bucket. Reads `$JOB_S3_BUCKET`. Required when enabled.            |
| `access_key_id`         | S3 access key id. Reads `$JOB_S3_ACCESS_KEY_ID`.                        |
| `secret_access_key`     | S3 secret access key. Reads `$JOB_S3_SECRET_ACCESS_KEY`.                |
| `prefix`                | Key prefix; objects live under `<bucket>/<prefix>/<job_id>/`.            |
| `heartbeat_interval_s`  | How often status (and liveness) is refreshed while a stage runs.         |
| `sample_system_metrics` | Include a CPU/RAM/GPU snapshot in each status push.                      |
| `upload_artifacts`      | Upload each output shard as a downloadable artifact the moment it closes (shard size is `runner.shard_size`). |
| `metrics_flush_s`       | Minimum interval between `metrics.jsonl` re-uploads.                     |
