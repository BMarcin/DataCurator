# Configuration

DataCurator is configured through Hydra. Configs live under `config/`
and are grouped by concern — `config/config.yaml` is the entrypoint,
`config/experiment/` declares the pipeline flow, `config/stages/` holds
reusable stage templates, `config/model/` defines LLM endpoints, and
`config/notifications/` configures the notifier. Secrets are kept out of
the config files and read from `.env` via OmegaConf's `oc.env` resolver.
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
| `dataset.input`              | Input JSONL dataset: a single file, a glob, or a list of either. |
| `dataset.output_dir`         | Directory holding each stage's shard subdir `<NN>_<stage>/step-NNNN.jsonl`. |
| `dataset.limit`              | Process only the first N input rows (`null` = all); handy for smoke runs. |
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
| `stage._target_`       | Import path of the `Stage` subclass to instantiate.                           |
| `stage.model`          | Required model preset for this stage, e.g. `${_models.qwen35}` (no default).  |
| `stage.response_model` | Dotted path to the Pydantic structured-output schema.                         |
| `stage.output_field`   | Context field the parsed result is stored under.                              |
| `stage.input_field`    | Field holding a pre-built conversation (used when `prompt` is null).          |
| `stage.prompts_dir`    | Directory holding the Jinja2 prompt files (`./prompts` by default).           |
| `stage.prompt`         | Optional templated conversation: `{role, template}` entries naming prompt files. |
| `stage.modifiers`      | List of modifier configs (each with `_target_`) attached to the stage.        |

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
| `DataCurator.pipeline.modifiers.google_translate.GoogleTranslateModifier` | `source_language_code`, `target_language_code` (required); `retries`, `timeout`, `concurrency`. |
| `DataCurator.pipeline.modifiers.languagetool.LanguageToolModifier`        | `language` (e.g. `pl-PL`), `remote_server`, `issues_field` (where unresolved issues are stashed), `allowed_fixes`, `retries`, `max_passes`. |

The two **structural modifiers** reshape the context rather than transform a
single field:

| `_target_`                                                  | Key arguments                                                                                       |
|------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `DataCurator.pipeline.modifiers.rename.RenameModifier`     | `fields` (an `input -> output` mapping); `keep` (false = rename/drop input, true = copy — use this when a later stage still needs the original); `required`; `exists_ok` (false = raise if an output field already exists, true = overwrite); `phase`. |
| `DataCurator.pipeline.modifiers.constants.ConstantsModifier` | `values` (a `name -> literal` mapping injected into the context); `overwrite`/`exists_ok` (false = raise on an existing field, true = replace); `phase`. |
| `DataCurator.pipeline.modifiers.candidates.CandidatesModifier` | `target` (field to write); `sources` (ordered field names or dotted paths to collect into a `[{id, text}, …]` list); `id_prefix` (ids become `{prefix}{n}` from 1, e.g. `Q` → `Q1`/`Q2`/`Q3`); `required` (false = drop missing sources); `exists_ok`; `phase`. |

## Environment variables

These are read by config files via `${oc.env:VAR,null}`. Missing values
resolve to `null` rather than raising.

| Variable     | Used by                                | Purpose                              |
|--------------|----------------------------------------|--------------------------------------|
| `NTFY_URL`   | `config/notifications/ntfy.yaml`       | ntfy server base URL.                |
| `NTFY_TOKEN` | `config/notifications/ntfy.yaml`       | Bearer token for private ntfy topics. |
| `JOB_S3_ENDPOINT` | `config/reporting/s3.yaml`        | S3 endpoint URL for job reporting.   |
| `JOB_S3_REGION` | `config/reporting/s3.yaml`          | S3 region (defaults to `garage`).    |
| `JOB_S3_BUCKET` | `config/reporting/s3.yaml`          | Bucket holding the `<prefix>/<job_id>/` objects. |
| `JOB_S3_ACCESS_KEY_ID` | `config/reporting/s3.yaml`   | S3 access key id.                    |
| `JOB_S3_SECRET_ACCESS_KEY` | `config/reporting/s3.yaml` | S3 secret access key.                |

## Model config (`config/model/`)

Each file under `config/model/` is a model preset. Two ship in-tree:
`qwen35.yaml` (a vLLM-served Qwen3.6-35B endpoint) and `bielik.yaml`
(a Bielik-11B endpoint). An experiment loads the presets it needs under
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
| `concurrency`        | Max concurrent in-flight requests.                       |
| `retries`            | Retry budget per request.                                |
| `top_p`              | Nucleus sampling threshold.                              |
| `top_k`              | Top-k sampling (vLLM, via `extra_body`).                 |
| `min_p`              | Minimum-probability filter (vLLM, via `extra_body`).     |
| `presence_penalty`   | OpenAI presence penalty.                                 |
| `repetition_penalty` | Repetition penalty (vLLM, via `extra_body`).             |
| `enable_thinking`    | Qwen3 thinking mode toggle (via `extra_body`).           |
| `extra_body`         | Free-form passthrough merged last into `extra_body`.     |

## Notification configs (`config/notifications/`)

Two variants ship: `disabled.yaml` (the default — sets `enabled: false`
and nothing else) and `ntfy.yaml`. The notifier reads its sub-config
through `notify(notifications_cfg, ...)` and ignores everything when
`enabled` is false.

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
reporter and the job runs unchanged. Connection secrets come from `.env`
via `${oc.env:...}` (see [Environment variables](#environment-variables)).
Enable with `uv run python run_pipeline.py reporting=s3`. The consumer-side
contract for these objects is documented in `job-panel/README.md`.

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
