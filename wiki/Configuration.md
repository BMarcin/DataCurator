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
| `runner.pause_on_model_switch` | Block for operator confirmation when a stage needs a different `model`. |
| `runner.start_from`            | Resume the pipeline at this stage id, reading the prior stage's output. |

## Experiment configs (`config/experiment/`)

An experiment declares the pipeline flow. `example.yaml` composes stage
configs into `pipeline.<id>` via the defaults list, lists the execution
`order`, and points at the dataset. Each stage's output JSONL becomes the
next stage's input.

| Key                  | Meaning                                                  |
|----------------------|----------------------------------------------------------|
| `experiment_name`    | Name used in output paths and notifications.             |
| `dataset.input`      | Path to the input JSONL dataset.                         |
| `dataset.output_dir` | Directory for per-stage `<NN>_<stage>.jsonl` outputs.    |
| `order`              | Ordered list of stage ids to run.                        |
| `pipeline.<id>`      | Per-stage config; overrides the composed stage template. |

## Stage configs (`config/stages/`)

`regex_fix.yaml` is the in-tree stage template. The `stage` block is what
`build_pipeline` instantiates; the sibling keys are runner metadata.

| Key               | Meaning                                                                  |
|-------------------|--------------------------------------------------------------------------|
| `enabled`         | When false, the stage is skipped and its input passes through unchanged. |
| `model`           | Model label used only for model-switch detection; `null` means no model. |
| `stage._target_`  | Import path of the `Stage` subclass to instantiate.                      |
| `stage.field`     | Field the `RegexFixStage` rewrites.                                      |
| `stage.rules`     | Ordered list of `{pattern, replace}` substitutions.                      |
| `stage.modifiers` | List of modifier configs (each with `_target_`) attached to the stage.   |

## Environment variables

These are read by config files via `${oc.env:VAR,null}`. Missing values
resolve to `null` rather than raising.

| Variable     | Used by                                | Purpose                              |
|--------------|----------------------------------------|--------------------------------------|
| `NTFY_URL`   | `config/notifications/ntfy.yaml`       | ntfy server base URL.                |
| `NTFY_TOKEN` | `config/notifications/ntfy.yaml`       | Bearer token for private ntfy topics. |

## Model config (`config/model/`)

`qwen35.yaml` is the in-tree example targeting a vLLM-served
Qwen3.6-35B endpoint. The same keys apply to any OpenAI-compatible
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
