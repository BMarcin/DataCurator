# Configuration

DataCurator is configured through Hydra. Configs live under `config/`
and are grouped by concern — `config/model/` for LLM endpoints,
`config/notifications/` for the notifier. Secrets are kept out of the
config files and read from `.env` via OmegaConf's `oc.env` resolver.
This page lists the keys that exist today; refer to
[Components](Components) for how each one is consumed.

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
