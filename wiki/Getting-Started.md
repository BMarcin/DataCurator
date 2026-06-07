# Getting started

DataCurator targets Python 3.13 and is managed with `uv`. All commands
are expected to run through `uv run` so the project venv is used. The
repo exposes a Hydra-driven pipeline runner (`run_pipeline.py`) plus
individual components (a Google Translate client, a ntfy notifier, an
S3 job reporter, a JSONL browser, a prompt template, and a LanguageTool
Docker service) — the sections below cover the pieces that are runnable
today. See
[Components](Components) for a deeper description of each one and
[Configuration](Configuration) for the keys they accept.

## Install

```bash
git clone <repo-url> DataCurator
cd DataCurator
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock` and provisions the
virtual environment. Do not invoke `pip` directly.

## Configure secrets

Secrets are read from a `.env` file at the repo root (see `.env.example`
for a template). Config files reference them via OmegaConf's `oc.env`
resolver — the notifier reads ntfy credentials and, when reporting is
enabled, the S3 connection:

```env
NTFY_URL=https://ntfy.example.com
NTFY_TOKEN=tk_xxxxxxxxxxxxxxxxxxxx
JOB_S3_ENDPOINT=https://s3.example.com
JOB_S3_BUCKET=jobs
JOB_S3_ACCESS_KEY_ID=...
JOB_S3_SECRET_ACCESS_KEY=...
```

All are optional — `oc.env` falls back to `null` when a variable is unset,
and both the notifier and the reporter degrade to no-ops when their
required values are missing. See [Configuration](Configuration#environment-variables)
for the full list.

## Run the pipeline

`run_pipeline.py` is the Hydra entrypoint. It runs whichever experiment,
notifications and reporting groups `config/config.yaml` selects in its
`defaults` list (override any of them on the CLI). The bundled `example`
experiment runs one LLM review stage with Google Translate and LanguageTool
modifiers enriching each record first, writing one JSONL file per stage
under `dataset/example/`:

```bash
uv run python run_pipeline.py experiment=example
```

This experiment expects a reachable OpenAI-compatible endpoint (the
`review` stage uses `config/model/qwen35.yaml`; stages can each
run on a different preset — see [Configuration](Configuration#experiment-configs-configexperiment))
and, for the LanguageTool modifier, the
[LanguageTool service](#start-the-languagetool-service). The flow is
config-driven, so you can change it from the command line — switch
experiments, disable a stage, resume from one, or toggle notifications and
reporting:

```bash
uv run python run_pipeline.py pipeline.review.enabled=false
uv run python run_pipeline.py runner.start_from=review
uv run python run_pipeline.py notifications=ntfy notifications.topic=my-topic
uv run python run_pipeline.py reporting=disabled
```

Reruns skip records already present in a stage's output unless that
stage's config changed. See [Configuration](Configuration) for the full
key set and [Components](Components#pipeline-runner) for how the runner
behaves.

## Browse a JSONL file

The Textual viewer in `helpers/browse_jsonl.py` opens any JSONL file
sample-by-sample. Each top-level field renders as a collapsible
section.

```bash
uv run python helpers/browse_jsonl.py path/to/file.jsonl
```

Keybindings are documented in the [Components](Components#jsonl-browser)
section.

## Start the LanguageTool service

The bundled Compose file brings up a load-balanced LanguageTool
instance behind Traefik, with fasttext language detection baked in.

```bash
cd languagetool
docker compose up -d --build
```

The service answers on `http://languagetool.loc` once Traefik routes
are active; the container exposes the standard LanguageTool HTTP API on
port 8010 internally.
