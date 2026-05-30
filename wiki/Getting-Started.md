# Getting started

DataCurator targets Python 3.13 and is managed with `uv`. All commands
are expected to run through `uv run` so the project venv is used. The
repo exposes a Hydra-driven pipeline runner (`run_pipeline.py`) plus
individual components (a Google Translate client, a ntfy notifier, a
JSONL browser, a prompt template, and a LanguageTool Docker service) —
the sections below cover the pieces that are runnable today. See
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

Secrets are read from a `.env` file at the repo root. The notifier
config references two variables via OmegaConf's `oc.env` resolver:

```env
NTFY_URL=https://ntfy.example.com
NTFY_TOKEN=tk_xxxxxxxxxxxxxxxxxxxx
```

Both are optional — `oc.env` falls back to `null` when the variable is
unset. See [Configuration](Configuration) for the full list.

## Run the pipeline

`run_pipeline.py` is the Hydra entrypoint. The bundled `example`
experiment runs two regex-fix stages over `data/example_input.jsonl` and
writes one JSONL file per stage under `dataset/example/`:

```bash
uv run python run_pipeline.py
```

The flow is config-driven, so you can change it from the command line.
Resume from a stage, disable a stage, or enable notifications:

```bash
uv run python run_pipeline.py runner.start_from=punctuation
uv run python run_pipeline.py pipeline.whitespace.enabled=false
uv run python run_pipeline.py notifications=ntfy notifications.topic=my-topic
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
