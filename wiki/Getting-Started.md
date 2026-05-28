# Getting started

DataCurator targets Python 3.13 and is managed with `uv`. All commands
are expected to run through `uv run` so the project venv is used. The
repo currently exposes individual components (a Google Translate
client, a ntfy notifier, a JSONL browser, a prompt template, and a
LanguageTool Docker service) rather than a single CLI — the sections
below cover the pieces that are runnable today. See
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
