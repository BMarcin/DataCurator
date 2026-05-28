# Components

This page documents the modules that currently live in the repo and
how they fit together. The pipeline orchestration that strings them
into stages is not yet in this tree — what is here are the building
blocks. Each section maps directly to a file or directory and only
describes behavior that is visible in the source. For the config keys
each component reads, see [Configuration](Configuration).

## GoogleTranslator

An async client around the public Google Translate web endpoint
(`clients5.google.com/translate_a/t`). It is intended to run as a
pre-pass that produces a reference translation alongside the original
target, so downstream LLM review can compare forms.

`GoogleTranslator` is constructed with a source/target language pair
and three HTTP knobs:

- `retries` — tenacity attempt budget for `translate_with_retries`.
- `timeout` — per-request timeout, applied via `httpx.Timeout`.
- `concurrency` — both `max_connections` and `max_keepalive_connections`
  on the underlying `httpx.AsyncClient` built by `_client()`.

`translate(client, text)` issues a single request and normalises the
response, which Google returns in three different shapes (flat string,
list of segments, or list of `[segment, source]` pairs).
`translate_with_retries(client, text)` wraps that call in an
exponential-backoff retry loop and returns `None` after the budget is
exhausted, logging the failure via loguru. Empty input short-circuits
to an empty string.

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

`prompts/improve-translations.j2` is the Jinja2 template used to ask
the LLM to review and rewrite a translation. It instructs the model to
act as a {source_language}-to-{target_language} reviewer following the
MQM framework: classify issues by category and severity, then produce
three rewrites — a conservative `improved_primary`, a polished
`improved_polished`, and a meaningfully different `improved_alternative`.

The template expects the following variables:

- `source_language`, `target_language` — language names used throughout
  the instructions.
- `source_segment`, `target_segment` — the texts under review.
- `google_translation` — the reference translation produced by the
  Google Translate pre-pass (see [GoogleTranslator](#googletranslator)).
- `target_language_tool_fixes` — an optional list of LanguageTool
  findings; the LanguageTool-specific section of the prompt is only
  rendered when this is truthy.

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
docker compose up -d --build
```

The Traefik router exposes the service at `http://languagetool.loc`;
the upstream container listens on port 8010 and is health-checked
against `/v2/languages`. Tuning is environment-driven inside the
Compose file — notable settings include `langtool_maxCheckThreads`,
`langtool_pipelineCaching`, `langtool_cacheSize`, and `Java_Xmx`.
