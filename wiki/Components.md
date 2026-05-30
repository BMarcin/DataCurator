# Components

This page documents the modules that currently live in the repo and
how they fit together. The stage/modifier framework that the other
components plug into now exists; the concrete stage implementations
(LLM, regex, translation) and the runner that orders them are still
being ported in. Each section maps directly to a file or directory and
only describes behavior that is visible in the source. For the config
keys each component reads, see [Configuration](Configuration).

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
`PipelineRunner.run` then feeds the dataset through them. Stage `N` reads
the output of stage `N-1` — or the dataset input for the first stage —
and writes its own output to `<output_dir>/<NN>_<stage>.jsonl`. Records
are processed concurrently up to `runner.concurrency` and each result is
flushed to disk as soon as it is produced, so an interrupted run never
loses completed work.

The runner implements the pipeline-design rules directly. Resume is
driven by a sidecar `<NN>_<stage>.meta.json` holding a signature hashed
from the stage's resolved config: when the signature is unchanged and
`runner.resume` is set, records already present in the output (matched by
`runner.id_field`) are skipped; when the config or logic changes the
signature differs and the stage is recomputed from scratch. Stages with
`enabled: false` are passed through, and `runner.start_from` resumes at a
named stage by reading the previous stage's output. When two consecutive
stages declare different `model` names the runner logs a warning, sends an
urgent notification, and — unless `runner.pause_on_model_switch` is false
or no TTY is attached — blocks for operator confirmation so the deployed
GPU model can be swapped. Notifications fire through
[`notify`](#notifications) at the points enabled in the notifications
config.

The Hydra entrypoint is `run_pipeline.py` at the repo root:

```bash
uv run python run_pipeline.py
```

## Regex-fix stage

`DataCurator.pipeline.stages.regex_fix` is the in-tree example stage. Both
`RegexFixStage` and `RegexFixModifier` apply an ordered list of
`{pattern, replace}` rules to a text field; the stage rewrites the field
in place, while the modifier (a `FieldModifier`) can write the result to a
separate `target` field and attach to any other stage as a pre-pass. Rules
are plain mappings so they can be declared directly in YAML, which makes
this a self-contained, deterministic illustration of how a stage and its
modifiers are wired up through config.

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
