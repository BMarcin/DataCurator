# DataCurator

DataCurator is being assembled as a multi-stage, LLM-powered pipeline
for cleaning, post-editing, and quality-assessing parallel translation
datasets. It composes rule-based fixers (LanguageTool), reference
machine-translation (Google Translate), and structured-output LLM calls
against any OpenAI-compatible endpoint, with run-time wiring handled by
Hydra. Intermediate stage outputs are persisted as JSONL so any stage
can be re-run in isolation. This wiki documents the components that
currently exist in the repository — the stage/modifier framework and a
Hydra-driven runner that flows a dataset through an ordered list of
stages are in place; the LLM and translation stages that will plug into
them are still being ported in.

## What lives in the repo today

- A generic stage/modifier framework with a shared variable context
  (see [Components](Components#stages-and-modifiers)).
- A Hydra-driven pipeline runner with per-stage JSONL persistence and
  resume (see [Components](Components#pipeline-runner)).
- A generic LLM stage for OpenAI-compatible/vLLM endpoints with
  structured output (see [Components](Components#llm-stage)).
- Stage modifiers: Google Translate and LanguageTool field pre-passes
  plus field-rename, constant-injection and candidate-list helpers
  (see [Components](Components#stage-modifiers)).
- A Google Translate async client used as a pre-pass translator
  (see [Components](Components#googletranslator)).
- A ntfy.sh notifier driven by Hydra config
  (see [Components](Components#notifications)).
- A best-effort job reporter that publishes live status, metrics and
  artifacts to an S3 dashboard backend
  (see [Components](Components#job-reporting)).
- A Textual TUI for browsing JSONL files
  (see [Components](Components#jsonl-browser)).
- Jinja2 prompt templates for the LLM review step and the final
  candidate-selection step (see
  [Components](Components#improve-translations-prompt) and
  [Components](Components#pick-translations-prompt)).
- A LanguageTool service packaged with Docker Compose and Traefik
  (see [Components](Components#languagetool-service)).
- Hydra configs for the runner, experiments, stages, the LLM and the
  notifier (see [Configuration](Configuration)).

## Where to start

- New here? Read [Getting started](Getting-Started) for the install and
  first-run walkthrough.
- Looking up a config key or env var? Go to
  [Configuration](Configuration).
- Want details on a specific module? Open
  [Components](Components).
