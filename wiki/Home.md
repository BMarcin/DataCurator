# DataCurator

DataCurator is being assembled as a multi-stage, LLM-powered pipeline
for cleaning, post-editing, and quality-assessing parallel translation
datasets. It composes rule-based fixers (LanguageTool), reference
machine-translation (Google Translate), and structured-output LLM calls
against any OpenAI-compatible endpoint, with run-time wiring handled by
Hydra. Intermediate stage outputs are persisted as JSONL so any stage
can be re-run in isolation. This wiki documents the components that
currently exist in the repository — the orchestration layer that ties
them into a single end-to-end pipeline is still being ported in.

## What lives in the repo today

- A Google Translate async client used as a pre-pass translator
  (see [Components](Components#googletranslator)).
- A ntfy.sh notifier driven by Hydra config
  (see [Components](Components#notifications)).
- A Textual TUI for browsing JSONL files
  (see [Components](Components#jsonl-browser)).
- A Jinja2 prompt template for the LLM review step
  (see [Components](Components#improve-translations-prompt)).
- A LanguageTool service packaged with Docker Compose and Traefik
  (see [Components](Components#languagetool-service)).
- Hydra configs for the LLM and the notifier
  (see [Configuration](Configuration)).

## Where to start

- New here? Read [Getting started](Getting-Started) for the install and
  first-run walkthrough.
- Looking up a config key or env var? Go to
  [Configuration](Configuration).
- Want details on a specific module? Open
  [Components](Components).
