# DataCurator

DataCurator is a multi-stage, LLM-powered pipeline for cleaning, post-editing
and quality-assessing parallel translation datasets. It chains together
rule-based fixers (LanguageTool), reference machine-translation
(Google Translate), and structured-output LLM calls (any
OpenAI-compatible endpoint — vLLM, llama.cpp, OpenAI, etc.) into a
reproducible, Hydra-driven workflow.

The design goal is a **zero-drop** workflow: instead of throwing away
weak samples, every stage tries to *improve* them, and every intermediate
output is persisted to disk as JSONL so you can audit, branch from, or
re-run any single stage in isolation.

## Auxiliary tools

### `helpers/browse_jsonl.py` — TUI sample browser

A [Textual](https://textual.textualize.io/) terminal app for reviewing
the JSONL output of any pipeline stage sample-by-sample. Each top-level
field is shown as a collapsible section.

```bash
uv run python browse_jsonl.py dataset/<experiment>/02_autolqa.jsonl
```

Keybindings:

| Key                        | Action                            |
|----------------------------|-----------------------------------|
| `n` / `→` / `space`        | Next sample                       |
| `p` / `←`                  | Previous sample                   |
| `home` / `end`             | Jump to first / last sample       |
| `g`                        | Jump to a specific 1-based index  |
| `c`                        | Collapse / expand all sections    |
| `s`                        | Pick which top-level columns show |
| `ctrl+r`                   | Reload the file from disk         |
| `q`                        | Quit                              |