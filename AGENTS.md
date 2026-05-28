# Setup
- Python 3.13+, managed by `uv` (do NOT use pip directly)
- `uv sync` installs everything from `pyproject.toml` and `uv.lock`
- Always run commands via `uv run <cmd>` so the right venv is used
- It is a Github Repository which has CI CD enabled

# Conventions
- Type hints required on every function signature
- Each function needs to have a docstring
- Use `pathlib.Path`, not `os.path`
- Everything must be configured by using `hydra` configs. Each step of a pipeline must be easy to enable/disable or change the input values.
- LLM prompts are jinja2 templates

# Wiki
The wiki source of truth lives in `wiki/` in this repo. A CI job mirrors it to the Github Wiki tab. Each time you add/upate the logic this needs to be reflected in the wiki.
You are responsible for structuring the wiki in a good way.

## Basic structure
- `wiki/Home.md` - landing page, links to all other pages, short introduction to the project. What is the purpose and what the whole project does.
- `wiki/Getting-Started.md` - install + first-run walkthrough
- `wiki/Configuration.md` - env vars, config files, defaults
- `wiki/_Sidebar.md` - navigation (manual order, do not auto-sort)
- `wiki/_Footer.md` - appears on every page.

## Grounding rule
Every claim in the wiki MUST be traceable to something in codebase:
- a function
- a config key
- a test
- a CLI flag
- a docstring
- a model
If you can't point to the source, do NOT write it. Prefer omissions over invention.

## What goes to the wiki
- Conceptual overviews (what this project does and why)
- How modules fit together at a 1-paragraph level
- How the pipelines part works and how they fit together
- Configuration references
- Common workflows
- Functions/classes

## What does NOT go in the wiki
- File paths beyond 1-2 well-known entry points (paths drift, then the
  wiki lies — agents and humans both end up looking in the wrong place)
- Line numbers, ever
- TODO lists, roadmaps, or anything time-sensitive
- Copies of code that already exists in the repo — link instead
- Anything you'd write differently next week

## Style
- Sentence case in headings ("Getting started", not "Getting Started")
- 80–120 words per section before a code block or list
- Code blocks must be runnable as shown — no "..." placeholders
- Use the project's actual command names (e.g. `uv run pytest`,
  not generic `pytest`) — match the rest of AGENTS.md
- Audience: a competent developer new to this project, not a beginner
  programmer. Don't explain what a function is.

## Update strategy
- Each time you add something to the repository or make any changes, you need to reflect them in the wiki
- Do not regenerate pages that are not connected with the update in the code you are making

