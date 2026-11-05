"""Prompt sources for LLM stages: inline text or a Jinja2 file on disk.

Each message of a stage's ``prompt`` list carries exactly one of

* ``text`` — the Jinja2 template itself, written in the experiment config; or
* ``template`` — a path to a ``.j2`` file (resolved against the working dir).

Prompt files are experiment assets, not code: they live outside git (by
convention under ``assets/prompts/<experiment>/``). Before a stage is
instantiated, signed and — in distributed mode — shipped, every ``template``
is replaced by the file's ``text`` (:func:`inline_prompt_files`). The stage
signature therefore covers the prompt text, so editing a prompt invalidates
resume while moving the file does not, and workers receive the text inside the
stage spec instead of reading files of their own.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping


def load_message(message: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``message`` with its template text under ``text`` and its origin under ``source``.

    ``source`` is the file path for a ``template`` entry and ``None`` for an
    inline ``text`` entry. Raises ``ValueError`` unless exactly one of the two
    keys is set, and ``FileNotFoundError`` when the template file is missing.
    """
    has_template, has_text = "template" in message, "text" in message
    if has_template == has_text:
        raise ValueError(
            f"prompt message {dict(message)!r} must set exactly one of `template` "
            "(path to a .j2 file) or `text` (inline template)"
        )
    rest = {k: v for k, v in message.items() if k not in ("template", "text", "source")}
    if has_text:
        return {**rest, "text": str(message["text"]), "source": message.get("source")}
    path = Path(str(message["template"]))
    if not path.is_file():
        raise FileNotFoundError(
            f"prompt template {str(path)!r} not found (relative to {Path.cwd()}); "
            "prompt files are experiment assets kept outside git"
        )
    return {**rest, "text": path.read_text(encoding="utf-8"), "source": str(path)}


def _map_prompts(node: Any, fn: Callable[[Mapping[str, Any]], Any]) -> Any:
    """Copy a plain config container, applying ``fn`` to every ``prompt`` list entry."""
    if isinstance(node, Mapping):
        out: Dict[str, Any] = {}
        for key, value in node.items():
            if key == "prompt" and isinstance(value, list):
                out[key] = [fn(m) if isinstance(m, Mapping) else m for m in value]
            else:
                out[key] = _map_prompts(value, fn)
        return out
    if isinstance(node, list):
        return [_map_prompts(v, fn) for v in node]
    return node


def inline_prompt_files(node: Any) -> Any:
    """Replace every ``template`` path in ``node``'s prompts with the file's ``text``.

    ``node`` is a resolved plain container (``OmegaConf.to_container``). The
    file path is dropped from the result so the stage signature depends on the
    prompt text alone; :func:`prompt_snapshot` keeps it for the record.
    """
    def inline(message: Mapping[str, Any]) -> Dict[str, Any]:
        """Swap one message's template path for its text."""
        loaded = load_message(message)
        loaded.pop("source")
        return loaded

    return _map_prompts(node, inline)


def prompt_snapshot(node: Any) -> List[Dict[str, Any]]:
    """List ``{role, source, text}`` for every prompt message in ``node``, in order."""
    found: List[Dict[str, Any]] = []

    def collect(message: Mapping[str, Any]) -> Mapping[str, Any]:
        """Record one message's role, origin and text."""
        loaded = load_message(message)
        found.append({"role": loaded.get("role"), "source": loaded["source"], "text": loaded["text"]})
        return message

    _map_prompts(node, collect)
    return found
