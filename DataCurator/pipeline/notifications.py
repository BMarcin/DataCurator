"""Lightweight ntfy.sh notifier.

Uses stdlib ``urllib`` so this adds no extra dependency. Configure under
the ``notifications`` key of the pipeline config — see
``config/notifications/ntfy.yaml`` for the schema.

Notifications are best-effort: a failed POST logs a warning and the
pipeline keeps running. Network problems should never abort an
LLM batch that's halfway through.
"""
from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from an OmegaConf ``DictConfig`` or plain ``dict``, returning ``default`` if absent."""
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default


def notify(
    notifications_cfg: Any,
    title: str,
    message: str,
    priority: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Send an ntfy notification if notifications are enabled.

    ``notifications_cfg`` is the ``cfg.notifications`` sub-config (a
    :class:`DictConfig` or plain dict). Missing keys fall back to safe
    defaults.
    """
    if not _cfg_get(notifications_cfg, "enabled", False):
        return

    backend = _cfg_get(notifications_cfg, "backend", "ntfy")
    if backend != "ntfy":
        logger.warning(f"Unknown notifications backend: {backend!r}")
        return

    server = _cfg_get(notifications_cfg, "server")
    if not server:
        raise ValueError(
            "ntfy notifications enabled but `notifications.server` is unset; "
            "set it (e.g. via $NTFY_URL) or disable notifications"
        )
    server = str(server).rstrip("/")
    topic = _cfg_get(notifications_cfg, "topic")
    if not topic:
        logger.warning("ntfy notifications enabled but `notifications.topic` is unset")
        return

    eff_priority = priority or _cfg_get(notifications_cfg, "priority", "default")
    eff_tags = tags if tags is not None else _cfg_get(notifications_cfg, "tags", None)
    if isinstance(eff_tags, DictConfig):
        eff_tags = OmegaConf.to_container(eff_tags)

    headers = {
        "Title": title,
        "Priority": str(eff_priority),
    }
    if eff_tags:
        if isinstance(eff_tags, (list, tuple)):
            headers["Tags"] = ",".join(str(t) for t in eff_tags)
        else:
            headers["Tags"] = str(eff_tags)

    auth_token = _cfg_get(notifications_cfg, "auth_token")
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    url = f"{server}/{topic}"
    timeout = float(_cfg_get(notifications_cfg, "timeout", 10))
    request = urllib.request.Request(
        url,
        data=message.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning(f"ntfy notification failed ({url}): {e}")
