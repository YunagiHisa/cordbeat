"""Shared utilities for platform adapters."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _read_soul_keywords() -> list[str]:
    """Return the AI name from soul.yaml as a single-element keyword list.

    Reads ``~/.cordbeat/soul/soul.yaml`` (or the path configured via
    ``CORDBEAT_HOME``) to find ``identity.name``.  Returns an empty list if
    the file is missing or cannot be parsed.
    """
    try:
        import yaml

        from cordbeat.config import cordbeat_home

        soul_yaml = cordbeat_home() / "soul" / "soul.yaml"
        if not soul_yaml.is_file():
            return []
        with soul_yaml.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        name: str = data.get("identity", {}).get("name", "")
        return [name] if name else []
    except Exception:  # noqa: BLE001
        logger.debug("Could not read soul name for keyword filter", exc_info=True)
        return []
