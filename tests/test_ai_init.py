"""Tests for the lazy ``__getattr__`` loader in ``cordbeat.ai``.

The package defers all submodule imports to avoid a circular-import cycle
(ai.extraction → agent.soul → … → ai.extraction). These tests pin that the
public names resolve to the right objects and that unknown names still raise
a normal ``AttributeError``.
"""

from __future__ import annotations

import pytest

import cordbeat.ai as ai


def test_all_mirrors_export_registry() -> None:
    # ``__all__`` is derived from the export table; keep them in lock-step.
    assert set(ai.__all__) == set(ai._EXPORTS)


def test_known_name_resolves_via_lazy_loader() -> None:
    # ``sanitize`` lives in ai.prompt, which does not pull in the
    # extraction→soul→engine import cycle, so it is safe to resolve eagerly.
    from cordbeat.ai.prompt import sanitize

    assert ai.sanitize is sanitize


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
        _ = ai.does_not_exist
