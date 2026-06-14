"""Tests for the inspect_image skill."""

from __future__ import annotations

import importlib.util
import io
import socket
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

_SKILL_DIR = Path(__file__).parent.parent / "skills" / "inspect_image"
_spec = importlib.util.spec_from_file_location(
    "_inspect_image_skill_main", _SKILL_DIR / "main.py"
)
assert _spec is not None and _spec.loader is not None
inspect_image = importlib.util.module_from_spec(_spec)
sys.modules["_inspect_image_skill_main"] = inspect_image
_spec.loader.exec_module(inspect_image)

pytestmark = pytest.mark.asyncio


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, content: bytes) -> None:
        self.response = _Response(content)

    def __call__(self, *_args: Any, **_kwargs: Any) -> _Client:
        return self

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def get(self, *_args: Any, **_kwargs: Any) -> _Response:
        return self.response


async def test_rejects_private_url() -> None:
    result = await inspect_image.execute(url="http://127.0.0.1/image.png")
    assert "error" in result


async def test_fetches_and_normalizes_image() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), "red").save(buf, format="PNG")
    gai = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
    with (
        patch("_inspect_image_skill_main.socket.getaddrinfo", return_value=gai),
        patch("_inspect_image_skill_main.httpx.AsyncClient", _Client(buf.getvalue())),
    ):
        result = await inspect_image.execute(url="https://example.com/image.png")

    assert result["mime_type"] == "image/jpeg"
    assert result["width"] == 32
    assert result["height"] == 24
    assert result["image_base64"]
