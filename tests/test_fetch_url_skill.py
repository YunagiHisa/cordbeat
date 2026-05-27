"""Tests for the fetch_url skill."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Load skills/fetch_url/main.py under a unique module name so it doesn't
# clash with other skills' main.py (e.g. api_call) in sys.modules.
_SKILL_DIR = Path(__file__).parent.parent / "skills" / "fetch_url"
_spec = importlib.util.spec_from_file_location(
    "_fetch_url_skill_main", _SKILL_DIR / "main.py"
)
assert _spec is not None and _spec.loader is not None
fetch_url = importlib.util.module_from_spec(_spec)
sys.modules["_fetch_url_skill_main"] = fetch_url
_spec.loader.exec_module(fetch_url)


pytestmark = pytest.mark.asyncio


def _fake_getaddrinfo(ip: str) -> Any:
    import socket

    if ":" in ip:
        return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", (ip, 0, 0, 0))]
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]


class _FakeResp:
    def __init__(
        self,
        body: str,
        *,
        status: int = 200,
        content_type: str = "text/html; charset=utf-8",
        encoding: str = "utf-8",
    ) -> None:
        self.status_code = status
        self.headers = {"content-type": content_type}
        self.content = body.encode(encoding)
        self.encoding = encoding


class _FakeClient:
    def __init__(self, response: _FakeResp) -> None:
        self._response = response

    def __call__(self, *_a: Any, **_kw: Any) -> _FakeClient:
        return self

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_a: Any) -> None:
        return None

    async def get(self, *_a: Any, **_kw: Any) -> _FakeResp:
        return self._response


async def test_rejects_empty_url() -> None:
    result = await fetch_url.execute(url="")
    assert "error" in result


async def test_rejects_non_http_scheme() -> None:
    result = await fetch_url.execute(url="file:///etc/passwd")
    assert "error" in result
    assert "scheme" in result["error"].lower()


async def test_rejects_literal_loopback() -> None:
    result = await fetch_url.execute(url="http://127.0.0.1/")
    assert "error" in result
    assert "local/internal" in result["error"]


async def test_rejects_dns_to_private_ip() -> None:
    with patch(
        "_fetch_url_skill_main.socket.getaddrinfo",
        return_value=_fake_getaddrinfo("10.0.0.5"),
    ):
        result = await fetch_url.execute(url="http://evil.example.com/")
    assert "error" in result


async def test_rejects_cloud_metadata() -> None:
    result = await fetch_url.execute(url="http://169.254.169.254/latest/")
    assert "error" in result


async def test_strips_html_tags() -> None:
    body = (
        "<html><head><style>body{color:red}</style>"
        "<script>alert(1)</script></head>"
        "<body><h1>Title</h1><p>Hello&nbsp;<b>world</b>&#33;</p></body></html>"
    )
    fake = _FakeClient(_FakeResp(body))
    gai = _fake_getaddrinfo("93.184.216.34")
    with (
        patch("_fetch_url_skill_main.socket.getaddrinfo", return_value=gai),
        patch("_fetch_url_skill_main.httpx.AsyncClient", fake),
    ):
        result = await fetch_url.execute(url="http://example.com/article")
    assert result["status_code"] == 200
    assert "<" not in result["text"]
    assert "alert(1)" not in result["text"]
    assert "Title" in result["text"]
    assert "Hello world!" in result["text"]


async def test_truncates_to_max_length() -> None:
    body = "<html><body>" + ("x" * 20000) + "</body></html>"
    fake = _FakeClient(_FakeResp(body))
    gai = _fake_getaddrinfo("93.184.216.34")
    with (
        patch("_fetch_url_skill_main.socket.getaddrinfo", return_value=gai),
        patch("_fetch_url_skill_main.httpx.AsyncClient", fake),
    ):
        result = await fetch_url.execute(url="http://example.com/", max_length=500)
    assert result["truncated"] is True
    assert result["length"] == 500


async def test_plain_text_passthrough() -> None:
    body = "Hello plain text\nLine 2"
    fake = _FakeClient(_FakeResp(body, content_type="text/plain; charset=utf-8"))
    gai = _fake_getaddrinfo("93.184.216.34")
    with (
        patch("_fetch_url_skill_main.socket.getaddrinfo", return_value=gai),
        patch("_fetch_url_skill_main.httpx.AsyncClient", fake),
    ):
        result = await fetch_url.execute(url="http://example.com/raw")
    assert "Hello plain text" in result["text"]
    assert "Line 2" in result["text"]


async def test_returns_status_and_content_type() -> None:
    fake = _FakeClient(_FakeResp("<p>ok</p>", status=201))
    gai = _fake_getaddrinfo("93.184.216.34")
    with (
        patch("_fetch_url_skill_main.socket.getaddrinfo", return_value=gai),
        patch("_fetch_url_skill_main.httpx.AsyncClient", fake),
    ):
        result = await fetch_url.execute(url="http://example.com/")
    assert result["status_code"] == 201
    assert "text/html" in result["content_type"]
