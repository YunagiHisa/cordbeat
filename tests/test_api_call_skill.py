"""Tests for SSRF protections in the api_call skill."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# The skill module lives in skills/api_call/main.py which isn't a Python
# package. Load it by path.
_SKILLS_DIR = Path(__file__).parent.parent / "skills" / "api_call"
sys.path.insert(0, str(_SKILLS_DIR))
import main as api_call  # type: ignore[import-not-found]  # noqa: E402

sys.path.pop(0)


pytestmark = pytest.mark.asyncio


def _fake_getaddrinfo(ip: str) -> Any:
    """Return a minimal getaddrinfo-style tuple list resolving to *ip*."""
    import socket

    if ":" in ip:
        return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", (ip, 0, 0, 0))]
    return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]


async def test_rejects_bad_method() -> None:
    result = await api_call.execute(url="https://example.com/", method="DELETE")
    assert "error" in result
    assert "Unsupported method" in result["error"]


async def test_rejects_non_http_scheme() -> None:
    result = await api_call.execute(url="file:///etc/passwd")
    assert "error" in result
    assert "scheme" in result["error"].lower()


async def test_rejects_ftp_scheme() -> None:
    result = await api_call.execute(url="ftp://example.com/")
    assert "error" in result


async def test_rejects_literal_loopback() -> None:
    result = await api_call.execute(url="http://127.0.0.1/")
    assert "error" in result
    assert "local/internal" in result["error"]


async def test_rejects_literal_private_ipv4() -> None:
    for ip in ["10.0.0.1", "192.168.1.1", "172.16.0.1"]:
        result = await api_call.execute(url=f"http://{ip}/")
        assert "error" in result, f"failed to block {ip}"
        assert "local/internal" in result["error"]


async def test_rejects_literal_ipv6_loopback() -> None:
    result = await api_call.execute(url="http://[::1]/")
    assert "error" in result
    assert "local/internal" in result["error"]


async def test_rejects_ipv6_link_local() -> None:
    result = await api_call.execute(url="http://[fe80::1]/")
    assert "error" in result
    assert "local/internal" in result["error"]


async def test_rejects_cloud_metadata() -> None:
    result = await api_call.execute(url="http://169.254.169.254/latest/meta-data/")
    assert "error" in result
    assert "local/internal" in result["error"]


async def test_rejects_localhost_by_name() -> None:
    result = await api_call.execute(url="http://localhost/")
    assert "error" in result


async def test_rejects_dns_resolving_to_private_ip() -> None:
    """A hostname that resolves to a private IP must be blocked."""
    with patch("main.socket.getaddrinfo", return_value=_fake_getaddrinfo("10.1.2.3")):
        result = await api_call.execute(url="http://evil.example.com/")
    assert "error" in result
    assert "local/internal" in result["error"]


async def test_rejects_dns_resolving_to_metadata_ipv6() -> None:
    with patch(
        "main.socket.getaddrinfo",
        return_value=_fake_getaddrinfo("fd00:ec2::254"),
    ):
        result = await api_call.execute(url="http://evil.example.com/")
    assert "error" in result


async def test_unresolvable_host() -> None:
    import socket as _socket

    with patch(
        "main.socket.getaddrinfo",
        side_effect=_socket.gaierror("no such host"),
    ):
        result = await api_call.execute(url="http://nonexistent.invalid/")
    assert "error" in result
    assert "resolve" in result["error"].lower()


async def test_accepts_public_ip() -> None:
    """A public IP should *not* be blocked (network call is mocked)."""

    class FakeResp:
        status_code = 200
        headers: dict[str, str] = {}
        text = '{"ok": true}'

        def json(self) -> Any:
            return {"ok": True}

    class FakeClient:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def get(self, *_a: Any, **_kw: Any) -> FakeResp:
            return FakeResp()

        async def post(self, *_a: Any, **_kw: Any) -> FakeResp:
            return FakeResp()

    gai = _fake_getaddrinfo("93.184.216.34")
    with (
        patch("main.socket.getaddrinfo", return_value=gai),
        patch("main.httpx.AsyncClient", FakeClient),
    ):
        result = await api_call.execute(url="http://example.com/api")
    assert result.get("status_code") == 200
