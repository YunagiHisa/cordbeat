"""Tests for the centralized SSRF guard installed for network-enabled skills.

The guard lives in the subprocess runner and validates the *resolved*
destination IP at ``connect()`` time, so every ``network=true`` skill — not
just the hand-written ``fetch_url`` / ``api_call`` skills — is protected
against reaching private, loopback, link-local, or cloud-metadata services.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from cordbeat.skills.runner import (
    SkillPermissionError,
    _install_ssrf_guard,
    _is_blocked_ip,
)


def _ip(value: str):  # type: ignore[no-untyped-def]
    import ipaddress

    return ipaddress.ip_address(value)


class TestIsBlockedIp:
    @pytest.mark.parametrize(
        "addr",
        [
            "127.0.0.1",  # loopback
            "10.0.0.5",  # private
            "192.168.1.1",  # private
            "172.16.0.1",  # private
            "169.254.169.254",  # AWS/GCP/Azure metadata
            "169.254.1.1",  # link-local
            "224.0.0.1",  # multicast
            "0.0.0.0",  # unspecified
            "fd00:ec2::254",  # IPv6 metadata
            "::1",  # IPv6 loopback
        ],
    )
    def test_internal_addresses_are_blocked(self, addr: str) -> None:
        assert _is_blocked_ip(_ip(addr)) is True

    @pytest.mark.parametrize("addr", ["1.1.1.1", "8.8.8.8", "93.184.216.34"])
    def test_public_addresses_are_allowed(self, addr: str) -> None:
        assert _is_blocked_ip(_ip(addr)) is False


class TestSsrfGuard:
    def test_blocks_connect_to_metadata_and_private_ips(self) -> None:
        saved_connect_ex = socket.socket.connect_ex
        try:
            # patch.object restores socket.socket.connect on exit; the guard
            # captures the (mocked) original so no real network is touched.
            with patch.object(socket.socket, "connect") as mock_connect:
                _install_ssrf_guard()

                # Public address is allowed through to the (mocked) connect.
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(("1.1.1.1", 443))
                s.close()
                mock_connect.assert_called_once()

                # Metadata and private addresses are rejected before connect.
                for blocked in ("169.254.169.254", "10.0.0.1", "127.0.0.1"):
                    bs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    with pytest.raises(SkillPermissionError, match="SSRF protection"):
                        bs.connect((blocked, 80))
                    bs.close()

                # mock_connect was only called for the allowed address.
                assert mock_connect.call_count == 1
        finally:
            socket.socket.connect_ex = saved_connect_ex  # type: ignore[method-assign]

    def test_blocks_hostname_resolving_to_internal_ip(self) -> None:
        saved_connect_ex = socket.socket.connect_ex
        try:
            with patch.object(socket.socket, "connect"):
                # A hostname passed straight to connect() is resolved and
                # rejected if it maps to an internal address (DNS-rebinding).
                fake_info = [(2, 1, 6, "", ("10.1.2.3", 80))]
                with patch(
                    "cordbeat.skills.runner._socket_module.getaddrinfo",
                    return_value=fake_info,
                ):
                    _install_ssrf_guard()
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    with pytest.raises(SkillPermissionError, match="SSRF protection"):
                        s.connect(("evil.internal.example", 80))
                    s.close()
        finally:
            socket.socket.connect_ex = saved_connect_ex  # type: ignore[method-assign]

    def test_non_inet_address_is_ignored(self) -> None:
        saved_connect_ex = socket.socket.connect_ex
        try:
            with patch.object(socket.socket, "connect") as mock_connect:
                _install_ssrf_guard()
                # A non-tuple address (e.g. an AF_UNIX path) bypasses the IP
                # check and is handed to the lower layer unchanged.
                s = MagicMock()
                socket.socket.connect(s, "/tmp/some.sock")  # type: ignore[arg-type]
                mock_connect.assert_called_once()
        finally:
            socket.socket.connect_ex = saved_connect_ex  # type: ignore[method-assign]


_REPO_SKILLS_DIR = Path(__file__).parent.parent / "skills"


def _load_builtin_skill_module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"test_ssrf_preflight_{name}",
        str(_REPO_SKILLS_DIR / name / "main.py"),
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFixedDomainPreflight:
    """web_search / weather resolve their fixed domain before connecting."""

    @pytest.mark.parametrize("skill_name", ["web_search", "weather"])
    def test_blocks_internal_resolution(
        self,
        skill_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mod = _load_builtin_skill_module(skill_name)

        def fake_getaddrinfo(host: str, port: Any, **_kwargs: Any) -> list[Any]:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]

        monkeypatch.setattr(mod.socket, "getaddrinfo", fake_getaddrinfo)
        assert mod._preflight_blocked("example.com") is not None

    @pytest.mark.parametrize("skill_name", ["web_search", "weather"])
    def test_allows_public_resolution(
        self,
        skill_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mod = _load_builtin_skill_module(skill_name)

        def fake_getaddrinfo(host: str, port: Any, **_kwargs: Any) -> list[Any]:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

        monkeypatch.setattr(mod.socket, "getaddrinfo", fake_getaddrinfo)
        assert mod._preflight_blocked("example.com") is None

    @pytest.mark.parametrize("skill_name", ["web_search", "weather"])
    def test_fails_open_on_resolution_error(
        self,
        skill_name: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Resolution failure is not blocking; the request fails on its own."""
        mod = _load_builtin_skill_module(skill_name)

        def fake_getaddrinfo(host: str, port: Any, **_kwargs: Any) -> list[Any]:
            raise socket.gaierror("resolution failed")

        monkeypatch.setattr(mod.socket, "getaddrinfo", fake_getaddrinfo)
        assert mod._preflight_blocked("example.com") is None

    async def test_web_search_execute_short_circuits_on_internal_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mod = _load_builtin_skill_module("web_search")

        def fake_getaddrinfo(host: str, port: Any, **_kwargs: Any) -> list[Any]:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 443))]

        monkeypatch.setattr(mod.socket, "getaddrinfo", fake_getaddrinfo)
        result = await mod.execute(query="hello")

        assert "internal address" in result["error"]
        assert result["results"] == []

    async def test_weather_execute_short_circuits_on_internal_resolution(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mod = _load_builtin_skill_module("weather")

        def fake_getaddrinfo(host: str, port: Any, **_kwargs: Any) -> list[Any]:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", 443))]

        monkeypatch.setattr(mod.socket, "getaddrinfo", fake_getaddrinfo)
        result = await mod.execute(location="Tokyo")

        assert "internal address" in result["error"]


class TestRunSkillGuardOrdering:
    def test_fs_guard_roots_derive_from_trimmed_sys_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """sys.path is trimmed BEFORE the fs-guard read roots are captured.

        Otherwise entries like the runner's own package directory or
        PYTHONPATH injections become readable roots for filesystem=false
        skills.
        """
        import asyncio

        from cordbeat.skills import runner

        skill_dir = tmp_path / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "main.py").write_text(
            "def execute(**kwargs):\n    return {'ok': True}\n",
            encoding="utf-8",
        )
        injected = tmp_path / "pythonpath_injected"
        injected.mkdir()
        monkeypatch.syspath_prepend(str(injected))

        calls: list[str] = []
        captured: dict[str, tuple[Path, ...]] = {}

        def fake_restrict(skill_dir_arg: Path, work_dir_arg: Path) -> None:
            calls.append("restrict")
            sys.path.remove(str(injected))

        def fake_fs_guard(
            work_dir: Path,
            *,
            allowed_read_roots: tuple[Path, ...] = (),
        ) -> None:
            calls.append("fs_guard")
            captured["roots"] = allowed_read_roots

        monkeypatch.setattr(runner, "_restrict_sys_and_env", fake_restrict)
        monkeypatch.setattr(runner, "_install_fs_guard", fake_fs_guard)
        monkeypatch.setattr(runner, "_install_network_guard", lambda: None)
        monkeypatch.setattr(
            runner, "_apply_resource_limits", lambda **_kwargs: None
        )

        loop = asyncio.new_event_loop()
        try:
            result = runner._run_skill(
                skill_dir,
                "demo",
                {},
                False,
                {
                    "work_dir": str(tmp_path / "work"),
                    "network": False,
                    "filesystem": False,
                },
                loop,
            )
        finally:
            loop.close()

        assert result == {"ok": True}
        assert calls.index("restrict") < calls.index("fs_guard")
        assert Path(str(injected)) not in captured["roots"]
