"""Tests for the centralized SSRF guard installed for network-enabled skills.

The guard lives in the subprocess runner and validates the *resolved*
destination IP at ``connect()`` time, so every ``network=true`` skill — not
just the hand-written ``fetch_url`` / ``api_call`` skills — is protected
against reaching private, loopback, link-local, or cloud-metadata services.
"""

from __future__ import annotations

import socket
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
