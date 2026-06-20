"""Tests for CLI adapter."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import websockets

from cordbeat.adapters.cli import (
    _build_cli_completion_tree,
    _mark_cli_proposal_action_sent,
    cli_main,
    main,
)


def _aiter_over(items: list[str]) -> AsyncIterator[str]:
    """Build an async iterator yielding *items*, for mocking ``async for``."""

    async def _gen() -> AsyncIterator[str]:
        for item in items:
            yield item

    return _gen()


def test_build_cli_completion_tree_includes_pending_ids() -> None:
    tree = _build_cli_completion_tree(["abc-123"])
    assert "/approve" in tree
    assert "/reject" in tree
    assert tree["/approve"] == {"abc-123": None}
    assert tree["/reject"] == {"abc-123": None}
    assert tree["/proposals"] is None


def test_mark_cli_proposal_action_sent_removes_pending_id() -> None:
    pending = {"abc-123": {"skill_name": "draw"}}
    _mark_cli_proposal_action_sent("/approve abc-123", pending)
    assert pending == {}


class TestCLIAdapter:
    async def test_main_connects_and_sends(self) -> None:
        """CLI adapter connects, completes handshake, then exits on EOF."""
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"content": "Welcome"}))
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "cordbeat.adapters.cli.websockets.connect", return_value=mock_connect
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            await main("ws://localhost:8765")

        # Verify handshake was sent
        mock_ws.send.assert_awaited_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent == {"adapter_id": "cli"}

    async def test_main_sends_message(self) -> None:
        """CLI adapter sends user input as message."""
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"content": "Welcome"}))
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        call_count = 0

        def fake_input(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Hello CordBeat"
            raise EOFError

        with (
            patch(
                "cordbeat.adapters.cli.websockets.connect", return_value=mock_connect
            ),
            patch("builtins.input", side_effect=fake_input),
        ):
            await main()

        # Handshake + 1 message
        assert mock_ws.send.await_count == 2
        msg = json.loads(mock_ws.send.call_args_list[1][0][0])
        assert msg["type"] == "message"
        assert msg["content"] == "Hello CordBeat"
        assert msg["adapter_id"] == "cli"
        assert msg["platform_user_id"] == "cli_user"

    async def test_main_skips_blank_input(self) -> None:
        """Blank lines are not sent."""
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"content": "OK"}))
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        call_count = 0

        def fake_input(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ""
            if call_count == 2:
                return "   "
            raise EOFError

        with (
            patch(
                "cordbeat.adapters.cli.websockets.connect", return_value=mock_connect
            ),
            patch("builtins.input", side_effect=fake_input),
        ):
            await main()

        # Only handshake was sent, no messages
        assert mock_ws.send.await_count == 1

    async def test_main_keyboard_interrupt(self) -> None:
        """Ctrl+C gracefully exits."""
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"content": "OK"}))
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "cordbeat.adapters.cli.websockets.connect", return_value=mock_connect
            ),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            await main()

        # Should not raise — graceful shutdown
        mock_ws.send.assert_awaited_once()  # handshake only

    async def test_main_handshake_includes_auth_token(self) -> None:
        """When an auth token is supplied it is included in the handshake."""
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"content": "OK"}))
        mock_ws.__aiter__ = MagicMock(return_value=_aiter_over([]))

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "cordbeat.adapters.cli.websockets.connect", return_value=mock_connect
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            await main("ws://localhost:8765", auth_token="secret-token")

        handshake = json.loads(mock_ws.send.call_args_list[0][0][0])
        assert handshake == {"adapter_id": "cli", "auth_token": "secret-token"}

    async def test_listener_renders_all_message_types(self, capsys: Any) -> None:
        """The listener prints errors, skill confirmations, messages, images."""
        png = base64.b64encode(b"fake-png-bytes").decode("ascii")
        incoming = [
            json.dumps({"type": "error", "content": "boom"}),
            json.dumps(
                {
                    "type": "skill_confirm",
                    "metadata": {
                        "skill_name": "draw",
                        "skill_params": {"prompt": "a cat"},
                        "proposal_id": "prop-1",
                    },
                }
            ),
            json.dumps({"type": "message", "content": "hi there"}),
            json.dumps({"type": "weird", "content": "?"}),
            json.dumps(
                {"type": "message", "content": "with image", "images": [png, "!!bad"]}
            ),
        ]
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"content": "Welcome"}))
        mock_ws.__aiter__ = MagicMock(return_value=_aiter_over(incoming))

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "cordbeat.adapters.cli.websockets.connect", return_value=mock_connect
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            await main()

        out = capsys.readouterr().out
        assert "[error] boom" in out
        assert "Skill Execution Required" in out
        assert "Bot: hi there" in out
        assert "[weird] ?" in out
        assert "Image 1" in out  # valid base64 decoded
        assert "decode error" in out  # invalid base64 branch

    async def test_listener_handles_connection_closed(self, capsys: Any) -> None:
        """A dropped connection prints a disconnect notice without raising."""

        async def _raise_closed() -> AsyncIterator[str]:
            if False:  # pragma: no cover - establishes async-generator type
                yield ""
            raise websockets.ConnectionClosed(None, None)

        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=json.dumps({"content": "OK"}))
        mock_ws.__aiter__ = MagicMock(return_value=_raise_closed())

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "cordbeat.adapters.cli.websockets.connect", return_value=mock_connect
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            await main()

        assert "Disconnected from Core." in capsys.readouterr().out


def test_cli_main_resolves_config_and_runs() -> None:
    """cli_main loads config, builds the ws URL, and drives main()."""
    fake_config = MagicMock()
    fake_config.gateway.host = "example.test"
    fake_config.gateway.port = 9999
    fake_config.gateway.auth_token = "tok"

    with (
        patch("cordbeat.config.load_config", return_value=fake_config) as load_cfg,
        patch("cordbeat.adapters.cli.asyncio.run") as run_mock,
    ):
        cli_main(config_path="/tmp/cfg.yaml")

    load_cfg.assert_called_once_with("/tmp/cfg.yaml")
    run_mock.assert_called_once()
    # The coroutine passed to asyncio.run targets ws://<host>:<port>
    coro = run_mock.call_args[0][0]
    assert coro.cr_frame.f_locals["ws_url"] == "ws://example.test:9999"
    coro.close()


def test_cli_main_swallows_keyboard_interrupt() -> None:
    """A Ctrl+C during the session is swallowed by cli_main."""
    fake_config = MagicMock()
    fake_config.gateway.host = "h"
    fake_config.gateway.port = 1
    fake_config.gateway.auth_token = ""

    with (
        patch("cordbeat.config.load_config", return_value=fake_config),
        patch(
            "cordbeat.adapters.cli.asyncio.run", side_effect=KeyboardInterrupt
        ) as run_mock,
    ):
        cli_main(config_path="/tmp/cfg.yaml")  # must not raise

    # Close the un-awaited coroutine handed to the patched asyncio.run.
    run_mock.call_args[0][0].close()
