"""Tests for CLI adapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from cordbeat.adapters.cli import (
    _build_cli_completion_tree,
    _mark_cli_proposal_action_sent,
    main,
)


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
