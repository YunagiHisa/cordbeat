"""Tests for CLI adapter."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from cordbeat.cli_adapter import main


class TestCLIAdapter:
    async def test_main_connects_and_sends(self) -> None:
        """CLI adapter connects, completes handshake, then exits on EOF."""
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(
            return_value=json.dumps({"content": "Welcome"})
        )
        mock_ws.__aiter__ = MagicMock(
            return_value=iter([])
        )

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("cordbeat.cli_adapter.websockets.connect", return_value=mock_connect),
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
        mock_ws.recv = AsyncMock(
            return_value=json.dumps({"content": "Welcome"})
        )
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
            patch("cordbeat.cli_adapter.websockets.connect", return_value=mock_connect),
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
        mock_ws.recv = AsyncMock(
            return_value=json.dumps({"content": "OK"})
        )
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
            patch("cordbeat.cli_adapter.websockets.connect", return_value=mock_connect),
            patch("builtins.input", side_effect=fake_input),
        ):
            await main()

        # Only handshake was sent, no messages
        assert mock_ws.send.await_count == 1

    async def test_main_keyboard_interrupt(self) -> None:
        """Ctrl+C gracefully exits."""
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(
            return_value=json.dumps({"content": "OK"})
        )
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        mock_connect = AsyncMock()
        mock_connect.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_connect.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("cordbeat.cli_adapter.websockets.connect", return_value=mock_connect),
            patch("builtins.input", side_effect=KeyboardInterrupt),
        ):
            await main()

        # Should not raise — graceful shutdown
        mock_ws.send.assert_awaited_once()  # handshake only
