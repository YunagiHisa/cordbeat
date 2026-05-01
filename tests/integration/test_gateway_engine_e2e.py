"""End-to-end integration test: real WebSocket client → GatewayServer → CoreEngine.

Tests the full pipeline without mocking any WebSocket internals:
  real WS client → GatewayServer (real) → MessageQueue (real) → CoreEngine (AI mocked)

This proves the WS handshake, message routing, and reply path work correctly,
giving confidence that Discord/Telegram/CLI adapters all share the same working
protocol.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import websockets

from cordbeat.agent.soul import Soul
from cordbeat.config import GatewayConfig, MemoryConfig
from cordbeat.core.engine import CoreEngine
from cordbeat.core.gateway import GatewayServer, MessageQueue
from cordbeat.memory import MemoryStore
from cordbeat.models import GatewayMessage, MessageType
from cordbeat.skills import SkillRegistry


# ── Helpers ────────────────────────────────────────────────────────────


def _make_mock_ai() -> AsyncMock:
    ai = AsyncMock()

    async def _generate(**kwargs: object) -> str:
        prompt = str(kwargs.get("prompt", ""))
        if "recall keywords" in prompt.lower():
            return '{"keywords": []}'
        if "what emotion" in prompt.lower():
            return '{"emotion": "neutral", "intensity": 0.5}'
        if "extract memory" in prompt.lower():
            return (
                '{"topic": "greeting", "emotional_tone": "neutral",'
                ' "facts": [], "episode_summary": ""}'
            )
        return "E2E reply: pong!"

    ai.generate = AsyncMock(side_effect=_generate)
    return ai


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
async def live_system(tmp_path: Path) -> dict:  # type: ignore[type-arg]
    """Start a real GatewayServer + CoreEngine stack and yield connection info."""
    # Choose a random free port to avoid conflicts
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    gw_config = GatewayConfig(host="127.0.0.1", port=port, auth_token="")
    queue = MessageQueue()
    gateway = GatewayServer(gw_config, queue)

    mem_config = MemoryConfig(sqlite_path=str(tmp_path / "e2e.db"))
    memory = MemoryStore(mem_config)
    await memory.initialize()

    soul = Soul(tmp_path / "soul")
    skills = SkillRegistry(tmp_path / "skills")
    ai = _make_mock_ai()

    engine = CoreEngine(
        ai=ai,
        soul=soul,
        memory=memory,
        skills=skills,
        gateway=gateway,
    )
    queue.set_handler(engine.handle_message)

    # Start server and queue processor
    await gateway.start()
    queue_task = asyncio.create_task(queue.process_loop())

    yield {"url": f"ws://127.0.0.1:{port}", "ai": ai}

    queue_task.cancel()
    await gateway.stop()
    await memory.close()


# ── Tests ─────────────────────────────────────────────────────────────


class TestGatewayEngineE2E:
    """Full round-trip: WS client → gateway → engine → WS reply."""

    async def test_handshake_ack(self, live_system: dict) -> None:
        """Real WS handshake returns an ACK from the gateway."""
        async with websockets.connect(live_system["url"]) as ws:
            await ws.send(json.dumps({"adapter_id": "e2e_test"}))
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            ack = json.loads(raw)

        assert ack["type"] == MessageType.ACK.value
        assert "e2e_test" in ack["content"]

    async def test_chat_message_gets_reply(self, live_system: dict) -> None:
        """Sending a chat message returns a reply through the WS pipeline."""
        async with websockets.connect(live_system["url"]) as ws:
            # Handshake
            await ws.send(json.dumps({"adapter_id": "e2e_test"}))
            await asyncio.wait_for(ws.recv(), timeout=3.0)

            # Send chat message
            msg = GatewayMessage(
                type=MessageType.MESSAGE,
                adapter_id="e2e_test",
                platform_user_id="user_001",
                content="ping",
            )
            await ws.send(
                json.dumps(
                    {
                        "type": msg.type.value,
                        "adapter_id": msg.adapter_id,
                        "platform_user_id": msg.platform_user_id,
                        "content": msg.content,
                    }
                )
            )

            # Wait for reply (engine processing + AI mock)
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            reply = json.loads(raw)

        assert reply["type"] in (
            MessageType.MESSAGE.value,
            "heartbeat_message",
            MessageType.ERROR.value,
        )
        assert reply.get("platform_user_id") == "user_001"
        assert reply.get("content")  # non-empty response

    async def test_invalid_message_gets_error(self, live_system: dict) -> None:
        """Malformed message after handshake returns an error frame."""
        async with websockets.connect(live_system["url"]) as ws:
            # Handshake
            await ws.send(json.dumps({"adapter_id": "e2e_test"}))
            await asyncio.wait_for(ws.recv(), timeout=3.0)

            # Send garbage JSON
            await ws.send("not valid json {{{{")

            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            reply = json.loads(raw)

        assert reply["type"] == MessageType.ERROR.value

    async def test_missing_adapter_id_closes(self, live_system: dict) -> None:
        """Handshake without adapter_id results in a closed connection."""
        async with websockets.connect(live_system["url"]) as ws:
            await ws.send(json.dumps({"no_adapter_id": True}))
            # Server should close the connection
            with pytest.raises((websockets.ConnectionClosed, EOFError, OSError)):
                # Either recv() raises or we need to wait for close
                await asyncio.wait_for(ws.recv(), timeout=3.0)

    async def test_multiple_clients_independent(self, live_system: dict) -> None:
        """Multiple adapters can connect simultaneously and get separate replies."""
        url = live_system["url"]

        async def _client_roundtrip(adapter_id: str) -> dict:  # type: ignore[type-arg]
            async with websockets.connect(url) as ws:
                await ws.send(json.dumps({"adapter_id": adapter_id}))
                await asyncio.wait_for(ws.recv(), timeout=3.0)
                await ws.send(
                    json.dumps(
                        {
                            "type": "message",
                            "adapter_id": adapter_id,
                            "platform_user_id": f"user_{adapter_id}",
                            "content": "hello",
                        }
                    )
                )
                raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                return json.loads(raw)

        r1, r2 = await asyncio.gather(
            _client_roundtrip("client_a"),
            _client_roundtrip("client_b"),
        )

        assert r1["platform_user_id"] == "user_client_a"
        assert r2["platform_user_id"] == "user_client_b"
