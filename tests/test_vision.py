"""Tests for E-1 vision / image support."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

from cordbeat.ai_backend import (
    OllamaBackend,
    OpenAICompatBackend,
    _detect_image_mime,
)
from cordbeat.config import AIBackendConfig
from cordbeat.engine import CoreEngine
from cordbeat.models import GatewayMessage, MessageType

# ── Helper ────────────────────────────────────────────────────────────


def _make_b64(magic: bytes) -> str:
    """Return a base64 string whose decoded prefix matches magic."""
    raw = magic + b"\x00" * 20
    return base64.b64encode(raw).decode("ascii")


# ── MIME detection ────────────────────────────────────────────────────


class TestDetectImageMime:
    def test_jpeg(self) -> None:
        assert _detect_image_mime(_make_b64(b"\xff\xd8\xff")) == "image/jpeg"

    def test_png(self) -> None:
        assert _detect_image_mime(_make_b64(b"\x89PNG")) == "image/png"

    def test_gif(self) -> None:
        assert _detect_image_mime(_make_b64(b"GIF8")) == "image/gif"

    def test_default_unknown(self) -> None:
        assert _detect_image_mime(_make_b64(b"\x00\x00\x00\x00")) == "image/jpeg"


# ── GatewayMessage.images field ───────────────────────────────────────


class TestGatewayMessageImages:
    def test_images_default_empty(self) -> None:
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="u1",
            content="hello",
        )
        assert msg.images == []

    def test_images_set(self) -> None:
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="test",
            platform_user_id="u1",
            content="hello",
            images=["abc123"],
        )
        assert msg.images == ["abc123"]


# ── OllamaBackend.generate_with_vision ───────────────────────────────


class TestOllamaVision:
    async def test_generate_with_vision_sends_chat_api(self) -> None:
        cfg = AIBackendConfig(provider="ollama", model="llava")
        backend = OllamaBackend(cfg)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "I see a cat"}}

        with patch.object(
            backend._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            result = await backend.generate_with_vision(
                prompt="What is in this image?",
                images=["base64data"],
                system="You are helpful",
            )

        assert result == "I see a cat"
        call_args = mock_post.call_args
        assert "/api/chat" in call_args[0][0]
        body = call_args[1]["json"]
        user_msg = next(m for m in body["messages"] if m["role"] == "user")
        assert user_msg["images"] == ["base64data"]

    async def test_generate_with_vision_includes_system(self) -> None:
        cfg = AIBackendConfig(provider="ollama", model="llava")
        backend = OllamaBackend(cfg)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}

        with patch.object(
            backend._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            await backend.generate_with_vision(
                "prompt", ["img"], system="sys prompt"
            )

        body = mock_post.call_args[1]["json"]
        roles = [m["role"] for m in body["messages"]]
        assert "system" in roles


# ── OpenAICompatBackend.generate_with_vision ─────────────────────────


class TestOpenAIVision:
    async def test_generate_with_vision_uses_content_array(self) -> None:
        cfg = AIBackendConfig(provider="openai_compat", model="gpt-4o")
        backend = OpenAICompatBackend(cfg)

        jpeg_b64 = _make_b64(b"\xff\xd8\xff")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "A dog"}}]
        }

        with patch.object(
            backend._client, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            result = await backend.generate_with_vision(
                prompt="Describe this",
                images=[jpeg_b64],
            )

        assert result == "A dog"
        body = mock_post.call_args[1]["json"]
        user_msg = next(m for m in body["messages"] if m["role"] == "user")
        content = user_msg["content"]
        assert isinstance(content, list)
        texts = [c for c in content if c["type"] == "text"]
        imgs = [c for c in content if c["type"] == "image_url"]
        assert len(texts) == 1
        assert len(imgs) == 1
        assert "data:image/jpeg;base64," in imgs[0]["image_url"]["url"]


# ── CoreEngine vision routing ─────────────────────────────────────────


class TestCoreEngineVision:
    def _make_engine(self, vision_enabled: bool = True) -> tuple[CoreEngine, MagicMock]:
        from cordbeat.config import MemoryConfig

        ai = MagicMock()
        ai.generate = AsyncMock(return_value="text reply")
        ai.generate_with_vision = AsyncMock(return_value="vision reply")

        soul = MagicMock()
        soul.get_soul_snapshot.return_value = {
            "name": "Cord",
            "notes": "",
            "traits": [],
            "immutable_rules": ["Be helpful"],
            "emotion": {
                "primary": "calm",
                "intensity": 0.5,
                "secondary": None,
                "secondary_intensity": 0.0,
            },
            "quiet_hours": {"start": "01:00", "end": "07:00"},
        }
        memory = MagicMock()
        memory.resolve_user = AsyncMock(return_value="uid1")
        memory.get_or_create_user = AsyncMock(
            return_value=MagicMock(
                display_name="User",
                last_talked_at=None,
                last_platform=None,
            )
        )
        memory.update_user_summary = AsyncMock()
        memory.get_core_profile = AsyncMock(return_value=None)
        memory.get_recent_messages = AsyncMock(return_value=[])
        memory.search_semantic = AsyncMock(return_value=[])
        memory.search_episodic = AsyncMock(return_value=[])
        memory.search_by_emotion = AsyncMock(return_value=[])
        memory.get_chain_links = AsyncMock(return_value=[])
        memory.get_recall_hints = AsyncMock(return_value=[])
        memory.add_message = AsyncMock()

        extractor = MagicMock()
        extractor.extract_recall_keywords = AsyncMock(return_value=[])
        extractor.infer_and_update_emotion = AsyncMock()
        extractor.extract_and_store_memories = AsyncMock()

        skills = MagicMock()
        skills.available_skills = []
        gateway = MagicMock()
        gateway.send_to_adapter = AsyncMock()

        engine = CoreEngine(
            ai=ai,
            soul=soul,
            memory=memory,
            skills=skills,
            gateway=gateway,
            memory_config=MemoryConfig(),
            vision_enabled=vision_enabled,
        )
        engine._extractor = extractor
        return engine, ai

    async def test_uses_generate_with_vision_when_enabled(self) -> None:
        engine, ai = self._make_engine(vision_enabled=True)
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="u1",
            content="What is this?",
            images=["imgdata"],
        )
        await engine.handle_message(msg)
        ai.generate_with_vision.assert_called_once()
        ai.generate.assert_not_called()

    async def test_uses_generate_when_vision_disabled(self) -> None:
        engine, ai = self._make_engine(vision_enabled=False)
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="u1",
            content="What is this?",
            images=["imgdata"],
        )
        await engine.handle_message(msg)
        ai.generate.assert_called_once()
        ai.generate_with_vision.assert_not_called()

    async def test_uses_generate_when_no_images(self) -> None:
        engine, ai = self._make_engine(vision_enabled=True)
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="u1",
            content="Hello",
        )
        await engine.handle_message(msg)
        ai.generate.assert_called_once()
        ai.generate_with_vision.assert_not_called()

    async def test_falls_back_to_text_when_vision_fails(self) -> None:
        """If generate_with_vision raises, engine falls back to text-only generate."""
        engine, ai = self._make_engine(vision_enabled=True)
        ai.generate_with_vision.side_effect = RuntimeError(
            "model does not support vision"
        )
        msg = GatewayMessage(
            type=MessageType.MESSAGE,
            adapter_id="discord",
            platform_user_id="u1",
            content="What is this?",
            images=["imgdata"],
        )
        await engine.handle_message(msg)
        ai.generate_with_vision.assert_called_once()
        ai.generate.assert_called_once()
