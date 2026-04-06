"""Tests for configuration loader."""

from __future__ import annotations

from pathlib import Path

from cordbeat.config import Config, load_config


class TestLoadConfig:
    def test_default_config(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(config, Config)
        assert config.gateway.host == "0.0.0.0"
        assert config.gateway.port == 8765

    def test_load_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "gateway:\n"
            "  host: 127.0.0.1\n"
            "  port: 9000\n"
            "ai_backend:\n"
            "  provider: ollama\n"
            "  model: mistral\n"
            "adapters:\n"
            "  discord:\n"
            "    core_ws_url: ws://localhost:9000\n"
            "    enabled: true\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.gateway.host == "127.0.0.1"
        assert config.gateway.port == 9000
        assert config.ai_backend.model == "mistral"
        assert "discord" in config.adapters
        assert config.adapters["discord"].core_ws_url == "ws://localhost:9000"
