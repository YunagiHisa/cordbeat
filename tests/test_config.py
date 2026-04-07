"""Tests for configuration loader."""

from __future__ import annotations

from pathlib import Path

from cordbeat.config import (
    Config,
    load_config,
)


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

    def test_unknown_keys_ignored(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "gateway:\n  host: 127.0.0.1\n  unknown_field: value\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.gateway.host == "127.0.0.1"

    def test_heartbeat_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "heartbeat:\n  default_interval_minutes: 30\n  min_interval_minutes: 10\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.heartbeat.default_interval_minutes == 30
        assert config.heartbeat.min_interval_minutes == 10

    def test_memory_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "memory:\n  sqlite_path: custom.db\n  decay_rate: 0.5\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.memory.sqlite_path == "custom.db"
        assert config.memory.decay_rate == 0.5

    def test_ai_backend_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "ai_backend:\n"
            "  provider: openai\n"
            "  base_url: http://localhost:8000\n"
            "  model: gpt-4\n"
            "  options:\n"
            "    api_key: sk-test\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.ai_backend.provider == "openai"
        assert config.ai_backend.model == "gpt-4"
        assert config.ai_backend.options["api_key"] == "sk-test"

    def test_empty_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        config = load_config(cfg_file)
        assert isinstance(config, Config)

    def test_top_level_fields(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "soul_dir: custom/soul\nskills_dir: custom/skills\ndata_dir: custom/data\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.soul_dir == "custom/soul"
        assert config.skills_dir == "custom/skills"
        assert config.data_dir == "custom/data"
