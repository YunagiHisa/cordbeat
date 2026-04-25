"""Tests for configuration loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cordbeat.config import (
    Config,
    _apply_env_overrides,
    _coerce_value,
    _load_dotenv,
    load_config,
)


class TestLoadConfig:
    def test_default_config(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(config, Config)
        assert config.gateway.host == "127.0.0.1"
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
        # Relative paths are resolved against the config file's directory
        assert config.memory.sqlite_path == str((tmp_path / "custom.db").resolve())
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
        # Relative paths are resolved against the config file's directory
        assert config.soul_dir == str((tmp_path / "custom/soul").resolve())
        assert config.skills_dir == str((tmp_path / "custom/skills").resolve())
        assert config.data_dir == str((tmp_path / "custom/data").resolve())


class TestCoerceValue:
    def test_true_values(self) -> None:
        for v in ("true", "True", "yes"):
            assert _coerce_value(v) is True

    def test_false_values(self) -> None:
        for v in ("false", "False", "no"):
            assert _coerce_value(v) is False

    def test_int(self) -> None:
        assert _coerce_value("42") == 42

    def test_int_zero_and_one(self) -> None:
        assert _coerce_value("0") == 0
        assert _coerce_value("1") == 1

    def test_float(self) -> None:
        assert _coerce_value("0.5") == 0.5

    def test_string(self) -> None:
        assert _coerce_value("hello") == "hello"


class TestLoadDotenv:
    def test_loads_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            '# comment\nMY_KEY=my_value\nQUOTED="quoted_val"\n',
            encoding="utf-8",
        )
        monkeypatch.delenv("MY_KEY", raising=False)
        monkeypatch.delenv("QUOTED", raising=False)
        _load_dotenv(env_file)
        assert os.environ["MY_KEY"] == "my_value"
        assert os.environ["QUOTED"] == "quoted_val"

    def test_does_not_overwrite_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("MY_KEY=from_file\n", encoding="utf-8")
        monkeypatch.setenv("MY_KEY", "from_env")
        _load_dotenv(env_file)
        assert os.environ["MY_KEY"] == "from_env"

    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        _load_dotenv(tmp_path / "nonexistent")


class TestApplyEnvOverrides:
    def test_simple_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raw: dict = {"gateway": {"host": "0.0.0.0"}}
        monkeypatch.setenv("CORDBEAT_GATEWAY__HOST", "127.0.0.1")
        _apply_env_overrides(raw)
        assert raw["gateway"]["host"] == "127.0.0.1"

    def test_nested_options(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raw: dict = {"adapters": {"discord": {"options": {}}}}
        monkeypatch.setenv("CORDBEAT_ADAPTERS__DISCORD__OPTIONS__TOKEN", "secret")
        _apply_env_overrides(raw)
        assert raw["adapters"]["discord"]["options"]["token"] == "secret"

    def test_creates_missing_sections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raw: dict = {}
        monkeypatch.setenv("CORDBEAT_AI_BACKEND__MODEL", "gpt-4")
        _apply_env_overrides(raw)
        assert raw["ai_backend"]["model"] == "gpt-4"

    def test_int_coercion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        raw: dict = {"gateway": {}}
        monkeypatch.setenv("CORDBEAT_GATEWAY__PORT", "9000")
        _apply_env_overrides(raw)
        assert raw["gateway"]["port"] == 9000


class TestLoadConfigWithEnv:
    def test_env_overrides_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "ai_backend:\n  model: llama3\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("CORDBEAT_AI_BACKEND__MODEL", "gpt-4")
        config = load_config(cfg_file)
        assert config.ai_backend.model == "gpt-4"

    def test_dotenv_file_loaded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        env_file = tmp_path / ".env"
        env_file.write_text(
            "CORDBEAT_GATEWAY__PORT=7777\n",
            encoding="utf-8",
        )
        monkeypatch.delenv("CORDBEAT_GATEWAY__PORT", raising=False)
        config = load_config(cfg_file)
        assert config.gateway.port == 7777


class TestLogConfig:
    def test_default_log_config(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        config = load_config(cfg_file)
        assert config.log.level == "INFO"
        assert "%(asctime)s" in config.log.format

    def test_custom_log_level(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "log:\n  level: DEBUG\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.log.level == "DEBUG"

    def test_custom_log_format(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            'log:\n  format: "%(message)s"\n',
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.log.format == "%(message)s"

    def test_env_override_log_level(self, tmp_path: Path) -> None:
        import os

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        os.environ["CORDBEAT_LOG__LEVEL"] = "WARNING"
        try:
            config = load_config(cfg_file)
            assert config.log.level == "WARNING"
        finally:
            os.environ.pop("CORDBEAT_LOG__LEVEL", None)


class TestPathResolution:
    """Relative paths in config are resolved relative to the config file."""

    def test_memory_paths_resolved_to_config_dir(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "memory:\n  sqlite_path: db/app.db\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.memory.sqlite_path == str((tmp_path / "db/app.db").resolve())

    def test_soul_dir_resolved(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("soul:\n  soul_dir: my_soul\n", encoding="utf-8")
        config = load_config(cfg_file)
        assert config.soul.soul_dir == str((tmp_path / "my_soul").resolve())

    def test_absolute_paths_unchanged(self, tmp_path: Path) -> None:
        abs_path = str(tmp_path / "absolute" / "path.db")
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            f"memory:\n  sqlite_path: {abs_path}\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.memory.sqlite_path == abs_path

    def test_nested_config_dir(self, tmp_path: Path) -> None:
        sub_dir = tmp_path / "conf" / "sub"
        sub_dir.mkdir(parents=True)
        cfg_file = sub_dir / "config.yaml"
        cfg_file.write_text(
            "data_dir: ../data\nskills_dir: ../../skills\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.data_dir == str((sub_dir / "../data").resolve())
        assert config.skills_dir == str((sub_dir / "../../skills").resolve())

    def test_log_file_resolved(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "log:\n  file: logs/cordbeat.log\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.log.file == str((tmp_path / "logs/cordbeat.log").resolve())

    def test_empty_log_file_not_resolved(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("log:\n  level: DEBUG\n", encoding="utf-8")
        config = load_config(cfg_file)
        assert config.log.file == ""

    def test_default_paths_resolved_to_config_dir(self, tmp_path: Path) -> None:
        """Even default values like 'data/cordbeat.db' get resolved."""
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        config = load_config(cfg_file)
        assert config.memory.sqlite_path == str(
            (tmp_path / "data/cordbeat.db").resolve()
        )
        assert config.data_dir == str((tmp_path / "data").resolve())


class TestLogRotationConfig:
    """LogConfig rotation fields are parsed correctly."""

    def test_default_rotation_values(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        config = load_config(cfg_file)
        assert config.log.file == ""
        assert config.log.max_bytes == 10_485_760
        assert config.log.backup_count == 5

    def test_custom_rotation_values(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "log:\n  file: app.log\n  max_bytes: 5242880\n  backup_count: 3\n",
            encoding="utf-8",
        )
        config = load_config(cfg_file)
        assert config.log.max_bytes == 5_242_880
        assert config.log.backup_count == 3
