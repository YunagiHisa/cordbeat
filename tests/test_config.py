"""Tests for configuration loader."""

from __future__ import annotations

from pathlib import Path

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


class TestCoerceValue:
    def test_true_values(self) -> None:
        for v in ("true", "True", "1", "yes"):
            assert _coerce_value(v) is True

    def test_false_values(self) -> None:
        for v in ("false", "False", "0", "no"):
            assert _coerce_value(v) is False

    def test_int(self) -> None:
        assert _coerce_value("42") == 42

    def test_float(self) -> None:
        assert _coerce_value("0.5") == 0.5

    def test_string(self) -> None:
        assert _coerce_value("hello") == "hello"


class TestLoadDotenv:
    def test_loads_env_file(self, tmp_path: Path, monkeypatch: object) -> None:
        import os

        env_file = tmp_path / ".env"
        env_file.write_text(
            '# comment\nMY_KEY=my_value\nQUOTED="quoted_val"\n',
            encoding="utf-8",
        )
        # Ensure MY_KEY is not already set
        os.environ.pop("MY_KEY", None)
        os.environ.pop("QUOTED", None)
        try:
            _load_dotenv(env_file)
            assert os.environ["MY_KEY"] == "my_value"
            assert os.environ["QUOTED"] == "quoted_val"
        finally:
            os.environ.pop("MY_KEY", None)
            os.environ.pop("QUOTED", None)

    def test_does_not_overwrite_existing(self, tmp_path: Path) -> None:
        import os

        env_file = tmp_path / ".env"
        env_file.write_text("MY_KEY=from_file\n", encoding="utf-8")
        os.environ["MY_KEY"] = "from_env"
        try:
            _load_dotenv(env_file)
            assert os.environ["MY_KEY"] == "from_env"
        finally:
            os.environ.pop("MY_KEY", None)

    def test_missing_file_is_noop(self, tmp_path: Path) -> None:
        _load_dotenv(tmp_path / "nonexistent")


class TestApplyEnvOverrides:
    def test_simple_override(self) -> None:
        import os

        raw: dict = {"gateway": {"host": "0.0.0.0"}}
        os.environ["CORDBEAT_GATEWAY__HOST"] = "127.0.0.1"
        try:
            _apply_env_overrides(raw)
            assert raw["gateway"]["host"] == "127.0.0.1"
        finally:
            os.environ.pop("CORDBEAT_GATEWAY__HOST")

    def test_nested_options(self) -> None:
        import os

        raw: dict = {"adapters": {"discord": {"options": {}}}}
        os.environ["CORDBEAT_ADAPTERS__DISCORD__OPTIONS__TOKEN"] = "secret"
        try:
            _apply_env_overrides(raw)
            assert raw["adapters"]["discord"]["options"]["token"] == "secret"
        finally:
            os.environ.pop("CORDBEAT_ADAPTERS__DISCORD__OPTIONS__TOKEN")

    def test_creates_missing_sections(self) -> None:
        import os

        raw: dict = {}
        os.environ["CORDBEAT_AI_BACKEND__MODEL"] = "gpt-4"
        try:
            _apply_env_overrides(raw)
            assert raw["ai_backend"]["model"] == "gpt-4"
        finally:
            os.environ.pop("CORDBEAT_AI_BACKEND__MODEL")

    def test_int_coercion(self) -> None:
        import os

        raw: dict = {"gateway": {}}
        os.environ["CORDBEAT_GATEWAY__PORT"] = "9000"
        try:
            _apply_env_overrides(raw)
            assert raw["gateway"]["port"] == 9000
        finally:
            os.environ.pop("CORDBEAT_GATEWAY__PORT")


class TestLoadConfigWithEnv:
    def test_env_overrides_yaml(self, tmp_path: Path) -> None:
        import os

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(
            "ai_backend:\n  model: llama3\n",
            encoding="utf-8",
        )
        os.environ["CORDBEAT_AI_BACKEND__MODEL"] = "gpt-4"
        try:
            config = load_config(cfg_file)
            assert config.ai_backend.model == "gpt-4"
        finally:
            os.environ.pop("CORDBEAT_AI_BACKEND__MODEL")

    def test_dotenv_file_loaded(self, tmp_path: Path) -> None:
        import os

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("", encoding="utf-8")
        env_file = tmp_path / ".env"
        env_file.write_text(
            "CORDBEAT_GATEWAY__PORT=7777\n",
            encoding="utf-8",
        )
        os.environ.pop("CORDBEAT_GATEWAY__PORT", None)
        try:
            config = load_config(cfg_file)
            assert config.gateway.port == 7777
        finally:
            os.environ.pop("CORDBEAT_GATEWAY__PORT", None)
