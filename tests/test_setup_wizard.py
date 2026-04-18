"""Tests for the setup wizard."""

from __future__ import annotations

import json
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest
import yaml

from cordbeat.setup_wizard import (
    _build_config,
    _build_soul_yaml,
    _probe_llama_cpp,
    _probe_ollama,
    _probe_provider,
    _write_soul_files,
    run_wizard,
)

# -- Fixtures ──────────────────────────────────────────────────────────


class _OllamaHandler(BaseHTTPRequestHandler):
    """Minimal handler that mimics Ollama /api/tags."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/tags":
            body = json.dumps({"models": [{"name": "llama3:latest"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args: object) -> None:  # silence logs
        pass


@pytest.fixture()
def ollama_server() -> Generator[str, None, None]:
    server = HTTPServer(("127.0.0.1", 0), _OllamaHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class _LlamaCppHandler(BaseHTTPRequestHandler):
    """Minimal handler that mimics llama.cpp /v1/models."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps(
                {"data": [{"id": "my-model", "object": "model"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture()
def llama_cpp_server() -> Generator[str, None, None]:
    server = HTTPServer(("127.0.0.1", 0), _LlamaCppHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# -- _probe_ollama ─────────────────────────────────────────────────────


class TestProbeOllama:
    def test_detected(self, ollama_server: str) -> None:
        model = _probe_ollama(ollama_server)
        assert model == "llama3:latest"

    def test_not_detected(self) -> None:
        assert _probe_ollama("http://127.0.0.1:1") is None


# -- _probe_provider ───────────────────────────────────────────────────


class TestProbeProvider:
    def test_ollama_reachable(self, ollama_server: str) -> None:
        assert _probe_provider("ollama", ollama_server) is True

    def test_unreachable(self) -> None:
        assert _probe_provider("ollama", "http://127.0.0.1:1") is False


# -- _probe_llama_cpp ──────────────────────────────────────────────────


class TestProbeLlamaCpp:
    def test_detected(self, llama_cpp_server: str) -> None:
        model = _probe_llama_cpp(llama_cpp_server)
        assert model == "my-model"

    def test_not_detected(self) -> None:
        assert _probe_llama_cpp("http://127.0.0.1:1") is None


# -- _build_soul_yaml ──────────────────────────────────────────────────


class TestBuildSoulYaml:
    def test_default(self) -> None:
        data = _build_soul_yaml("Mika", "ja")
        assert data["identity"]["name"] == "Mika"
        assert data["identity"]["language"] == "ja"
        assert "traits" in data["personality"]


# -- _build_config ─────────────────────────────────────────────────────


class TestBuildConfig:
    def test_paths_anchored_to_home(self, tmp_path: Path) -> None:
        cfg = _build_config(
            tmp_path,
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3",
        )
        assert cfg["memory"]["sqlite_path"] == str(tmp_path / "cordbeat.db")
        assert cfg["soul"]["soul_dir"] == str(tmp_path / "soul")
        assert cfg["data_dir"] == str(tmp_path)

    def test_api_key_included(self, tmp_path: Path) -> None:
        cfg = _build_config(
            tmp_path,
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="gpt-4",
            api_key="sk-test",
        )
        assert cfg["ai_backend"]["options"]["api_key"] == "sk-test"

    def test_no_api_key(self, tmp_path: Path) -> None:
        cfg = _build_config(
            tmp_path,
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3",
        )
        assert "options" not in cfg["ai_backend"]


# -- _write_soul_files ─────────────────────────────────────────────────


class TestWriteSoulFiles:
    def test_creates_all_files(self, tmp_path: Path) -> None:
        soul_dir = tmp_path / "soul"
        _write_soul_files(soul_dir, "TestBot", "en")
        assert (soul_dir / "soul_core.yaml").is_file()
        assert (soul_dir / "soul.yaml").is_file()
        assert (soul_dir / "soul_notes.md").is_file()

    def test_soul_yaml_content(self, tmp_path: Path) -> None:
        soul_dir = tmp_path / "soul"
        _write_soul_files(soul_dir, "TestBot", "ja")
        data = yaml.safe_load((soul_dir / "soul.yaml").read_text(encoding="utf-8"))
        assert data["identity"]["name"] == "TestBot"
        assert data["identity"]["language"] == "ja"

    def test_does_not_overwrite(self, tmp_path: Path) -> None:
        soul_dir = tmp_path / "soul"
        soul_dir.mkdir()
        (soul_dir / "soul.yaml").write_text("custom: true", encoding="utf-8")
        _write_soul_files(soul_dir, "TestBot", "en")
        assert (soul_dir / "soul.yaml").read_text(encoding="utf-8") == "custom: true"


# -- run_wizard (integration) ──────────────────────────────────────────


class TestRunWizard:
    def test_ollama_detected_zero_questions(
        self, tmp_path: Path, ollama_server: str
    ) -> None:
        """When Ollama is detected, wizard asks only name + language."""
        inputs = iter(["TestBot", "en"])
        with (
            patch(
                "cordbeat.setup_wizard._probe_ollama",
                return_value="llama3:latest",
            ),
            patch("builtins.input", side_effect=inputs),
            patch(
                "cordbeat.setup_wizard._probe_provider",
                return_value=True,
            ),
        ):
            config_path = run_wizard(tmp_path)

        assert config_path == tmp_path / "config.yaml"
        assert config_path.is_file()
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert cfg["ai_backend"]["provider"] == "ollama"
        assert cfg["ai_backend"]["model"] == "llama3:latest"
        assert (tmp_path / "soul" / "soul.yaml").is_file()
        assert (tmp_path / "chroma").is_dir()
        assert (tmp_path / "skills").is_dir()

    def test_ollama_not_found_fallback(self, tmp_path: Path) -> None:
        """When Ollama is not detected, wizard asks provider details."""
        inputs = iter(["ollama", "http://myhost:11434", "mistral", "TestBot", "ja"])
        with (
            patch("cordbeat.setup_wizard._probe_ollama", return_value=None),
            patch("cordbeat.setup_wizard._probe_llama_cpp", return_value=None),
            patch("builtins.input", side_effect=inputs),
            patch(
                "cordbeat.setup_wizard._probe_provider",
                return_value=False,
            ),
        ):
            config_path = run_wizard(tmp_path)

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert cfg["ai_backend"]["base_url"] == "http://myhost:11434"
        assert cfg["ai_backend"]["model"] == "mistral"

    def test_llama_cpp_detected(self, tmp_path: Path) -> None:
        """When llama.cpp is detected (but not Ollama), wizard auto-configures."""
        inputs = iter(["TestBot", "en"])
        with (
            patch("cordbeat.setup_wizard._probe_ollama", return_value=None),
            patch(
                "cordbeat.setup_wizard._probe_llama_cpp",
                return_value="my-gguf-model",
            ),
            patch("builtins.input", side_effect=inputs),
            patch(
                "cordbeat.setup_wizard._probe_provider",
                return_value=True,
            ),
        ):
            config_path = run_wizard(tmp_path)

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert cfg["ai_backend"]["provider"] == "openai_compat"
        assert cfg["ai_backend"]["base_url"] == "http://localhost:8080/v1"
        assert cfg["ai_backend"]["model"] == "my-gguf-model"

    def test_openai_provider_asks_api_key(self, tmp_path: Path) -> None:
        """OpenAI provider asks for API key."""
        inputs = iter(
            [
                "openai",
                "https://api.openai.com/v1",
                "gpt-4",
                "sk-test123",
                "MyBot",
                "en",
            ]
        )
        with (
            patch("cordbeat.setup_wizard._probe_ollama", return_value=None),
            patch("cordbeat.setup_wizard._probe_llama_cpp", return_value=None),
            patch("builtins.input", side_effect=inputs),
            patch(
                "cordbeat.setup_wizard._probe_provider",
                return_value=True,
            ),
        ):
            config_path = run_wizard(tmp_path)

        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert cfg["ai_backend"]["provider"] == "openai"
        assert cfg["ai_backend"]["options"]["api_key"] == "sk-test123"

    def test_directory_structure_created(self, tmp_path: Path) -> None:
        inputs = iter(["Bot", "en"])
        with (
            patch(
                "cordbeat.setup_wizard._probe_ollama",
                return_value="llama3",
            ),
            patch("builtins.input", side_effect=inputs),
            patch(
                "cordbeat.setup_wizard._probe_provider",
                return_value=True,
            ),
        ):
            run_wizard(tmp_path)

        assert (tmp_path / "chroma").is_dir()
        assert (tmp_path / "skills").is_dir()
        assert (tmp_path / "soul").is_dir()
        assert (tmp_path / "soul" / "soul_core.yaml").is_file()
