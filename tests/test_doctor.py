"""Tests for the doctor diagnostic tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from cordbeat.doctor import run_doctor


class TestRunDoctor:
    def _setup_healthy(self, home: Path) -> None:
        """Create a minimal healthy installation at *home*."""
        home.mkdir(parents=True, exist_ok=True)
        (home / "skills").mkdir()
        soul_dir = home / "soul"
        soul_dir.mkdir()
        (soul_dir / "soul_core.yaml").write_text("immutable_rules: []\n")
        (soul_dir / "soul.yaml").write_text("identity:\n  name: Test\n")

        cfg = {
            "ai_backend": {
                "provider": "ollama",
                "base_url": "http://localhost:11434",
            },
            "memory": {
                "sqlite_path": str(home / "cordbeat.db"),
            },
            "soul": {"soul_dir": str(soul_dir)},
            "skills_dir": str(home / "skills"),
        }
        (home / "config.yaml").write_text(
            yaml.dump(cfg, default_flow_style=False), encoding="utf-8"
        )

    def test_healthy_installation(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        self._setup_healthy(home)
        with patch("cordbeat.doctor._probe", return_value=True):
            assert run_doctor(home) == 0

    def test_missing_config(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        home.mkdir()
        # No config.yaml → failure
        with patch("cordbeat.doctor._probe", return_value=True):
            assert run_doctor(home) == 1

    def test_unreachable_backend(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        self._setup_healthy(home)
        with patch("cordbeat.doctor._probe", return_value=False):
            assert run_doctor(home) == 1

    def test_missing_soul_files(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        self._setup_healthy(home)
        # Remove a soul file
        (home / "soul" / "soul_core.yaml").unlink()
        with patch("cordbeat.doctor._probe", return_value=True):
            assert run_doctor(home) == 1

    def test_custom_home_via_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "custom"
        self._setup_healthy(home)
        monkeypatch.setenv("CORDBEAT_HOME", str(home))
        with patch("cordbeat.doctor._probe", return_value=True):
            # run_doctor() without argument should pick up CORDBEAT_HOME
            assert run_doctor() == 0

    def test_missing_skills_dir(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        self._setup_healthy(home)
        import shutil

        shutil.rmtree(home / "skills")
        with patch("cordbeat.doctor._probe", return_value=True):
            assert run_doctor(home) == 1


class TestCordBeatHome:
    def test_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CORDBEAT_HOME", raising=False)
        from cordbeat.config import cordbeat_home

        assert cordbeat_home() == Path.home() / ".cordbeat"

    def test_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORDBEAT_HOME", str(tmp_path / "custom"))
        from cordbeat.config import cordbeat_home

        assert cordbeat_home() == (tmp_path / "custom").resolve()
