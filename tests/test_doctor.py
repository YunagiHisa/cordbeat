"""Tests for the doctor diagnostic tool."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from cordbeat.tools.doctor import (
    _find_example_config,
    _find_project_config,
    _mask_secrets,
    _probe,
    main,
    run_doctor,
    run_fix_config,
    run_show_config,
    run_sync_config,
)


class TestRunDoctor:
    def _setup_healthy(self, home: Path) -> None:
        """Create a minimal healthy installation at *home*."""
        home.mkdir(parents=True, exist_ok=True)
        (home / "sandbox" / "skills").mkdir(parents=True)
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
            "skills_dir": str(home / "sandbox" / "skills"),
            "gateway": {"auth_token": "test-token-1234567890"},
        }
        (home / "config.yaml").write_text(
            yaml.dump(cfg, default_flow_style=False), encoding="utf-8"
        )

    def test_healthy_installation(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        self._setup_healthy(home)
        with patch("cordbeat.tools.doctor._probe", return_value=True):
            assert run_doctor(home) == 0

    def test_doctor_masks_gateway_auth_token(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        home = tmp_path / ".cordbeat"
        self._setup_healthy(home)
        with patch("cordbeat.tools.doctor._probe", return_value=True):
            assert run_doctor(home) == 0

        out = capsys.readouterr().out
        assert "test-token-1234567890" not in out
        assert "test-t...7890" in out

    def test_missing_config(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        home.mkdir()
        # No config.yaml → failure
        with patch("cordbeat.tools.doctor._probe", return_value=True):
            assert run_doctor(home) == 1

    def test_unreachable_backend(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        self._setup_healthy(home)
        with patch("cordbeat.tools.doctor._probe", return_value=False):
            assert run_doctor(home) == 1

    def test_missing_soul_files(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        self._setup_healthy(home)
        # Remove a soul file
        (home / "soul" / "soul_core.yaml").unlink()
        with patch("cordbeat.tools.doctor._probe", return_value=True):
            assert run_doctor(home) == 1

    def test_custom_home_via_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "custom"
        self._setup_healthy(home)
        monkeypatch.setenv("CORDBEAT_HOME", str(home))
        with patch("cordbeat.tools.doctor._probe", return_value=True):
            # run_doctor() without argument should pick up CORDBEAT_HOME
            assert run_doctor() == 0

    def test_missing_skills_dir(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        self._setup_healthy(home)
        import shutil

        shutil.rmtree(home / "sandbox" / "skills")
        with patch("cordbeat.tools.doctor._probe", return_value=True):
            assert run_doctor(home) == 1


class TestProbe:
    def test_probe_success(self) -> None:
        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=MagicMock())
        cm.__exit__ = MagicMock(return_value=False)
        with patch("cordbeat.tools.doctor.urllib.request.urlopen", return_value=cm):
            assert _probe("http://localhost:11434/api/tags") is True

    def test_probe_failure(self) -> None:
        import urllib.error

        with patch(
            "cordbeat.tools.doctor.urllib.request.urlopen",
            side_effect=urllib.error.URLError("nope"),
        ):
            assert _probe("http://localhost:11434/api/tags") is False


class TestDoctorExtraBranches:
    def test_non_ollama_provider_probes_models_endpoint(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        home.mkdir(parents=True)
        soul_dir = home / "soul"
        soul_dir.mkdir()
        (soul_dir / "soul_core.yaml").write_text("immutable_rules: []\n")
        (soul_dir / "soul.yaml").write_text("identity:\n  name: T\n")
        (home / "sandbox" / "skills").mkdir(parents=True)
        cfg = {
            "ai_backend": {"provider": "openai", "base_url": "https://api.example.com"},
            "soul": {"soul_dir": str(soul_dir)},
            "skills_dir": str(home / "sandbox" / "skills"),
            "gateway": {"auth_token": "x" * 20},
        }
        (home / "config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")

        with patch("cordbeat.tools.doctor._probe", return_value=True) as probe:
            run_doctor(home)
        probe.assert_called_once_with("https://api.example.com/models")

    def test_sqlite_vec_load_failure_is_reported(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("ai_backend: {}\n", encoding="utf-8")
        with (
            patch("cordbeat.tools.doctor._probe", return_value=True),
            patch("sqlite_vec.load", side_effect=RuntimeError("vec missing")),
        ):
            # Returns 1 overall (other checks also fail), but must not raise.
            assert run_doctor(home) == 1


class TestFindExampleConfig:
    def test_locates_repo_example_config(self) -> None:
        found = _find_example_config()
        assert found is not None
        assert found.name == "config.example.yaml"
        assert found.is_file()

    def test_returns_none_when_no_example_in_ancestry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point __file__ at an isolated tree with no config.example.yaml above it.
        fake_file = tmp_path / "a" / "b" / "doctor.py"
        fake_file.parent.mkdir(parents=True)
        monkeypatch.setattr("cordbeat.tools.doctor.__file__", str(fake_file))
        assert _find_example_config() is None


class TestFindProjectConfig:
    def test_finds_config_in_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "config.yaml").write_text("ai_backend: {}\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert _find_project_config() == tmp_path / "config.yaml"

    def test_finds_config_in_repo_ancestor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A project root is recognised by config.yaml + pyproject.toml together.
        proj = tmp_path / "proj"
        (proj / "pkg").mkdir(parents=True)
        (proj / "config.yaml").write_text("ai_backend: {}\n", encoding="utf-8")
        (proj / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
        empty_cwd = tmp_path / "elsewhere"
        empty_cwd.mkdir()
        monkeypatch.chdir(empty_cwd)  # cwd has no config.yaml
        monkeypatch.setattr(
            "cordbeat.tools.doctor.__file__", str(proj / "pkg" / "doctor.py")
        )
        assert _find_project_config() == proj / "config.yaml"

    def test_returns_none_when_no_project_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        empty_cwd = tmp_path / "elsewhere"
        empty_cwd.mkdir()
        monkeypatch.chdir(empty_cwd)
        fake_file = tmp_path / "iso" / "pkg" / "doctor.py"
        fake_file.parent.mkdir(parents=True)
        monkeypatch.setattr("cordbeat.tools.doctor.__file__", str(fake_file))
        assert _find_project_config() is None


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


# ── fix-config ────────────────────────────────────────────────────────────────


class TestRunFixConfig:
    def _write_cfg(self, home: Path, cfg: dict) -> None:
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text(
            yaml.dump(cfg, default_flow_style=False), encoding="utf-8"
        )

    def _write_example(self, path: Path, cfg: dict) -> None:
        path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")

    def test_no_config(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        home.mkdir()
        assert run_fix_config(home) == 1

    def test_no_example(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        self._write_cfg(home, {"ai_backend": {}})
        with patch("cordbeat.tools.doctor._find_example_config", return_value=None):
            assert run_fix_config(home) == 1

    def test_already_up_to_date(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        example_path = tmp_path / "config.example.yaml"
        cfg = {"ai_backend": {"provider": "ollama"}, "memory": {}}
        self._write_cfg(home, cfg)
        self._write_example(
            example_path, {"ai_backend": {"provider": "ollama"}, "memory": {}}
        )
        with patch(
            "cordbeat.tools.doctor._find_example_config", return_value=example_path
        ):
            assert run_fix_config(home) == 0

    def test_missing_section_added_on_yes(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        example_path = tmp_path / "config.example.yaml"
        self._write_cfg(home, {"ai_backend": {}})
        self._write_example(
            example_path, {"ai_backend": {}, "metrics": {"enabled": False}}
        )

        with (
            patch(
                "cordbeat.tools.doctor._find_example_config",
                return_value=example_path,
            ),
            patch("builtins.input", return_value="y"),
        ):
            assert run_fix_config(home) == 0

        updated = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert "metrics" in updated

    def test_missing_section_skipped_on_no(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        example_path = tmp_path / "config.example.yaml"
        self._write_cfg(home, {"ai_backend": {}})
        self._write_example(
            example_path, {"ai_backend": {}, "metrics": {"enabled": False}}
        )

        with (
            patch(
                "cordbeat.tools.doctor._find_example_config",
                return_value=example_path,
            ),
            patch("builtins.input", return_value="n"),
        ):
            assert run_fix_config(home) == 0

        updated = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert "metrics" not in updated

    def test_eof_at_prompt_aborts_cleanly(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        example_path = tmp_path / "config.example.yaml"
        self._write_cfg(home, {"ai_backend": {}})
        self._write_example(
            example_path, {"ai_backend": {}, "metrics": {"enabled": False}}
        )
        with (
            patch(
                "cordbeat.tools.doctor._find_example_config",
                return_value=example_path,
            ),
            patch("builtins.input", side_effect=EOFError),
        ):
            assert run_fix_config(home) == 0
        # Nothing was appended.
        updated = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert "metrics" not in updated


# ── show-config ───────────────────────────────────────────────────────────────


class TestRunShowConfig:
    def test_no_config(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        home.mkdir()
        assert run_show_config(home) == 1

    def test_secrets_masked(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        home = tmp_path / ".cordbeat"
        home.mkdir()
        cfg = {
            "gateway": {"auth_token": "supersecrettoken"},
            "ai_backend": {"api_key": "key123456"},
        }
        (home / "config.yaml").write_text(yaml.dump(cfg), encoding="utf-8")

        assert run_show_config(home) == 0
        out = capsys.readouterr().out
        assert "supersecrettoken" not in out
        assert "key123456" not in out
        assert "***" in out


class TestMaskSecrets:
    def test_masks_top_level(self) -> None:
        d: dict = {"token": "abc123def", "name": "test"}
        _mask_secrets(d)
        assert "abc123def" not in d["token"]
        assert d["name"] == "test"

    def test_masks_nested(self) -> None:
        d: dict = {"gateway": {"auth_token": "supersecret"}}
        _mask_secrets(d)
        assert "supersecret" not in d["gateway"]["auth_token"]

    def test_short_secret(self) -> None:
        d: dict = {"token": "abc"}
        _mask_secrets(d)
        assert d["token"] == "***"

    def test_empty_secret_unchanged(self) -> None:
        d: dict = {"token": ""}
        _mask_secrets(d)
        assert d["token"] == ""

    def test_non_dict_input_is_noop(self) -> None:
        # Guard clause: non-dict values are ignored without error.
        _mask_secrets("not a dict")
        _mask_secrets(["list", "of", "things"])


# ── sync-config ───────────────────────────────────────────────────────────────


class TestRunSyncConfig:
    def _make_cfg(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")

    def test_no_source_found(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        with patch("cordbeat.tools.doctor._find_project_config", return_value=None):
            assert run_sync_config(home) == 1

    def test_already_in_sync(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        src = tmp_path / "src" / "config.yaml"
        cfg = {"ai_backend": {"provider": "ollama"}}
        self._make_cfg(src, cfg)
        self._make_cfg(home / "config.yaml", cfg)

        assert run_sync_config(home, src_path=src) == 0

    def test_new_config_written(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        src = tmp_path / "src" / "config.yaml"
        self._make_cfg(src, {"ai_backend": {}, "metrics": {"enabled": True}})

        with patch("builtins.input", return_value="y"):
            assert run_sync_config(home, src_path=src) == 0

        result = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert result["metrics"]["enabled"] is True

    def test_backup_created_on_overwrite(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        src = tmp_path / "src" / "config.yaml"
        old_cfg = {"ai_backend": {"provider": "ollama"}}
        new_cfg = {"ai_backend": {"provider": "openai"}}
        self._make_cfg(home / "config.yaml", old_cfg)
        self._make_cfg(src, new_cfg)

        with patch("builtins.input", return_value="y"):
            assert run_sync_config(home, src_path=src) == 0

        assert (home / "config.yaml.bak").is_file()

    def test_dst_keys_preserved_in_merge(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        src = tmp_path / "src" / "config.yaml"
        self._make_cfg(src, {"ai_backend": {"provider": "openai"}})
        self._make_cfg(
            home / "config.yaml",
            {"ai_backend": {"provider": "ollama"}, "extra": {"key": "value"}},
        )

        with patch("builtins.input", return_value="y"):
            run_sync_config(home, src_path=src)

        result = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        # src wins for existing key
        assert result["ai_backend"]["provider"] == "openai"
        # dst-only key preserved
        assert result["extra"]["key"] == "value"

    def test_skip_on_no(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        src = tmp_path / "src" / "config.yaml"
        self._make_cfg(src, {"ai_backend": {"provider": "openai"}})
        self._make_cfg(home / "config.yaml", {"ai_backend": {"provider": "ollama"}})

        with patch("builtins.input", return_value="n"):
            assert run_sync_config(home, src_path=src) == 0

        result = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert result["ai_backend"]["provider"] == "ollama"

    def test_eof_at_prompt_aborts_cleanly(self, tmp_path: Path) -> None:
        home = tmp_path / ".cordbeat"
        src = tmp_path / "src" / "config.yaml"
        self._make_cfg(src, {"ai_backend": {"provider": "openai"}})
        self._make_cfg(home / "config.yaml", {"ai_backend": {"provider": "ollama"}})

        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert run_sync_config(home, src_path=src) == 0
        # Unchanged because the user aborted at the prompt.
        result = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
        assert result["ai_backend"]["provider"] == "ollama"


class TestMainMenu:
    def test_runs_selected_action_then_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # "1" → health check, "bad" → non-numeric, "9" → out of range, "0" → exit.
        answers = iter(["1", "bad", "9", "0"])
        monkeypatch.setattr("builtins.input", lambda *_: next(answers))
        monkeypatch.setattr(sys, "argv", ["cordbeat-doctor", str(tmp_path)])
        with patch("cordbeat.tools.doctor.run_doctor", return_value=0) as health:
            main()
        health.assert_called_once()

    def test_eof_exits_menu(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", MagicMock(side_effect=EOFError))
        monkeypatch.setattr(sys, "argv", ["cordbeat-doctor", str(tmp_path)])
        main()  # must not raise

    def test_config_file_arg_sets_home_to_parent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cfg = tmp_path / "config.yaml"
        cfg.write_text("ai_backend: {}\n", encoding="utf-8")
        monkeypatch.setattr("builtins.input", MagicMock(side_effect=EOFError))
        monkeypatch.setattr(sys, "argv", ["cordbeat-doctor", str(cfg)])
        main()  # EOF exits immediately; banner shows the resolved home (parent)
        assert str(tmp_path) in capsys.readouterr().out
