"""Tests for cordbeat-add (add_cmd.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from cordbeat.tools.add_cmd import (
    _check_deps_installed,
    _find_config,
    _load_yaml,
    _read_skill_deps,
    _read_skill_info,
    _save_yaml,
    _update_env_file,
    run_add,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# _find_config
# ---------------------------------------------------------------------------


def test_find_config_current_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("gateway:\n  port: 8765\n", encoding="utf-8")
    assert _find_config() == cfg


def test_find_config_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with patch("cordbeat.tools.add_cmd.cordbeat_home", return_value=tmp_path):
        assert _find_config() is None


# ---------------------------------------------------------------------------
# _update_env_file
# ---------------------------------------------------------------------------


def test_update_env_file_creates_new(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    _update_env_file(env, "FOO", "bar")
    assert "FOO=bar" in env.read_text(encoding="utf-8")


def test_update_env_file_updates_existing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("FOO=old\nBAR=baz\n", encoding="utf-8")
    _update_env_file(env, "FOO", "new")
    content = env.read_text(encoding="utf-8")
    assert "FOO=new" in content
    assert "FOO=old" not in content
    assert "BAR=baz" in content


def test_update_env_file_appends_missing(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("BAR=baz\n", encoding="utf-8")
    _update_env_file(env, "FOO", "bar")
    content = env.read_text(encoding="utf-8")
    assert "FOO=bar" in content
    assert "BAR=baz" in content


# ---------------------------------------------------------------------------
# _check_deps_installed
# ---------------------------------------------------------------------------


def test_check_deps_installed_present() -> None:
    # httpx is always importable in test env (it's a core dependency)
    missing = _check_deps_installed(["httpx>=0.27"])
    assert missing == []


def test_check_deps_installed_missing() -> None:
    missing = _check_deps_installed(["nonexistent-pkg-xyz>=1.0"])
    assert "nonexistent-pkg-xyz>=1.0" in missing


# ---------------------------------------------------------------------------
# _read_skill_info / _read_skill_deps
# ---------------------------------------------------------------------------


def test_read_skill_info_ok(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my_skill"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        "name: my_skill\ndescription: Test skill\nsafety:\n  level: safe\n",
        encoding="utf-8",
    )
    info = _read_skill_info(skill_dir)
    assert info is not None
    assert info["name"] == "my_skill"


def test_read_skill_info_missing_yaml(tmp_path: Path) -> None:
    skill_dir = tmp_path / "no_yaml"
    skill_dir.mkdir()
    assert _read_skill_info(skill_dir) is None


def test_read_skill_deps_ok(tmp_path: Path) -> None:
    skill_dir = tmp_path / "my_skill"
    skill_dir.mkdir()
    (skill_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["pillow>=10.0"]\n',
        encoding="utf-8",
    )
    deps = _read_skill_deps(skill_dir)
    assert "pillow>=10.0" in deps


def test_read_skill_deps_no_toml(tmp_path: Path) -> None:
    skill_dir = tmp_path / "no_toml"
    skill_dir.mkdir()
    assert _read_skill_deps(skill_dir) == []


# ---------------------------------------------------------------------------
# run_add — no config exits
# ---------------------------------------------------------------------------


def test_run_add_no_config_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with patch("cordbeat.tools.add_cmd.cordbeat_home", return_value=tmp_path):
        with pytest.raises(SystemExit):
            run_add()


# ---------------------------------------------------------------------------
# run_add — adapter flow (mocked I/O)
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "config.yaml"
    _write_yaml(
        cfg_path,
        {
            "adapters": {
                "cli": {"enabled": True, "core_ws_url": "ws://localhost:8765"},
            }
        },
    )
    return cfg_path


def test_run_add_adapter_discord(tmp_path: Path) -> None:
    cfg_path = _make_config(tmp_path)

    # menu=1(adapter), adapter=2(discord), install=Y, token
    inputs = iter(["1", "2", "Y", "BOT-TOKEN-123"])
    with (
        patch("cordbeat.tools.add_cmd._ask", side_effect=lambda p, d="": next(inputs)),
        patch("cordbeat.tools.add_cmd._is_importable", return_value=False),
        patch("cordbeat.tools.add_cmd._install_packages", return_value=True),
    ):
        run_add(config_path=cfg_path)

    updated = _load_yaml(cfg_path)
    assert updated["adapters"].get("discord", {}).get("enabled") is True

    env_path = tmp_path / ".env"
    assert "BOT-TOKEN-123" in env_path.read_text(encoding="utf-8")


def test_run_add_adapter_cli_warns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_path = _make_config(tmp_path)

    inputs = iter(["1", "1"])  # menu=1(adapter), adapter=1(CLI)
    with patch("cordbeat.tools.add_cmd._ask", side_effect=lambda p, d="": next(inputs)):
        run_add(config_path=cfg_path)

    # Should warn and NOT crash
    captured = capsys.readouterr()
    assert "CLI" in captured.out or "cli" in captured.out.lower()


# ---------------------------------------------------------------------------
# run_add — skill flow (mocked I/O)
# ---------------------------------------------------------------------------


def _make_skill(skills_dir: Path, name: str, *, deps: list[str] | None = None) -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.yaml").write_text(
        f"name: {name}\ndescription: Test\nsafety:\n  level: safe\n",
        encoding="utf-8",
    )
    if deps:
        toml = f'[project]\nname = "{name}"\ndependencies = {deps!r}\n'
        (skill_dir / "pyproject.toml").write_text(toml, encoding="utf-8")
    (skill_dir / "main.py").write_text(
        "async def execute(**kwargs): return {}\n", encoding="utf-8"
    )


def test_run_add_skill_no_deps(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "timer")
    cfg_path = _make_config(tmp_path)
    cfg = _load_yaml(cfg_path)
    cfg["skills_dir"] = str(skills_dir)
    _save_yaml(cfg_path, cfg)

    inputs = iter(["2", "1"])  # menu=2(skill), skill=1
    with (
        patch("cordbeat.tools.add_cmd._ask", side_effect=lambda p, d="": next(inputs)),
        patch("cordbeat.tools.add_cmd._find_bundled_skills_dir", return_value=None),
    ):
        run_add(config_path=cfg_path)  # Should not raise


def test_run_add_skill_installs_deps(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir, "draw", deps=["pillow>=10.0"])
    cfg_path = _make_config(tmp_path)
    cfg = _load_yaml(cfg_path)
    cfg["skills_dir"] = str(skills_dir)
    _save_yaml(cfg_path, cfg)

    install_called: list[bool] = []

    def fake_install(pkgs: list[str]) -> bool:
        install_called.append(True)
        return True

    inputs = iter(["2", "1", "Y"])  # menu=skill, skill=1, install=Y
    with (
        patch("cordbeat.tools.add_cmd._ask", side_effect=lambda p, d="": next(inputs)),
        patch("cordbeat.tools.add_cmd._find_bundled_skills_dir", return_value=None),
        patch(
            "cordbeat.tools.add_cmd._check_deps_installed",
            return_value=["pillow>=10.0"],
        ),
        patch("cordbeat.tools.add_cmd._install_skill_deps", side_effect=fake_install),
    ):
        run_add(config_path=cfg_path)

    assert install_called


def test_run_add_unknown_choice_exits(tmp_path: Path) -> None:
    cfg_path = _make_config(tmp_path)

    inputs = iter(["9"])  # invalid choice
    with (
        patch("cordbeat.tools.add_cmd._ask", side_effect=lambda p, d="": next(inputs)),
        pytest.raises(SystemExit),
    ):
        run_add(config_path=cfg_path)
