"""Tests for the per-skill uv-managed env manager."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cordbeat.skill_env import SkillEnvError, SkillEnvManager


@pytest.fixture()
def cache_root(tmp_path: Path) -> Path:
    root = tmp_path / "skill-envs"
    root.mkdir()
    return root


@pytest.fixture()
def skill_dir_with_deps(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "myskill"
    skill_dir.mkdir()
    (skill_dir / "main.py").write_text("def execute(): return {}\n")
    (skill_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = ["httpx>=0.27"]\n',
        encoding="utf-8",
    )
    return skill_dir


@pytest.fixture()
def skill_dir_no_deps(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "stdlibskill"
    skill_dir.mkdir()
    (skill_dir / "main.py").write_text("def execute(): return {}\n")
    return skill_dir


def _fake_uv_proc(returncode: int = 0, stderr: bytes = b"") -> MagicMock:
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(b"", stderr))
    return proc


async def test_prepare_returns_none_without_pyproject(
    cache_root: Path, skill_dir_no_deps: Path
) -> None:
    mgr = SkillEnvManager(cache_root=cache_root)
    result = await mgr.prepare(skill_dir_no_deps, "stdlibskill")
    assert result is None


async def test_prepare_builds_env_on_first_call(
    cache_root: Path, skill_dir_with_deps: Path
) -> None:
    mgr = SkillEnvManager(cache_root=cache_root)

    with patch("cordbeat.skill_env.shutil.which", return_value="/usr/bin/uv"):
        with patch(
            "cordbeat.skill_env.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_fake_uv_proc()),
        ) as spawn:
            python_exe = await mgr.prepare(skill_dir_with_deps, "myskill")

    assert python_exe is not None
    assert "myskill" in python_exe
    assert spawn.await_count == 2  # uv venv + uv pip install
    # Hash file written
    assert (cache_root / "myskill" / ".cordbeat-skill-hash").exists()


async def test_prepare_cache_hit_skips_build(
    cache_root: Path, skill_dir_with_deps: Path
) -> None:
    from cordbeat.skill_env import _python_in

    mgr = SkillEnvManager(cache_root=cache_root)
    env_dir = cache_root / "myskill"
    py_exe = _python_in(env_dir)
    py_exe.parent.mkdir(parents=True, exist_ok=True)
    py_exe.write_text("")
    import hashlib

    h = hashlib.sha256(
        (skill_dir_with_deps / "pyproject.toml").read_bytes()
    ).hexdigest()
    (env_dir / ".cordbeat-skill-hash").write_text(h, encoding="utf-8")

    with patch("cordbeat.skill_env.shutil.which", return_value="/usr/bin/uv"):
        with patch(
            "cordbeat.skill_env.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_fake_uv_proc()),
        ) as spawn:
            result = await mgr.prepare(skill_dir_with_deps, "myskill")

    assert result is not None
    assert spawn.await_count == 0  # no rebuild


async def test_prepare_hash_mismatch_triggers_rebuild(
    cache_root: Path, skill_dir_with_deps: Path
) -> None:
    mgr = SkillEnvManager(cache_root=cache_root)
    env_dir = cache_root / "myskill"
    env_dir.mkdir(parents=True)
    (env_dir / ".cordbeat-skill-hash").write_text("stalehash", encoding="utf-8")

    with patch("cordbeat.skill_env.shutil.which", return_value="/usr/bin/uv"):
        with patch(
            "cordbeat.skill_env.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_fake_uv_proc()),
        ) as spawn:
            await mgr.prepare(skill_dir_with_deps, "myskill")

    assert spawn.await_count == 2


async def test_prepare_raises_when_uv_missing(
    cache_root: Path, skill_dir_with_deps: Path
) -> None:
    mgr = SkillEnvManager(cache_root=cache_root)
    with patch("cordbeat.skill_env.shutil.which", return_value=None):
        with pytest.raises(SkillEnvError, match="uv is not available"):
            await mgr.prepare(skill_dir_with_deps, "myskill")


async def test_prepare_raises_when_uv_fails(
    cache_root: Path, skill_dir_with_deps: Path
) -> None:
    mgr = SkillEnvManager(cache_root=cache_root)
    with patch("cordbeat.skill_env.shutil.which", return_value="/usr/bin/uv"):
        with patch(
            "cordbeat.skill_env.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_fake_uv_proc(1, b"network error")),
        ):
            with pytest.raises(SkillEnvError, match="uv venv .* failed"):
                await mgr.prepare(skill_dir_with_deps, "myskill")


async def test_invalid_pyproject_raises(cache_root: Path, tmp_path: Path) -> None:
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "pyproject.toml").write_text(
        "[project]\ndependencies = 'not a list'\n", encoding="utf-8"
    )
    mgr = SkillEnvManager(cache_root=cache_root)
    with patch("cordbeat.skill_env.shutil.which", return_value="/usr/bin/uv"):
        with pytest.raises(SkillEnvError, match="must be a list"):
            await mgr.prepare(skill_dir, "broken")


async def test_no_deps_skips_pip_install(cache_root: Path, tmp_path: Path) -> None:
    skill_dir = tmp_path / "empty"
    skill_dir.mkdir()
    (skill_dir / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\ndependencies = []\n',
        encoding="utf-8",
    )
    mgr = SkillEnvManager(cache_root=cache_root)
    with patch("cordbeat.skill_env.shutil.which", return_value="/usr/bin/uv"):
        with patch(
            "cordbeat.skill_env.asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_fake_uv_proc()),
        ) as spawn:
            await mgr.prepare(skill_dir, "empty")
    # Only `uv venv` runs; no pip install since deps is empty
    assert spawn.await_count == 1
