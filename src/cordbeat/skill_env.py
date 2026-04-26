"""Per-skill ``uv``-managed Python environment manager.

When a skill ships its own ``pyproject.toml`` we install its declared
dependencies into an isolated environment under
``~/.cordbeat/skill-envs/<name>/`` so its dependencies cannot collide
with — or be tampered through — the host process. Skills without a
``pyproject.toml`` fall back to running with the host Python (this
preserves backward compatibility for ad-hoc test skills and
AI-generated skills that have no extra deps).

The environment is built lazily on first execution and cached. The
``pyproject.toml`` contents are SHA-256 hashed and stored alongside
the env; any change triggers a fresh rebuild.

Note: only the ``[project].dependencies`` list is honoured. We do
*not* run a wheel build for the skill itself — the runner imports
``main.py`` directly from the skill directory.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
import sys
import tomllib
from pathlib import Path

from .exceptions import SkillError

logger = logging.getLogger(__name__)


class SkillEnvError(SkillError):
    """Raised when a skill's uv-managed environment cannot be built."""


_DEFAULT_CACHE_ROOT = Path.home() / ".cordbeat" / "skill-envs"


def _python_in(env_dir: Path) -> Path:
    if sys.platform == "win32":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _read_dependencies(pyproject: Path) -> list[str]:
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SkillEnvError(f"Cannot parse {pyproject}: {exc}") from exc
    project = data.get("project", {})
    deps = project.get("dependencies", [])
    if not isinstance(deps, list):
        raise SkillEnvError(f"{pyproject}: [project].dependencies must be a list")
    return [str(d) for d in deps]


class SkillEnvManager:
    """Lazily prepare and cache uv-managed Python envs for skills."""

    def __init__(self, cache_root: Path | None = None) -> None:
        self._cache_root = cache_root or _DEFAULT_CACHE_ROOT
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, skill_name: str) -> asyncio.Lock:
        lock = self._locks.get(skill_name)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[skill_name] = lock
        return lock

    def env_dir_for(self, skill_name: str) -> Path:
        return self._cache_root / skill_name

    async def prepare(self, skill_dir: Path, skill_name: str) -> str | None:
        """Ensure the skill's env exists; return python exe path or ``None``.

        ``None`` means the skill has no ``pyproject.toml`` and should be
        executed with the host Python.
        """
        pyproject = skill_dir / "pyproject.toml"
        if not pyproject.exists():
            return None

        env_dir = self.env_dir_for(skill_name)
        python_exe = _python_in(env_dir)
        hash_file = env_dir / ".cordbeat-skill-hash"
        current_hash = hashlib.sha256(pyproject.read_bytes()).hexdigest()

        async with self._lock_for(skill_name):
            need_install = (
                not python_exe.exists()
                or not hash_file.exists()
                or hash_file.read_text(encoding="utf-8").strip() != current_hash
            )
            if need_install:
                logger.info(
                    "Preparing uv env for skill '%s' at %s", skill_name, env_dir
                )
                deps = _read_dependencies(pyproject)
                await self._build(env_dir, python_exe, deps)
                env_dir.mkdir(parents=True, exist_ok=True)
                hash_file.write_text(current_hash, encoding="utf-8")
        return str(python_exe)

    async def _build(self, env_dir: Path, python_exe: Path, deps: list[str]) -> None:
        if shutil.which("uv") is None:
            raise SkillEnvError("uv is not available on PATH; cannot build skill env")
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        if env_dir.exists():
            shutil.rmtree(env_dir, ignore_errors=True)

        await self._run_uv("venv", str(env_dir), "--quiet")
        if deps:
            await self._run_uv(
                "pip",
                "install",
                "--python",
                str(python_exe),
                "--quiet",
                *deps,
            )

    @staticmethod
    async def _run_uv(*args: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "uv",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise SkillEnvError(f"uv {' '.join(args)} failed: {err}")
