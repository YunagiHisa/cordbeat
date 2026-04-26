"""SKILL system — subprocess-isolated action loader and executor.

Skills are loaded purely from their YAML metadata; their Python source
is validated with :mod:`cordbeat.skill_validator` but **never imported
into the parent process**. All execution happens in a subprocess via
:mod:`cordbeat.skill_sandbox`, providing strong isolation regardless
of whether the skill code is trusted (built-in) or AI-proposed.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml

from cordbeat.metrics import (
    SKILL_EXEC_LATENCY,
    SKILL_EXEC_TOTAL,
    inc_counter,
    time_block,
)
from cordbeat.models import SafetyLevel, SkillMeta, SkillParam
from cordbeat.skill_env import SkillEnvManager
from cordbeat.skill_sandbox import (
    DEFAULT_CONFIG,
    SandboxConfig,
    SkillPermissionError,
    SkillSandboxError,
    run_skill_in_subprocess,
)
from cordbeat.skill_validator import validate_skill_source

logger = logging.getLogger(__name__)

__all__ = [
    "Skill",
    "SkillRegistry",
    "SkillEnvManager",
    "SkillPermissionError",
    "SkillSandboxError",
]


class Skill:
    """A loaded skill instance with metadata and subprocess execution."""

    def __init__(
        self,
        meta: SkillMeta,
        skill_dir: Path | None = None,
        sandbox_config: SandboxConfig | None = None,
        env_manager: SkillEnvManager | None = None,
        *,
        _test_callable: Any = None,
    ) -> None:
        self.meta = meta
        self._skill_dir = skill_dir
        self._sandbox_config = sandbox_config or DEFAULT_CONFIG
        self._env_manager = env_manager
        self._test_callable = _test_callable

    async def execute(
        self,
        params: dict[str, Any],
        memory: Any = None,
    ) -> dict[str, Any]:
        """Execute the skill in an isolated subprocess.

        ``memory`` is passed through to the sandbox only if provided;
        the subprocess may then request whitelisted memory calls which
        are executed against this object.
        """
        labels = {
            "skill": self.meta.name,
            "safety_level": self.meta.safety_level.value,
        }
        try:
            async with time_block(SKILL_EXEC_LATENCY, labels):
                result = await self._execute_inner(params, memory)
        except Exception:
            inc_counter(SKILL_EXEC_TOTAL, {**labels, "outcome": "error"})
            raise
        inc_counter(SKILL_EXEC_TOTAL, {**labels, "outcome": "ok"})
        return result

    async def _execute_inner(
        self,
        params: dict[str, Any],
        memory: Any,
    ) -> dict[str, Any]:
        if self._test_callable is not None:
            # Test-only in-process path. Never taken for skills loaded
            # from disk via SkillRegistry.
            import inspect  # noqa: PLC0415

            call_kwargs = dict(params)
            sig = inspect.signature(self._test_callable)
            if "context" in sig.parameters:
                from types import SimpleNamespace  # noqa: PLC0415

                call_kwargs["context"] = SimpleNamespace(
                    sandbox=self.meta.sandbox,
                    network=self.meta.network,
                    filesystem=self.meta.filesystem,
                    work_dir=None,
                    memory=memory,
                )
            result = self._test_callable(**call_kwargs)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, dict):
                return {"result": result}
            return result

        if self._skill_dir is None:
            raise RuntimeError(
                f"Skill {self.meta.name!r} has no skill_dir; cannot execute."
            )

        with tempfile.TemporaryDirectory(prefix="cordbeat_skill_") as tmpdir:
            sandbox_params = {
                "network": self.meta.network,
                "filesystem": self.meta.filesystem,
                "work_dir": str(tmpdir),
            }
            python_executable: str | None = None
            if self._env_manager is not None:
                python_executable = await self._env_manager.prepare(
                    self._skill_dir, self.meta.name
                )
            return await run_skill_in_subprocess(
                skill_dir=self._skill_dir,
                skill_name=self.meta.name,
                params=params,
                sandbox=sandbox_params,
                memory=memory,
                config=self._sandbox_config,
                python_executable=python_executable,
            )


class SkillRegistry:
    """Discovers and validates skills from the skills directory.

    Skills are loaded by parsing their ``skill.yaml`` metadata and
    statically validating ``main.py`` via
    :func:`cordbeat.skill_validator.validate_skill_source`. The source
    is never imported into the parent process.
    """

    def __init__(
        self,
        skills_dir: str | Path,
        sandbox_config: SandboxConfig | None = None,
        env_manager: SkillEnvManager | None = None,
    ) -> None:
        self._skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}
        self._sandbox_config = sandbox_config or DEFAULT_CONFIG
        self._env_manager = (
            env_manager if env_manager is not None else SkillEnvManager()
        )

    @property
    def skills_dir(self) -> Path:
        return self._skills_dir

    @property
    def available_skills(self) -> dict[str, SkillMeta]:
        return {name: skill.meta for name, skill in self._skills.items()}

    @property
    def enabled_skill_names(self) -> set[str]:
        return {name for name, skill in self._skills.items() if skill.meta.enabled}

    def load_all(self) -> None:
        """Scan the skills directory and load all valid skills."""
        self._skills.clear()
        if not self._skills_dir.exists():
            logger.warning("Skills directory not found: %s", self._skills_dir)
            return

        for skill_path in self._skills_dir.iterdir():
            if not skill_path.is_dir():
                continue
            yaml_path = skill_path / "skill.yaml"
            main_path = skill_path / "main.py"
            if not yaml_path.exists() or not main_path.exists():
                logger.debug("Skipping %s (missing skill.yaml or main.py)", skill_path)
                continue
            try:
                self._load_skill(skill_path)
            except Exception:
                logger.exception("Failed to load skill from %s", skill_path)

    def _load_skill(self, skill_path: Path) -> None:
        yaml_path = skill_path / "skill.yaml"
        main_path = skill_path / "main.py"

        # Security: ensure skill files are inside the skills directory
        try:
            main_path.resolve().relative_to(self._skills_dir.resolve())
        except ValueError:
            msg = f"Skill path escapes skills directory: {main_path}"
            raise PermissionError(msg) from None

        with yaml_path.open(encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        safety_raw = raw.get("safety", {})
        params_raw = raw.get("parameters") or []
        parameters = [
            SkillParam(
                name=p["name"],
                type=p.get("type", "string"),
                required=p.get("required", True),
                description=p.get("description", ""),
            )
            for p in params_raw
            if isinstance(p, dict) and "name" in p
        ]

        meta = SkillMeta(
            name=raw.get("name", skill_path.name),
            description=raw.get("description", ""),
            usage=raw.get("usage", ""),
            parameters=parameters,
            safety_level=SafetyLevel(safety_raw.get("level", "safe")),
            sandbox=safety_raw.get("sandbox", False),
            network=safety_raw.get("network", False),
            filesystem=safety_raw.get("filesystem", False),
            enabled=raw.get("enabled", True),
        )

        if meta.safety_level == SafetyLevel.DANGEROUS and "enabled" not in raw:
            meta.enabled = False

        # Integrity check: verify main.py hash if declared in skill.yaml
        source_bytes = main_path.read_bytes()
        expected_hash = raw.get("integrity", {}).get("sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(source_bytes).hexdigest()
            if actual_hash != expected_hash:
                msg = (
                    f"Skill '{meta.name}' integrity check failed: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
                raise ValueError(msg)

        # Static AST validation — protects the subprocess from trivially
        # malicious code and provides a fail-fast signal at load time.
        source = source_bytes.decode("utf-8", errors="replace")
        validate_skill_source(source, meta.name)

        self._skills[meta.name] = Skill(
            meta=meta,
            skill_dir=skill_path,
            sandbox_config=self._sandbox_config,
            env_manager=self._env_manager,
        )
        logger.info(
            "Loaded skill: %s (safety=%s, enabled=%s)",
            meta.name,
            meta.safety_level.value,
            meta.enabled,
        )

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def get_safe_skills(self) -> list[SkillMeta]:
        return [
            s.meta
            for s in self._skills.values()
            if s.meta.enabled and s.meta.safety_level == SafetyLevel.SAFE
        ]

    def get_skill_descriptions_for_prompt(self) -> str:
        """Build a skill catalog string for AI prompts."""
        lines = []
        for name, skill in self._skills.items():
            if not skill.meta.enabled:
                continue
            params_str = ", ".join(f"{p.name}: {p.type}" for p in skill.meta.parameters)
            lines.append(
                f"- {name}: {skill.meta.description} "
                f"(safety={skill.meta.safety_level.value}, params=[{params_str}])"
            )
        return "\n".join(lines) if lines else "(no skills available)"
