"""SKILL system — pluggable action loader and executor."""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import importlib.util
import logging
import socket as _socket_module
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import yaml

from cordbeat.models import SafetyLevel, SkillContext, SkillMeta, SkillParam

logger = logging.getLogger(__name__)


class SkillPermissionError(Exception):
    """Raised when a skill violates its sandbox permissions."""


@contextlib.contextmanager
def _block_network() -> Generator[None, None, None]:
    """Block all network access by replacing ``socket.socket``.

    While the context manager is active any attempt to create a socket
    raises :class:`SkillPermissionError`.
    """
    original = _socket_module.socket

    def _blocked(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise SkillPermissionError("Network access is not allowed for this skill")

    _socket_module.socket = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        _socket_module.socket = original  # type: ignore[assignment]


@contextlib.contextmanager
def _restrict_filesystem(work_dir: Path) -> Generator[None, None, None]:
    """Restrict ``open()`` to paths inside *work_dir*.

    Any call to the built-in ``open`` with a path outside the resolved
    *work_dir* raises :class:`SkillPermissionError`.
    """
    resolved_root = work_dir.resolve()
    original_open = builtins.open

    def _restricted_open(
        file: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            target = Path(file).resolve()
        except (TypeError, ValueError):
            # Non-path argument (e.g. file descriptor int) — pass through
            return original_open(file, *args, **kwargs)
        if not target.is_relative_to(resolved_root):
            raise SkillPermissionError(
                f"Filesystem access outside work directory is not allowed: {target}"
            )
        return original_open(file, *args, **kwargs)

    builtins.open = _restricted_open  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.open = original_open  # type: ignore[assignment]


class Skill:
    """A loaded skill instance with metadata and execute capability."""

    def __init__(self, meta: SkillMeta, module: Any) -> None:
        self.meta = meta
        self._module = module

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the skill's main function with permission context."""
        fn = getattr(self._module, "execute", None)
        if fn is None:
            msg = f"Skill '{self.meta.name}' has no execute() function"
            raise RuntimeError(msg)

        context = self._build_context()

        # Inject context if the skill accepts it
        import inspect

        sig = inspect.signature(fn)
        if "context" in sig.parameters:
            params = {**params, "context": context}

        if self.meta.sandbox:
            return await self._execute_sandboxed(fn, params, context)

        result = fn(**params)
        if hasattr(result, "__await__"):
            result = await result
        if not isinstance(result, dict):
            return {"result": result}
        return result

    async def _execute_sandboxed(
        self,
        fn: Any,
        params: dict[str, Any],
        context: SkillContext,
    ) -> dict[str, Any]:
        """Execute skill in a sandboxed context with enforced permissions."""
        with tempfile.TemporaryDirectory(prefix="cordbeat_skill_") as tmpdir:
            context.work_dir = Path(tmpdir)

            guards: list[contextlib.AbstractContextManager[None]] = []
            if not self.meta.network:
                guards.append(_block_network())
            if not self.meta.filesystem:
                guards.append(_restrict_filesystem(context.work_dir))

            with contextlib.ExitStack() as stack:
                for guard in guards:
                    stack.enter_context(guard)
                result = fn(**params)
                if hasattr(result, "__await__"):
                    result = await result

            if not isinstance(result, dict):
                return {"result": result}
            return result

    def _build_context(self) -> SkillContext:
        return SkillContext(
            sandbox=self.meta.sandbox,
            network=self.meta.network,
            filesystem=self.meta.filesystem,
        )


class SkillRegistry:
    """Discovers, loads, and manages skills from the skills directory."""

    def __init__(self, skills_dir: str | Path) -> None:
        self._skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}

    @property
    def available_skills(self) -> dict[str, SkillMeta]:
        return {name: skill.meta for name, skill in self._skills.items()}

    @property
    def enabled_skill_names(self) -> set[str]:
        return {name for name, skill in self._skills.items() if skill.meta.enabled}

    def load_all(self) -> None:
        """Scan the skills directory and load all valid skills."""
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
        params_raw = raw.get("parameters", [])
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

        # Dangerous skills default to disabled
        if meta.safety_level == SafetyLevel.DANGEROUS and "enabled" not in raw:
            meta.enabled = False

        # Integrity check: verify main.py hash if declared in skill.yaml
        expected_hash = raw.get("integrity", {}).get("sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(main_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                msg = (
                    f"Skill '{meta.name}' integrity check failed: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
                raise ValueError(msg)

        # Load the Python module
        spec = importlib.util.spec_from_file_location(
            f"cordbeat_skill_{meta.name}",
            str(main_path),
        )
        if spec is None or spec.loader is None:
            msg = f"Cannot load module from {main_path}"
            raise ImportError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self._skills[meta.name] = Skill(meta=meta, module=module)
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
