"""SKILL system — pluggable action loader and executor."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import yaml

from cordbeat.models import SafetyLevel, SkillMeta, SkillParam

logger = logging.getLogger(__name__)


class Skill:
    """A loaded skill instance with metadata and execute capability."""

    def __init__(self, meta: SkillMeta, module: Any) -> None:
        self.meta = meta
        self._module = module

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the skill's main function."""
        fn = getattr(self._module, "execute", None)
        if fn is None:
            msg = f"Skill '{self.meta.name}' has no execute() function"
            raise RuntimeError(msg)
        result = fn(**params)
        # Support both sync and async execute functions
        if hasattr(result, "__await__"):
            result = await result
        return dict(result)


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
