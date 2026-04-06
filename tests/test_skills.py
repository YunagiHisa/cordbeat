"""Tests for skill registry."""

from __future__ import annotations

from pathlib import Path

from cordbeat.models import SafetyLevel
from cordbeat.skills import SkillRegistry


def _create_skill(
    skills_dir: Path,
    name: str,
    *,
    safety: str = "safe",
    enabled: bool | None = None,
) -> None:
    """Helper to create a minimal skill in the filesystem."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)

    yaml_content = (
        f"name: {name}\ndescription: Test skill\nsafety:\n  level: {safety}\n"
    )
    if enabled is not None:
        yaml_content += f"enabled: {str(enabled).lower()}\n"
    (skill_dir / "skill.yaml").write_text(yaml_content, encoding="utf-8")

    (skill_dir / "main.py").write_text(
        "def execute(**kwargs):\n    return {'result': 'ok'}\n",
        encoding="utf-8",
    )


class TestSkillRegistry:
    def test_load_skills(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "greet")
        _create_skill(skills_dir, "search")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        assert len(registry.available_skills) == 2
        assert "greet" in registry.available_skills
        assert "search" in registry.available_skills

    def test_empty_dir(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        assert len(registry.available_skills) == 0

    def test_missing_dir(self, tmp_path: Path) -> None:
        registry = SkillRegistry(tmp_path / "nonexistent")
        registry.load_all()
        assert len(registry.available_skills) == 0

    def test_dangerous_skill_disabled_by_default(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "danger", safety="dangerous")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        meta = registry.available_skills["danger"]
        assert meta.safety_level == SafetyLevel.DANGEROUS
        assert meta.enabled is False

    def test_dangerous_skill_explicit_enable(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "danger", safety="dangerous", enabled=True)

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        assert registry.available_skills["danger"].enabled is True

    def test_get_safe_skills(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "safe_one", safety="safe")
        _create_skill(skills_dir, "risky", safety="dangerous")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        safe = registry.get_safe_skills()
        safe_names = [s.name for s in safe]
        assert "safe_one" in safe_names
        assert "risky" not in safe_names

    async def test_execute_skill(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "echo")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("echo")
        assert skill is not None
        result = await skill.execute({})
        assert result == {"result": "ok"}

    def test_get_nonexistent_skill(self, tmp_path: Path) -> None:
        registry = SkillRegistry(tmp_path / "skills")
        assert registry.get("nope") is None
