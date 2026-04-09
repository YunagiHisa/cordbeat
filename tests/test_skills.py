"""Tests for skill registry."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from cordbeat.models import SafetyLevel
from cordbeat.skills import SkillRegistry


def _create_skill(
    skills_dir: Path,
    name: str,
    *,
    safety: str = "safe",
    enabled: bool | None = None,
    sandbox: bool = False,
    network: bool = False,
    filesystem: bool = False,
    main_code: str | None = None,
) -> None:
    """Helper to create a minimal skill in the filesystem."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)

    yaml_content = (
        f"name: {name}\ndescription: Test skill\n"
        f"safety:\n  level: {safety}\n"
        f"  sandbox: {str(sandbox).lower()}\n"
        f"  network: {str(network).lower()}\n"
        f"  filesystem: {str(filesystem).lower()}\n"
    )
    if enabled is not None:
        yaml_content += f"enabled: {str(enabled).lower()}\n"
    (skill_dir / "skill.yaml").write_text(yaml_content, encoding="utf-8")

    code = main_code or "def execute(**kwargs):\n    return {'result': 'ok'}\n"
    (skill_dir / "main.py").write_text(code, encoding="utf-8")


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

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()

        # Create a skill with a symlink pointing outside the skills directory
        evil_dir = tmp_path / "evil"
        evil_dir.mkdir()
        (evil_dir / "skill.yaml").write_text(
            "name: evil\ndescription: bad\nsafety:\n  level: safe\n",
            encoding="utf-8",
        )
        (evil_dir / "main.py").write_text(
            "def execute(**kwargs):\n    return {'pwned': True}\n",
            encoding="utf-8",
        )

        link_path = skills_dir / "evil"
        try:
            os.symlink(evil_dir, link_path)
        except OSError:
            pytest.skip("Cannot create symlinks on this system")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        # The evil skill should not be loaded due to path traversal check
        assert registry.get("evil") is None

    def test_integrity_check_passes(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "verified"
        skill_dir.mkdir(parents=True)

        main_content = b"def execute(**kwargs):\n    return {'result': 'ok'}\n"
        sha256 = hashlib.sha256(main_content).hexdigest()

        (skill_dir / "skill.yaml").write_text(
            f"name: verified\ndescription: Test\nintegrity:\n  sha256: {sha256}\n",
            encoding="utf-8",
        )
        (skill_dir / "main.py").write_bytes(main_content)

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        assert registry.get("verified") is not None

    def test_integrity_check_rejects_tampered(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skill_dir = skills_dir / "tampered"
        skill_dir.mkdir(parents=True)

        fake_hash = "deadbeef" + "0" * 56
        (skill_dir / "skill.yaml").write_text(
            f"name: tampered\ndescription: Test\nintegrity:\n  sha256: {fake_hash}\n",
            encoding="utf-8",
        )
        (skill_dir / "main.py").write_text(
            "def execute(**kwargs):\n    return {'result': 'ok'}\n",
            encoding="utf-8",
        )

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        # Tampered skill should not be loaded
        assert registry.get("tampered") is None

    def test_no_integrity_field_still_loads(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "nocheck")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        assert registry.get("nocheck") is not None


class TestSkillSandbox:
    async def test_sandbox_skill_gets_temp_work_dir(self, tmp_path: Path) -> None:
        """Sandboxed skills should receive a temporary work directory."""
        code = (
            "def execute(context=None, **kwargs):\n"
            "    return {'has_context': context is not None,\n"
            "            'has_work_dir': context.work_dir is not None,\n"
            "            'sandbox': context.sandbox}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "sandboxed", sandbox=True, main_code=code)

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("sandboxed")
        assert skill is not None
        result = await skill.execute({})
        assert result["has_context"] is True
        assert result["has_work_dir"] is True
        assert result["sandbox"] is True

    async def test_non_sandbox_skill_no_work_dir(self, tmp_path: Path) -> None:
        """Non-sandboxed skills should not get a work_dir."""
        code = (
            "def execute(context=None, **kwargs):\n"
            "    return {'has_context': context is not None,\n"
            "            'work_dir_is_none': context.work_dir is None}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "normal", sandbox=False, main_code=code)

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("normal")
        assert skill is not None
        result = await skill.execute({})
        assert result["has_context"] is True
        assert result["work_dir_is_none"] is True

    async def test_skill_without_context_param(self, tmp_path: Path) -> None:
        """Skills that don't accept context should still work."""
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "simple")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("simple")
        assert skill is not None
        result = await skill.execute({})
        assert result == {"result": "ok"}

    async def test_sandbox_permissions_passed(self, tmp_path: Path) -> None:
        """Network and filesystem flags should be passed in context."""
        code = (
            "def execute(context=None, **kwargs):\n"
            "    return {'network': context.network,\n"
            "            'filesystem': context.filesystem}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(
            skills_dir, "net_skill", sandbox=True, network=True, main_code=code
        )

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("net_skill")
        assert skill is not None
        result = await skill.execute({})
        assert result["network"] is True
        assert result["filesystem"] is False
