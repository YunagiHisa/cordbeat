"""Tests for skill registry."""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from cordbeat.models import SafetyLevel
from cordbeat.skills import SkillPermissionError, SkillRegistry


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


class TestSkillDescriptions:
    def test_get_descriptions_for_prompt(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "greet", safety="safe")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        desc = registry.get_skill_descriptions_for_prompt()
        assert "greet" in desc
        assert "safe" in desc

    def test_no_skills_available(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        desc = registry.get_skill_descriptions_for_prompt()
        assert desc == "(no skills available)"

    def test_disabled_skills_excluded(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "off", enabled=False)
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        desc = registry.get_skill_descriptions_for_prompt()
        assert desc == "(no skills available)"


class TestSkillExecution:
    async def test_skill_no_execute_function(self, tmp_path: Path) -> None:
        """Skill module without execute() raises RuntimeError."""
        code = "def nope():\n    pass\n"
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "broken", main_code=code)
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("broken")
        assert skill is not None
        with pytest.raises(RuntimeError, match="no execute"):
            await skill.execute({})

    async def test_async_skill(self, tmp_path: Path) -> None:
        """Async execute functions should be awaited."""
        code = "async def execute(**kwargs):\n    return {'async': True}\n"
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "async_skill", main_code=code)
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("async_skill")
        assert skill is not None
        result = await skill.execute({})
        assert result == {"async": True}

    async def test_non_dict_result_wrapped(self, tmp_path: Path) -> None:
        """Non-dict return values are wrapped in {'result': value}."""
        code = "def execute(**kwargs):\n    return 42\n"
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "num_skill", main_code=code)
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("num_skill")
        assert skill is not None
        result = await skill.execute({})
        assert result == {"result": 42}

    async def test_sandboxed_async_skill(self, tmp_path: Path) -> None:
        """Async sandboxed skill is properly awaited."""
        code = "async def execute(**kwargs):\n    return {'sandboxed_async': True}\n"
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "sb_async", sandbox=True, main_code=code)
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("sb_async")
        assert skill is not None
        result = await skill.execute({})
        assert result == {"sandboxed_async": True}

    async def test_sandboxed_non_dict_wrapped(self, tmp_path: Path) -> None:
        """Non-dict from sandboxed skill is wrapped."""
        code = "def execute(**kwargs):\n    return 'text'\n"
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "sb_text", sandbox=True, main_code=code)
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("sb_text")
        assert skill is not None
        result = await skill.execute({})
        assert result == {"result": "text"}

    def test_enabled_skill_names(self, tmp_path: Path) -> None:
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "active", enabled=True)
        _create_skill(skills_dir, "inactive", enabled=False)
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        names = registry.enabled_skill_names
        assert "active" in names
        assert "inactive" not in names

    def test_skip_dir_without_yaml(self, tmp_path: Path) -> None:
        """Directory without skill.yaml is skipped."""
        skills_dir = tmp_path / "skills"
        bad_dir = skills_dir / "no_yaml"
        bad_dir.mkdir(parents=True)
        (bad_dir / "main.py").write_text("pass\n", encoding="utf-8")
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        assert registry.get("no_yaml") is None


class TestSandboxEnforcement:
    """Tests that sandbox network/filesystem restrictions are actually enforced."""

    async def test_network_blocked_in_sandbox(self, tmp_path: Path) -> None:
        """Sandboxed skill with network=False cannot create sockets."""
        code = (
            "import socket\n"
            "def execute(**kwargs):\n"
            "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "    s.close()\n"
            "    return {'result': 'connected'}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(
            skills_dir, "net_blocked", sandbox=True, network=False, main_code=code
        )
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("net_blocked")
        assert skill is not None
        with pytest.raises(SkillPermissionError, match="Network access"):
            await skill.execute({})

    async def test_network_allowed_in_sandbox(self, tmp_path: Path) -> None:
        """Sandboxed skill with network=True can create sockets."""
        code = (
            "import socket\n"
            "def execute(**kwargs):\n"
            "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "    s.close()\n"
            "    return {'result': 'connected'}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(
            skills_dir, "net_allowed", sandbox=True, network=True, main_code=code
        )
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("net_allowed")
        assert skill is not None
        result = await skill.execute({})
        assert result["result"] == "connected"

    async def test_filesystem_blocked_outside_workdir(self, tmp_path: Path) -> None:
        """Sandboxed skill with filesystem=False cannot write outside work_dir."""
        outside_path = str(tmp_path / "outside.txt").replace("\\", "\\\\")
        code = (
            f"def execute(**kwargs):\n"
            f"    with open('{outside_path}', 'w') as f:\n"
            f"        f.write('escape')\n"
            f"    return {{'result': 'wrote'}}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(
            skills_dir,
            "fs_blocked",
            sandbox=True,
            filesystem=False,
            main_code=code,
        )
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("fs_blocked")
        assert skill is not None
        with pytest.raises(SkillPermissionError, match="Filesystem access"):
            await skill.execute({})

    async def test_filesystem_allowed_inside_workdir(self, tmp_path: Path) -> None:
        """Sandboxed skill with filesystem=False CAN write inside work_dir."""
        code = (
            "from pathlib import Path\n"
            "def execute(context=None, **kwargs):\n"
            "    out = context.work_dir / 'output.txt'\n"
            "    with open(str(out), 'w') as f:\n"
            "        f.write('hello')\n"
            "    return {'result': 'wrote'}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(
            skills_dir,
            "fs_workdir",
            sandbox=True,
            filesystem=False,
            main_code=code,
        )
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("fs_workdir")
        assert skill is not None
        result = await skill.execute({})
        assert result["result"] == "wrote"

    async def test_filesystem_allowed_when_flag_true(self, tmp_path: Path) -> None:
        """Sandboxed skill with filesystem=True can write anywhere."""
        outside = tmp_path / "output.txt"
        outside_path = str(outside).replace("\\", "\\\\")
        code = (
            f"def execute(**kwargs):\n"
            f"    with open('{outside_path}', 'w') as f:\n"
            f"        f.write('allowed')\n"
            f"    return {{'result': 'wrote'}}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(
            skills_dir,
            "fs_allowed",
            sandbox=True,
            filesystem=True,
            main_code=code,
        )
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("fs_allowed")
        assert skill is not None
        result = await skill.execute({})
        assert result["result"] == "wrote"
        assert outside.read_text() == "allowed"

    async def test_non_sandboxed_not_restricted(self, tmp_path: Path) -> None:
        """Non-sandboxed skills are not restricted even with network=False."""
        code = (
            "import socket\n"
            "def execute(**kwargs):\n"
            "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "    s.close()\n"
            "    return {'result': 'ok'}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(
            skills_dir, "unrestricted", sandbox=False, network=False, main_code=code
        )
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("unrestricted")
        assert skill is not None
        # Non-sandboxed: enforcement is not applied
        result = await skill.execute({})
        assert result["result"] == "ok"

    async def test_network_guard_restores_socket(self, tmp_path: Path) -> None:
        """After sandbox execution, socket module is restored."""
        code = (
            "import socket\n"
            "def execute(**kwargs):\n"
            "    try:\n"
            "        socket.socket()\n"
            "    except Exception:\n"
            "        pass\n"
            "    return {'result': 'done'}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(
            skills_dir, "guard_test", sandbox=True, network=False, main_code=code
        )
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("guard_test")
        assert skill is not None

        # The skill catches the error internally, so it succeeds
        await skill.execute({})

        # After execution, socket should work normally
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.close()

    async def test_both_network_and_filesystem_blocked(self, tmp_path: Path) -> None:
        """Both network and filesystem can be blocked simultaneously."""
        code = (
            "import socket\n"
            "def execute(**kwargs):\n"
            "    socket.socket()\n"
            "    return {'result': 'fail'}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(
            skills_dir,
            "both_blocked",
            sandbox=True,
            network=False,
            filesystem=False,
            main_code=code,
        )
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("both_blocked")
        assert skill is not None
        with pytest.raises(SkillPermissionError):
            await skill.execute({})


# ── Memory Injection ──────────────────────────────────────────────────


class TestMemoryInjection:
    """Tests for memory parameter injection into SkillContext."""

    async def test_memory_passed_to_context(self, tmp_path: Path) -> None:
        """Memory object is accessible via context.memory inside the skill."""
        code = (
            "def execute(context=None, **kwargs):\n"
            "    return {'has_memory': context.memory is not None,\n"
            "            'memory_type': type(context.memory).__name__}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "mem_skill", main_code=code)
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("mem_skill")
        assert skill is not None

        mock_memory = MagicMock()
        result = await skill.execute({}, memory=mock_memory)
        assert result["has_memory"] is True
        assert result["memory_type"] == "MagicMock"

    async def test_memory_none_by_default(self, tmp_path: Path) -> None:
        """When memory is not passed, context.memory is None."""
        code = (
            "def execute(context=None, **kwargs):\n"
            "    return {'memory_is_none': context.memory is None}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "no_mem", main_code=code)
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("no_mem")
        assert skill is not None

        result = await skill.execute({})
        assert result["memory_is_none"] is True

    async def test_memory_in_sandboxed_skill(self, tmp_path: Path) -> None:
        """Memory is accessible even in sandboxed skills."""
        code = (
            "def execute(context=None, **kwargs):\n"
            "    return {'has_memory': context.memory is not None}\n"
        )
        skills_dir = tmp_path / "skills"
        _create_skill(skills_dir, "sb_mem", sandbox=True, main_code=code)
        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("sb_mem")
        assert skill is not None

        mock_memory = MagicMock()
        result = await skill.execute({}, memory=mock_memory)
        assert result["has_memory"] is True


# ── Built-in Skills ──────────────────────────────────────────────────

# Path to project-level skills directory
_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _copy_builtin_skill(dest_dir: Path, name: str) -> None:
    """Copy a real built-in skill to a temporary skills directory."""
    src = _BUILTIN_SKILLS_DIR / name
    dst = dest_dir / name
    shutil.copytree(src, dst)


class TestReadDiarySkill:
    async def test_read_diary_returns_entries(self, tmp_path: Path) -> None:
        """read_diary skill fetches records from memory."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _copy_builtin_skill(skills_dir, "read_diary")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("read_diary")
        assert skill is not None

        mock_memory = AsyncMock()
        mock_memory.get_certain_records.return_value = [
            {"content": "Today was good", "created_at": "2025-01-01T00:00:00"},
            {"content": "Rainy day", "created_at": "2025-01-02T00:00:00"},
        ]

        result = await skill.execute(
            {"user_id": "u1", "record_type": "diary", "limit": 5},
            memory=mock_memory,
        )
        assert result["count"] == 2
        assert result["user_id"] == "u1"
        assert result["record_type"] == "diary"
        assert result["entries"][0]["content"] == "Today was good"
        mock_memory.get_certain_records.assert_awaited_once_with(
            "u1", record_type="diary", limit=5
        )

    async def test_read_diary_no_memory(self, tmp_path: Path) -> None:
        """read_diary returns error when memory is not available."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _copy_builtin_skill(skills_dir, "read_diary")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("read_diary")
        assert skill is not None

        result = await skill.execute({"user_id": "u1"})
        assert "error" in result

    async def test_read_diary_empty_records(self, tmp_path: Path) -> None:
        """read_diary handles empty records list."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _copy_builtin_skill(skills_dir, "read_diary")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("read_diary")
        assert skill is not None

        mock_memory = AsyncMock()
        mock_memory.get_certain_records.return_value = []

        result = await skill.execute(
            {"user_id": "u1"},
            memory=mock_memory,
        )
        assert result["count"] == 0
        assert result["entries"] == []


class TestTimerSkill:
    async def test_timer_schedules_reminder(self, tmp_path: Path) -> None:
        """timer skill stores a reminder and returns schedule info."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _copy_builtin_skill(skills_dir, "timer")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("timer")
        assert skill is not None

        mock_memory = AsyncMock()
        mock_memory.add_certain_record.return_value = "rec-001"

        result = await skill.execute(
            {"user_id": "u1", "message": "Check oven", "minutes": 15},
            memory=mock_memory,
        )
        assert result["status"] == "scheduled"
        assert result["record_id"] == "rec-001"
        assert result["message"] == "Check oven"
        assert "remind_at" in result

        # Verify memory was called correctly
        mock_memory.add_certain_record.assert_awaited_once()
        call_kwargs = mock_memory.add_certain_record.call_args.kwargs
        assert call_kwargs["user_id"] == "u1"
        assert call_kwargs["content"] == "Check oven"
        assert call_kwargs["record_type"] == "reminder"
        assert call_kwargs["metadata"]["status"] == "pending"
        assert "remind_at" in call_kwargs["metadata"]

    async def test_timer_no_memory(self, tmp_path: Path) -> None:
        """timer returns error when memory is not available."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _copy_builtin_skill(skills_dir, "timer")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("timer")
        assert skill is not None

        result = await skill.execute({"user_id": "u1", "message": "test"})
        assert "error" in result


class TestFileReadSkill:
    async def test_file_read_returns_content(self, tmp_path: Path) -> None:
        """file_read skill reads a file and returns its content."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _copy_builtin_skill(skills_dir, "file_read")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        skill = registry.get("file_read")
        assert skill is not None

        # Create a test file inside work_dir (sandbox)
        test_file = tmp_path / "sandbox" / "data.txt"
        test_file.parent.mkdir()
        test_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

        # file_read is sandboxed with filesystem=true, so the sandbox will
        # restrict open() to work_dir. We pass the path as a param.
        # The sandbox work_dir is a tempdir created by the executor, so
        # we need to place the file inside it. Instead, call in non-sandboxed
        # mode by loading a version without sandbox for unit testing.
        # Actually, file_read IS sandboxed — let's test the function directly.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "test_file_read",
            str(_BUILTIN_SKILLS_DIR / "file_read" / "main.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.execute(path=str(test_file), max_lines=100)
        assert result["lines"] == 3
        assert result["truncated"] is False
        assert "line1" in result["content"]

    async def test_file_read_truncation(self, tmp_path: Path) -> None:
        """file_read truncates output when exceeding max_lines."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "test_file_read_trunc",
            str(_BUILTIN_SKILLS_DIR / "file_read" / "main.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        test_file = tmp_path / "big.txt"
        test_file.write_text("\n".join(f"line{i}" for i in range(50)), encoding="utf-8")

        result = mod.execute(path=str(test_file), max_lines=10)
        assert result["lines"] == 10
        assert result["total_lines"] == 50
        assert result["truncated"] is True

    async def test_file_read_missing_file(self, tmp_path: Path) -> None:
        """file_read returns error for missing files."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "test_file_read_missing",
            str(_BUILTIN_SKILLS_DIR / "file_read" / "main.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.execute(path=str(tmp_path / "nonexistent.txt"))
        assert "error" in result

    async def test_file_read_sandbox_enforced(self, tmp_path: Path) -> None:
        """file_read skill is loaded with sandbox=True."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _copy_builtin_skill(skills_dir, "file_read")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        meta = registry.available_skills["file_read"]
        assert meta.sandbox is True
        assert meta.filesystem is True
        assert meta.network is False


class TestShellExecSkill:
    def test_shell_exec_disabled_by_default(self, tmp_path: Path) -> None:
        """shell_exec is loaded as dangerous and disabled."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _copy_builtin_skill(skills_dir, "shell_exec")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        meta = registry.available_skills["shell_exec"]
        assert meta.safety_level == SafetyLevel.DANGEROUS
        assert meta.enabled is False

    def test_shell_exec_excluded_from_prompt(self, tmp_path: Path) -> None:
        """Disabled shell_exec should not appear in prompt descriptions."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        _copy_builtin_skill(skills_dir, "shell_exec")

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        desc = registry.get_skill_descriptions_for_prompt()
        assert "shell_exec" not in desc

    async def test_shell_exec_runs_command(self, tmp_path: Path) -> None:
        """shell_exec can execute commands when loaded."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "test_shell_exec",
            str(_BUILTIN_SKILLS_DIR / "shell_exec" / "main.py"),
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = await mod.execute(command="echo hello", timeout=10)
        assert result["returncode"] == 0
        assert "hello" in result["stdout"]


class TestBuiltinSkillRegistry:
    """Test loading all built-in skills together."""

    def test_load_all_builtins(self, tmp_path: Path) -> None:
        """All 4 built-in skills can be loaded at once."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        for name in ("read_diary", "timer", "file_read", "shell_exec"):
            _copy_builtin_skill(skills_dir, name)

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        assert len(registry.available_skills) == 4

    def test_enabled_builtins(self, tmp_path: Path) -> None:
        """Only safe skills are enabled by default."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        for name in ("read_diary", "timer", "file_read", "shell_exec"):
            _copy_builtin_skill(skills_dir, name)

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        enabled = registry.enabled_skill_names
        assert "read_diary" in enabled
        assert "timer" in enabled
        assert "file_read" in enabled
        assert "shell_exec" not in enabled

    def test_prompt_includes_safe_builtins(self, tmp_path: Path) -> None:
        """Prompt descriptions include enabled built-in skills."""
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        for name in ("read_diary", "timer", "file_read", "shell_exec"):
            _copy_builtin_skill(skills_dir, name)

        registry = SkillRegistry(skills_dir)
        registry.load_all()
        desc = registry.get_skill_descriptions_for_prompt()
        assert "read_diary" in desc
        assert "timer" in desc
        assert "file_read" in desc
        assert "shell_exec" not in desc
