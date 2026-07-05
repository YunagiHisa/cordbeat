"""Shared policy helpers for deciding how skills may run."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from cordbeat.models import SafetyLevel

from .validator import SkillValidationError, validate_skill_source

SANDBOX_SHARED_WORK_DIR = "shared"
SKILL_SETTINGS_FIELDS = {
    "ownership",
    "mutable_by_ai",
    "requires_approval_to_modify",
}
AI_SKILL_AUTHOR = "cordbeat-ai"
SYSTEM_SKILL_AUTHOR = "cordbeat"

_SANDBOX_LOCAL_FILE_PARAMS: dict[str, str] = {
    "file_read": "path",
    "file_write": "path",
    "file_mkdir": "path",
    "file_delete": "path",
    "file_search": "root",
}


@dataclass(frozen=True)
class SkillAccess:
    skill_name: str
    skill_dir: Path
    target_path: Path
    relative_path: str
    ownership: str
    mutable_by_ai: bool
    requires_approval_to_modify: bool
    is_settings_file: bool


def is_sandbox_relative_path(value: Any) -> bool:
    """Return true when a file parameter stays within CordBeat's sandbox."""
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1].strip()
    if not text:
        return False
    windows = PureWindowsPath(text)
    posix = PurePosixPath(text)
    if (
        windows.is_absolute()
        or posix.is_absolute()
        or windows.drive
        or windows.root
        or posix.root
    ):
        return False
    parts = tuple(windows.parts) + tuple(posix.parts)
    if any(part == ".." or part.startswith("~") for part in parts):
        return False
    return True


def default_skill_settings(author: str | None) -> dict[str, Any]:
    """Return ownership defaults for a skill author."""
    if author == AI_SKILL_AUTHOR:
        return {
            "ownership": "ai",
            "mutable_by_ai": True,
            "requires_approval_to_modify": False,
        }
    if author == SYSTEM_SKILL_AUTHOR:
        return {
            "ownership": "system",
            "mutable_by_ai": False,
            "requires_approval_to_modify": True,
        }
    return {
        "ownership": "user",
        "mutable_by_ai": False,
        "requires_approval_to_modify": True,
    }


def apply_default_skill_settings(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill missing skill ownership settings without overriding explicit values."""
    defaults = default_skill_settings(str(raw.get("author") or ""))
    for key, value in defaults.items():
        raw.setdefault(key, value)
    return raw


def _safe_skill_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text or not text.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid skill name: {text!r}")
    if "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError(f"Invalid skill name: {text!r}")
    return text


def _coerce_optional_bool(value: Any, *, field: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean for {field}: {value!r}")


def resolve_skill_file_access(
    skills_dir: Path,
    skill_name: Any,
    relative_path: Any,
) -> SkillAccess:
    """Resolve a file under a skill directory and read its current settings."""
    safe_name = _safe_skill_name(skill_name)
    rel_text = str(relative_path or "").strip()
    if not is_sandbox_relative_path(rel_text):
        raise ValueError(f"Invalid skill file path: {rel_text!r}")

    root = Path(skills_dir).expanduser().resolve()
    skill_dir = (root / safe_name).resolve()
    try:
        skill_dir.relative_to(root)
    except ValueError:
        raise ValueError(f"Skill path escapes skills directory: {safe_name}") from None
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"Skill not found: {safe_name}")

    target = (skill_dir / rel_text).resolve()
    try:
        target.relative_to(skill_dir)
    except ValueError:
        msg = f"Skill file path escapes skill directory: {rel_text}"
        raise ValueError(msg) from None

    yaml_path = skill_dir / "skill.yaml"
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    raw = apply_default_skill_settings(raw)

    return SkillAccess(
        skill_name=safe_name,
        skill_dir=skill_dir,
        target_path=target,
        relative_path=PurePosixPath(rel_text).as_posix(),
        ownership=str(raw.get("ownership") or "user"),
        mutable_by_ai=raw.get("mutable_by_ai") is True,
        requires_approval_to_modify=raw.get("requires_approval_to_modify") is not False,
        is_settings_file=PurePosixPath(rel_text).as_posix() == "skill.yaml",
    )


def skill_file_update_allowed(access: SkillAccess, *, approved: bool = False) -> bool:
    """Return whether AI may update this skill file without another approval."""
    if access.is_settings_file:
        return False
    if approved:
        return True
    return (
        access.ownership == "ai"
        and access.mutable_by_ai
        and not access.requires_approval_to_modify
    )


def skill_delete_allowed(access: SkillAccess, *, approved: bool = False) -> bool:
    """Return whether AI may delete this skill without another approval."""
    if approved:
        return True
    return (
        access.ownership == "ai"
        and access.mutable_by_ai
        and not access.requires_approval_to_modify
    )


def skill_file_delete_allowed(
    access: SkillAccess,
    *,
    approved: bool = False,
) -> bool:
    """Return whether AI may delete this skill file without another approval."""
    if access.is_settings_file:
        return False
    return skill_delete_allowed(access, approved=approved)


def read_skill_file(
    skills_dir: Path,
    *,
    skill_name: Any,
    path: Any,
    max_chars: int = 12000,
) -> dict[str, Any]:
    """Read a file from a skill directory after resolving it safely."""
    access = resolve_skill_file_access(skills_dir, skill_name, path)
    if not access.target_path.is_file():
        return {
            "error": f"File not found: {access.skill_name}/{access.relative_path}",
            "skill_name": access.skill_name,
            "path": access.relative_path,
        }
    try:
        content = access.target_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        error = f"Cannot read binary file: {access.skill_name}/{access.relative_path}"
        return {
            "error": error,
            "skill_name": access.skill_name,
            "path": access.relative_path,
        }
    max_len = max(1, int(max_chars))
    truncated = len(content) > max_len
    return {
        "skill_name": access.skill_name,
        "path": access.relative_path,
        "ownership": access.ownership,
        "mutable_by_ai": access.mutable_by_ai,
        "requires_approval_to_modify": access.requires_approval_to_modify,
        "truncated": truncated,
        "content": content[:max_len],
    }


def update_skill_file(
    skills_dir: Path,
    *,
    skill_name: Any,
    path: Any,
    content: Any,
    approved: bool = False,
) -> dict[str, Any]:
    """Update a non-settings skill file when the skill policy allows it."""
    access = resolve_skill_file_access(skills_dir, skill_name, path)
    if not skill_file_update_allowed(access, approved=approved):
        return {
            "error": "approval_required",
            "skill_name": access.skill_name,
            "path": access.relative_path,
            "ownership": access.ownership,
            "mutable_by_ai": access.mutable_by_ai,
            "requires_approval_to_modify": access.requires_approval_to_modify,
        }
    text = str(content or "")
    if access.target_path.suffix == ".py":
        yaml_path = access.skill_dir / "skill.yaml"
        try:
            raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        safety = raw.get("safety") or {}
        if not isinstance(safety, dict):
            safety = {}
        try:
            validate_skill_source(
                text,
                access.skill_name,
                allow_subprocess=safety.get("level") == SafetyLevel.DANGEROUS.value,
            )
        except SkillValidationError as exc:
            return {
                "error": "validation_failed",
                "detail": str(exc),
                "skill_name": access.skill_name,
                "path": access.relative_path,
            }
    access.target_path.parent.mkdir(parents=True, exist_ok=True)
    access.target_path.write_text(text, encoding="utf-8")
    return {
        "skill_name": access.skill_name,
        "path": access.relative_path,
        "bytes_written": len(text.encode("utf-8")),
        "status": "ok",
        "approved": approved,
    }


def delete_skill_file(
    skills_dir: Path,
    *,
    skill_name: Any,
    path: Any,
    recursive: Any = False,
    approved: bool = False,
) -> dict[str, Any]:
    """Delete a non-settings file or directory within a skill."""
    access = resolve_skill_file_access(skills_dir, skill_name, path)
    if access.target_path == access.skill_dir:
        return {
            "error": "use_delete_skill",
            "skill_name": access.skill_name,
            "path": access.relative_path,
        }
    if not skill_file_delete_allowed(access, approved=approved):
        return {
            "error": "approval_required",
            "skill_name": access.skill_name,
            "path": access.relative_path,
            "ownership": access.ownership,
            "mutable_by_ai": access.mutable_by_ai,
            "requires_approval_to_modify": access.requires_approval_to_modify,
        }
    if not access.target_path.exists():
        return {
            "error": f"File not found: {access.skill_name}/{access.relative_path}",
            "skill_name": access.skill_name,
            "path": access.relative_path,
        }
    recursive_bool = _coerce_optional_bool(recursive, field="recursive") is True
    if access.target_path.is_dir():
        if recursive_bool:
            shutil.rmtree(access.target_path)
            kind = "directory"
        else:
            access.target_path.rmdir()
            kind = "directory"
    else:
        access.target_path.unlink()
        kind = "file"
    return {
        "skill_name": access.skill_name,
        "path": access.relative_path,
        "kind": kind,
        "status": "ok",
        "approved": approved,
    }


def delete_skill(
    skills_dir: Path,
    *,
    skill_name: Any,
    approved: bool = False,
) -> dict[str, Any]:
    """Delete an installed skill directory when ownership policy allows it."""
    access = resolve_skill_file_access(skills_dir, skill_name, "skill.yaml")
    if not skill_delete_allowed(access, approved=approved):
        return {
            "error": "approval_required",
            "skill_name": access.skill_name,
            "ownership": access.ownership,
            "mutable_by_ai": access.mutable_by_ai,
            "requires_approval_to_modify": access.requires_approval_to_modify,
        }
    shutil.rmtree(access.skill_dir)
    return {
        "skill_name": access.skill_name,
        "status": "ok",
        "approved": approved,
    }


def update_skill_settings(
    skills_dir: Path,
    *,
    skill_name: Any,
    ownership: Any = None,
    mutable_by_ai: Any = None,
    requires_approval_to_modify: Any = None,
) -> dict[str, Any]:
    """Update guarded skill settings. Callers must require user approval."""
    access = resolve_skill_file_access(skills_dir, skill_name, "skill.yaml")
    yaml_path = access.skill_dir / "skill.yaml"
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid skill.yaml for {access.skill_name}")
    raw = apply_default_skill_settings(raw)

    if ownership is not None:
        owner = str(ownership).strip()
        if owner not in {"ai", "user", "system"}:
            raise ValueError(f"Invalid ownership: {owner!r}")
        raw["ownership"] = owner
    mutable = _coerce_optional_bool(mutable_by_ai, field="mutable_by_ai")
    if mutable is not None:
        raw["mutable_by_ai"] = mutable
    requires_approval = _coerce_optional_bool(
        requires_approval_to_modify,
        field="requires_approval_to_modify",
    )
    if requires_approval is not None:
        raw["requires_approval_to_modify"] = requires_approval

    yaml_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return {
        "skill_name": access.skill_name,
        "ownership": raw.get("ownership"),
        "mutable_by_ai": raw.get("mutable_by_ai"),
        "requires_approval_to_modify": raw.get("requires_approval_to_modify"),
        "status": "ok",
    }


def sandbox_overrides_for_skill(
    skill_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Return execution overrides for sandbox-local file tool calls."""
    path_param = _SANDBOX_LOCAL_FILE_PARAMS.get(skill_name)
    if path_param is None:
        return {}
    path_value = params.get(path_param)
    if skill_name == "file_search" and not str(path_value or "").strip():
        path_value = "."
    if is_sandbox_relative_path(path_value):
        return {
            "filesystem": False,
            "work_dir": SANDBOX_SHARED_WORK_DIR,
        }
    return {}


def skill_requires_confirmation(skill: Any, params: dict[str, Any]) -> bool:
    """Return whether a skill call must be approved before execution."""
    meta = skill.meta
    if sandbox_overrides_for_skill(meta.name, params).get("filesystem") is False:
        return False
    return meta.safety_level != SafetyLevel.SAFE or bool(meta.filesystem)
