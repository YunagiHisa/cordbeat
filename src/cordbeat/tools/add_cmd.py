"""cordbeat-add: interactive tool to add adapters or skills to an existing setup."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

from cordbeat.config import cordbeat_home
from cordbeat.tools.wizard import (
    _ADAPTER_SPECS,
    _ADAPTER_TOKEN_PROMPT,
    _BOLD,
    _GREEN,
    _PINK,
    _RED,  # noqa: F401  (imported for re-export)
    _YELLOW,  # noqa: F401
    _ask,
    _c,
    _err,
    _install_packages,
    _is_importable,
    _ok,
    _warn,
)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _find_config() -> Path | None:
    """Find the active config.yaml (current dir first, then ~/.cordbeat/)."""
    candidates = [
        Path("config.yaml"),
        cordbeat_home() / "config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def _update_env_file(env_path: Path, key: str, value: str) -> None:
    """Add or update a key=value pair in a .env file."""
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    key_prefix = f"{key}="
    found = False
    for i, line in enumerate(lines):
        if line.startswith(key_prefix):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")

    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Adapter flow
# ---------------------------------------------------------------------------


def _add_adapter(config_path: Path) -> None:
    cfg = _load_yaml(config_path)
    adapters: dict[str, Any] = cfg.get("adapters", {})

    enabled = [
        k for k, v in adapters.items() if isinstance(v, dict) and v.get("enabled")
    ]
    if enabled:
        print(f"\n  Currently enabled: {_c(', '.join(enabled), _GREEN)}")

    print("\n  Which adapter would you like to add?")
    for i, (key, name, _, _) in enumerate(_ADAPTER_SPECS, 1):
        status = _c(" (enabled)", _GREEN) if key in enabled else ""
        print(f"    {i}. {name}{status}")

    raw = _ask("Choice", "2")
    try:
        idx = int(raw) - 1
        if not 0 <= idx < len(_ADAPTER_SPECS):
            idx = 1
    except ValueError:
        idx = 1

    key, name, packages, probe = _ADAPTER_SPECS[idx]

    if key == "cli":
        _warn("CLI adapter is always enabled — nothing to add.")
        return

    # Check / install SDK
    if packages and not _is_importable(probe):
        _warn(f"{name} requires additional packages: {', '.join(packages)}")
        do_install = _ask("Install now?", "Y")
        if do_install.strip().upper() in ("Y", "YES", ""):
            print("  Installing packages...")
            if _install_packages(packages):
                _ok(f"{name} packages installed")
            else:
                _err(
                    f"Installation failed — install manually:\n"
                    f"    pip install {' '.join(packages)}"
                )

    # Credentials
    prompt = _ADAPTER_TOKEN_PROMPT.get(key, f"{name} token")
    token = _ask(prompt)

    # Update config.yaml
    if key not in adapters:
        adapters[key] = {}
    adapters[key]["enabled"] = True
    ws_url = adapters.get("cli", {}).get("core_ws_url") or "ws://localhost:8765"
    adapters[key].setdefault("core_ws_url", ws_url)
    cfg["adapters"] = adapters
    _save_yaml(config_path, cfg)
    _ok(f"config.yaml updated — {name} enabled")

    # Write token to .env (alongside config.yaml)
    if token:
        env_path = config_path.parent / ".env"
        env_key = f"CORDBEAT_ADAPTERS__{key.upper()}__OPTIONS__TOKEN"
        _update_env_file(env_path, env_key, token)
        _ok(f"Token written to {env_path}")

    start_cmd = f"cordbeat-{key}"
    print(f"\n  {_c('✓', _GREEN)} Start {name} with:  {_c(start_cmd, _BOLD)}\n")


# ---------------------------------------------------------------------------
# Skill flow
# ---------------------------------------------------------------------------


def _find_bundled_skills_dir() -> Path | None:
    """Locate the built-in skills/ directory (works in source and wheel installs)."""
    # Source layout: src/cordbeat/tools/add_cmd.py  →  ../../../../skills/
    src_root = Path(__file__).parent.parent.parent.parent
    bundled = src_root / "skills"
    if bundled.is_dir():
        return bundled
    return None


def _read_skill_info(skill_dir: Path) -> dict[str, Any] | None:
    yaml_path = skill_dir / "skill.yaml"
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return None


def _read_skill_deps(skill_dir: Path) -> list[str]:
    toml_path = skill_dir / "pyproject.toml"
    if not toml_path.exists():
        return []
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return list(data.get("project", {}).get("dependencies", []))
    except Exception:  # noqa: BLE001
        return []


def _check_deps_installed(deps: list[str]) -> list[str]:
    """Return subset of *deps* that are NOT currently importable."""
    # Map common package names to their import name
    _pkg_to_import: dict[str, str] = {
        "pillow": "PIL",
        "httpx": "httpx",
        "aiohttp": "aiohttp",
    }
    missing = []
    for dep in deps:
        # Strip extras and version specifiers → bare package name
        pkg = dep.split("[")[0].split(">=")[0].split("==")[0].split("!=")[0].strip()
        import_name = _pkg_to_import.get(pkg.lower(), pkg.lower().replace("-", "_"))
        if not _is_importable(import_name):
            missing.append(dep)
    return missing


def _install_skill_deps(deps: list[str]) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *deps],
        capture_output=True,
    )
    return result.returncode == 0


def _add_skill(config_path: Path) -> None:
    cfg = _load_yaml(config_path)
    skills_dir_str: str = cfg.get("skills_dir", "")
    if skills_dir_str:
        skills_dir = Path(skills_dir_str).expanduser()
        if not skills_dir.is_absolute():
            skills_dir = config_path.parent / skills_dir
    else:
        skills_dir = config_path.parent / "skills"

    # Collect skills from bundled dir + skills_dir (deduplicated by name)
    bundled = _find_bundled_skills_dir()
    search_dirs: list[Path] = []
    if bundled and bundled.is_dir():
        search_dirs.append(bundled)
    if skills_dir.is_dir() and skills_dir != bundled:
        search_dirs.append(skills_dir)

    skills: list[tuple[Path, dict[str, Any], list[str]]] = []
    seen: set[str] = set()
    for sdir in search_dirs:
        for sub in sorted(sdir.iterdir()):
            if not sub.is_dir() or sub.name in seen:
                continue
            info = _read_skill_info(sub)
            if info:
                deps = _read_skill_deps(sub)
                skills.append((sub, info, deps))
                seen.add(sub.name)

    if not skills:
        _err("No skills found. Check that skills_dir is set correctly in config.yaml.")
        return

    print("\n  Available skills:")
    for i, (path, info, deps) in enumerate(skills, 1):
        name = info.get("name", path.name)
        desc = info.get("description_en") or info.get("description", "")
        level = info.get("safety", {}).get("level", "safe")
        dep_str = f"  [needs: {', '.join(deps)}]" if deps else ""
        print(f"    {i}. {_c(name, _BOLD)}  [{level}]  {str(desc)[:60]}{dep_str}")

    raw = _ask("Choice", "1")
    try:
        idx = int(raw) - 1
        if not 0 <= idx < len(skills):
            idx = 0
    except ValueError:
        idx = 0

    skill_path, info, deps = skills[idx]
    skill_name: str = str(info.get("name", skill_path.name))

    # Check / install deps
    if deps:
        missing = _check_deps_installed(deps)
        if missing:
            _warn(f"Missing dependencies: {', '.join(missing)}")
            do_install = _ask("Install now?", "Y")
            if do_install.strip().upper() in ("Y", "YES", ""):
                if _install_skill_deps(missing):
                    _ok("Dependencies installed")
                else:
                    _err(
                        f"Installation failed. Install manually:\n"
                        f"    pip install {' '.join(missing)}"
                    )
        else:
            _ok("All dependencies already installed")

    # Copy skill to skills_dir if it came from bundled/ and skills_dir is elsewhere
    skill_dest = skills_dir / skill_path.name
    if skill_path.resolve() != skill_dest.resolve():
        skills_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(skill_path), str(skill_dest), dirs_exist_ok=True)
        _ok(f"Skill '{skill_name}' deployed to {skill_dest}")
    else:
        _ok(f"Skill '{skill_name}' is already in skills_dir")

    print(
        f"\n  {_c('✓', _GREEN)} Skill {_c(skill_name, _BOLD)} is ready"
        " — restart CordBeat to load it.\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_add(config_path: Path | None = None) -> None:
    """Run the add-component wizard."""
    print(f"\n  {_c('CordBeat — Add Component', _PINK + _BOLD)}\n")

    if config_path is None:
        config_path = _find_config()

    if config_path is None:
        _err("No config.yaml found. Run cordbeat-init first.")
        sys.exit(1)

    _ok(f"Using config: {config_path}")

    print("\n  What would you like to add?")
    print("    1. Adapter  (Discord, Telegram, Slack, ...)")
    print("    2. Skill    (draw, web_search, weather, ...)")

    raw = _ask("Choice", "1")
    choice = raw.strip()

    if choice in ("1", "adapter", "adapters"):
        _add_adapter(config_path)
    elif choice in ("2", "skill", "skills"):
        _add_skill(config_path)
    else:
        _err(f"Unknown choice: {choice!r}")
        sys.exit(1)


def cordbeat_add_cli() -> None:
    """CLI entry point for ``cordbeat-add``."""
    run_add()
