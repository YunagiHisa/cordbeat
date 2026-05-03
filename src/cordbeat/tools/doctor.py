"""Diagnostic checks for a CordBeat installation.

Run ``cordbeat-doctor`` for an interactive menu.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from cordbeat.config import cordbeat_home

_SECRET_KEYS: frozenset[str] = frozenset(
    {"token", "api_key", "bot_token", "webhook_secret", "auth_token", "password"}
)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    suffix = f" — {detail}" if detail else ""
    print(f"  {mark} {label}{suffix}")
    return ok


def _probe(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=5):  # noqa: S310
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def run_doctor(home: Path | None = None) -> int:
    """Run diagnostic checks and return 0 if healthy, 1 otherwise."""
    home = home or cordbeat_home()
    print(f"\n  CordBeat Doctor — checking {home}\n")

    ok = True

    # ── Home directory ────────────────────────────────────────────
    ok &= _check("Home directory exists", home.is_dir())

    # ── Config ────────────────────────────────────────────────────
    config_path = home / "config.yaml"
    ok &= _check("config.yaml exists", config_path.is_file())

    cfg: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    # ── Gateway auth token ────────────────────────────────────────────
    gateway = cfg.get("gateway", {})
    auth_token: str = gateway.get("auth_token", "")
    has_token = bool(auth_token)
    ok &= _check("Gateway auth_token set", has_token, "required for adapter auth")
    if has_token:
        masked = (
            auth_token[:6] + "..." + auth_token[-4:] if len(auth_token) > 10 else "***"
        )
        print(f"    Partial token : {masked}")
        print(
            "    Copy to adapter .env:\n"
            f"      CORDBEAT_GATEWAY__AUTH_TOKEN={auth_token}\n"
        )
    else:
        print("    Run cordbeat-init to generate one.\n")

    # ── AI backend ────────────────────────────────────────────────
    ai = cfg.get("ai_backend", {})
    provider = ai.get("provider", "ollama")
    base_url = ai.get("base_url", "http://localhost:11434")
    if provider == "ollama":
        probe_url = base_url.rstrip("/") + "/api/tags"
    else:
        probe_url = base_url.rstrip("/") + "/models"
    ok &= _check(
        f"AI backend reachable ({provider})",
        _probe(probe_url),
        base_url,
    )

    # ── Soul files ────────────────────────────────────────────────
    soul_dir = Path(cfg.get("soul", {}).get("soul_dir", str(home / "soul")))
    ok &= _check("soul_core.yaml exists", (soul_dir / "soul_core.yaml").is_file())
    ok &= _check("soul.yaml exists", (soul_dir / "soul.yaml").is_file())

    # ── Data paths ────────────────────────────────────────────────
    mem = cfg.get("memory", {})
    db_path = Path(mem.get("sqlite_path", str(home / "cordbeat.db")))
    ok &= _check(
        "SQLite path writable",
        db_path.parent.is_dir() or home.is_dir(),
        str(db_path),
    )

    # ── sqlite-vec extension ──────────────────────────────────────
    try:
        import sqlite3

        import sqlite_vec

        probe = sqlite3.connect(":memory:")
        probe.enable_load_extension(True)
        sqlite_vec.load(probe)
        probe.close()
        ok &= _check("sqlite-vec extension loadable", True, "vec0 available")
    except Exception as e:  # noqa: BLE001
        ok &= _check("sqlite-vec extension loadable", False, str(e))

    # ── Skills directory ──────────────────────────────────────────
    skills_dir = Path(cfg.get("skills_dir", str(home / "skills")))
    ok &= _check("Skills directory exists", skills_dir.is_dir(), str(skills_dir))

    # ── Summary ───────────────────────────────────────────────────
    if ok:
        print("\n  All checks passed!\n")
    else:
        print("\n  Some checks failed. Run `cordbeat` to set up.\n")

    return 0 if ok else 1


# ── Helper: locate config.example.yaml ────────────────────────────────────────


def _find_example_config() -> Path | None:
    """Locate config.example.yaml from package data or repo root."""
    # Walk up from this file to find the repo / install root
    for ancestor in Path(__file__).parents:
        candidate = ancestor / "config.example.yaml"
        if candidate.is_file():
            return candidate
    return None


# ── fix-config ────────────────────────────────────────────────────────────────


def _deep_missing_keys(
    user: dict[str, Any],
    example: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    """Return dotted paths for keys in *example* not present in *user*."""
    missing: list[str] = []
    for k, v in example.items():
        full = f"{prefix}.{k}" if prefix else k
        if k not in user:
            missing.append(full)
        elif isinstance(v, dict) and isinstance(user.get(k), dict):
            missing.extend(_deep_missing_keys(user[k], v, full))
    return missing


def run_fix_config(home: Path) -> int:
    """Compare config.yaml against config.example.yaml; offer to add missing keys."""
    config_path = home / "config.yaml"
    if not config_path.is_file():
        print("  ✗ config.yaml not found. Run cordbeat-init first.")
        return 1

    example_path = _find_example_config()
    if not example_path:
        print("  ✗ config.example.yaml not found — cannot compare.")
        return 1

    with config_path.open(encoding="utf-8") as f:
        user_cfg: dict[str, Any] = yaml.safe_load(f) or {}
    with example_path.open(encoding="utf-8") as f:
        example_cfg: dict[str, Any] = yaml.safe_load(f) or {}

    # Check only top-level missing sections (adding nested keys would corrupt YAML)
    missing_top = [k for k in example_cfg if k not in user_cfg]
    if not missing_top:
        print("  ✓ Config is up to date — no missing top-level keys.")
        return 0

    print(f"  Found {len(missing_top)} missing section(s):")
    for k in missing_top:
        print(f"    - {k}")

    try:
        answer = input("\n  Add missing sections with defaults? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0

    if answer != "y":
        print("  Skipped.")
        return 0

    additions = ["\n\n# ── Added by cordbeat-doctor fix-config ──────────────────\n"]
    for k in missing_top:
        additions.append(
            yaml.dump({k: example_cfg[k]}, default_flow_style=False, allow_unicode=True)
        )

    with config_path.open("a", encoding="utf-8") as f:
        for block in additions:
            f.write(block)

    print(f"  ✓ Added {len(missing_top)} section(s) to {config_path}")
    return 0


# ── show-config ───────────────────────────────────────────────────────────────


def _mask_secrets(d: Any) -> None:
    if not isinstance(d, dict):
        return
    for k in list(d):
        v = d[k]
        if k in _SECRET_KEYS and isinstance(v, str) and v:
            d[k] = v[:3] + "***" + v[-3:] if len(v) > 6 else "***"
        elif isinstance(v, dict):
            _mask_secrets(v)


def run_show_config(home: Path) -> int:
    """Print current config with secrets masked."""
    config_path = home / "config.yaml"
    if not config_path.is_file():
        print("  ✗ config.yaml not found.")
        return 1

    with config_path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    _mask_secrets(cfg)
    print(yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False))
    return 0


# ── sync-config ───────────────────────────────────────────────────────────────


def _find_project_config() -> Path | None:
    """Locate a ``config.yaml`` in CWD or repo ancestors (not config.example.yaml)."""
    import os

    # 1. Current working directory
    cwd_cfg = Path(os.getcwd()) / "config.yaml"
    if cwd_cfg.is_file():
        return cwd_cfg

    # 2. Walk up from this file's directory (repo root heuristic)
    for ancestor in Path(__file__).parents:
        candidate = ancestor / "config.yaml"
        # Avoid matching ~/.cordbeat/config.yaml
        if candidate.is_file() and (ancestor / "pyproject.toml").is_file():
            return candidate

    return None


def _deep_diff_keys(
    src: dict[str, Any], dst: dict[str, Any], prefix: str = ""
) -> list[str]:
    """Return dotted paths for keys in *src* not present or different in *dst*."""
    diff: list[str] = []
    for k, v in src.items():
        full = f"{prefix}.{k}" if prefix else k
        if k not in dst:
            diff.append(f"  + {full}  (new)")
        elif isinstance(v, dict) and isinstance(dst.get(k), dict):
            diff.extend(_deep_diff_keys(v, dst[k], full))
        elif v != dst.get(k):
            diff.append(f"  ~ {full}  (changed)")
    return diff


def run_sync_config(home: Path, src_path: Path | None = None) -> int:
    """Copy a project config.yaml into ~/.cordbeat/, merging missing keys."""
    import copy
    import shutil

    dst_path = home / "config.yaml"

    src = src_path or _find_project_config()
    if not src:
        print("  ✗ No project config.yaml found (checked CWD and repo root).")
        return 1

    print(f"  Source : {src}")
    print(f"  Target : {dst_path}")

    with src.open(encoding="utf-8") as f:
        src_cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if dst_path.is_file():
        with dst_path.open(encoding="utf-8") as f:
            dst_cfg: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        dst_cfg = {}

    diff = _deep_diff_keys(src_cfg, dst_cfg)
    if not diff:
        print("  ✓ ~/.cordbeat/config.yaml is already in sync.")
        return 0

    print(f"\n  {len(diff)} change(s) would be applied:")
    for line in diff:
        print(f"    {line}")

    try:
        answer = input("\n  Apply? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0

    if answer != "y":
        print("  Skipped.")
        return 0

    # Backup existing config
    if dst_path.is_file():
        backup = dst_path.with_suffix(".yaml.bak")
        shutil.copy2(dst_path, backup)
        print(f"  Backup : {backup}")

    # Deep-merge: src values win for scalars; dst keeps keys not in src
    def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(base)
        for k, v in overlay.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = _merge(result[k], v)
            else:
                result[k] = copy.deepcopy(v)
        return result

    merged = _merge(dst_cfg, src_cfg)
    home.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)

    print(f"  ✓ Synced to {dst_path}")
    return 0




def main() -> None:
    """Interactive CordBeat Doctor CLI."""
    import sys

    home = cordbeat_home()

    # Allow ``cordbeat-doctor <config-path>`` to override home
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        p = Path(sys.argv[1])
        home = p.parent if p.is_file() else p

    print(f"\n  CordBeat Doctor — {home}")

    menu: list[tuple[str, Any]] = [
        ("Health check", lambda: run_doctor(home)),
        ("Fix config (add missing keys from example)", lambda: run_fix_config(home)),
        ("Sync config: project → ~/.cordbeat", lambda: run_sync_config(home)),
        ("Show config (secrets masked)", lambda: run_show_config(home)),
    ]

    while True:
        print()
        for i, (label, _) in enumerate(menu, 1):
            print(f"  [{i}] {label}")
        print("  [0] Exit")
        print()

        try:
            choice = input("  Choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice in ("0", "q", "quit", "exit"):
            break

        try:
            idx = int(choice) - 1
        except ValueError:
            print("  Please enter a number.")
            continue

        if 0 <= idx < len(menu):
            print()
            menu[idx][1]()
        else:
            print("  Invalid choice.")

    print("\n  Goodbye.\n")
