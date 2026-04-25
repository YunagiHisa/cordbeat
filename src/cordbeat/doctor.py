"""Diagnostic checks for a CordBeat installation."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from cordbeat.config import cordbeat_home


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
