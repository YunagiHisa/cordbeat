"""Zero-friction setup wizard for CordBeat.

Automatically detects a running Ollama instance and bootstraps
``~/.cordbeat/`` with sensible defaults.  Asks questions only
when auto-detection fails.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from cordbeat.config import cordbeat_home

_BANNER = r"""
   ██████╗ ██████╗ ██████╗ ██████╗ ██████╗ ███████╗ █████╗ ████████╗
  ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
  ██║     ██║   ██║██████╔╝██║  ██║██████╔╝█████╗  ███████║   ██║
  ██║     ██║   ██║██╔══██╗██║  ██║██╔══██╗██╔══╝  ██╔══██║   ██║
  ╚██████╗╚██████╔╝██║  ██║██████╔╝██████╔╝███████╗██║  ██║   ██║
   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝

  A local-first autonomous AI agent that stays by your side.
"""

_DEFAULT_SOUL_CORE = {
    "immutable_rules": [
        "Never harm a user",
        "Never lie",
        "Never deny being an AI",
        "Never take critical actions without user approval",
        "Never disable the emotion system",
        "Never completely erase memories",
    ],
}


# -- Prompt helpers ────────────────────────────────────────────────────


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"  {prompt}{suffix}: ").strip()
    return answer or default


# -- Auto-detection ────────────────────────────────────────────────────


def _probe_ollama(base_url: str = "http://localhost:11434") -> str | None:
    """Test connection to Ollama and return the first model name, or None."""
    import json

    try:
        url = base_url.rstrip("/") + "/api/tags"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        models = data.get("models", [])
        if models:
            return str(models[0].get("name", "llama3"))
        return "llama3"
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None


def _probe_llama_cpp(
    base_url: str = "http://localhost:8080",
) -> str | None:
    """Test connection to llama.cpp server and return the model name, or None."""
    import json

    try:
        url = base_url.rstrip("/") + "/v1/models"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        models = data.get("data", [])
        if models:
            return str(models[0].get("id", "default"))
        return "default"
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return None


def _probe_provider(
    provider: str,
    base_url: str,
) -> bool:
    """Test whether an AI backend is reachable."""
    try:
        url = base_url.rstrip("/")
        if provider == "ollama":
            url += "/api/tags"
        else:
            url += "/models"
        req = urllib.request.Request(url, method="GET")  # noqa: S310
        with urllib.request.urlopen(req, timeout=5):  # noqa: S310
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


# -- Provider presets ──────────────────────────────────────────────────

_PROVIDERS: dict[str, dict[str, str]] = {
    "ollama": {
        "config_provider": "ollama",
        "base_url": "http://localhost:11434",
    },
    "openai": {
        "config_provider": "openai",
        "base_url": "https://api.openai.com/v1",
    },
    "openai_compat": {
        "config_provider": "openai_compat",
        "base_url": "http://localhost:8080/v1",
    },
}


# -- Soul helpers ──────────────────────────────────────────────────────


def _build_soul_yaml(name: str, language: str) -> dict[str, object]:
    return {
        "identity": {"name": name, "pronoun": "I", "language": language},
        "personality": {
            "traits": ["curious", "emotionally expressive", "caring"],
        },
        "current_emotion": {
            "primary": "calm",
            "primary_intensity": 0.5,
            "secondary": "curiosity",
            "secondary_intensity": 0.3,
        },
        "heartbeat": {"quiet_hours": {"start": "01:00", "end": "07:00"}},
    }


# -- Config builder ────────────────────────────────────────────────────


def _build_config(
    home: Path,
    *,
    provider: str,
    base_url: str,
    model: str,
    api_key: str = "",
) -> dict[str, Any]:
    """Build a config dict with paths anchored to *home*."""
    cfg: dict[str, Any] = {
        "gateway": {"host": "0.0.0.0", "port": 8765},
        "log": {"level": "INFO"},
        "ai_backend": {
            "provider": provider,
            "base_url": base_url,
            "model": model,
        },
        "memory": {
            "sqlite_path": str(home / "cordbeat.db"),
            "chroma_path": str(home / "chroma"),
        },
        "soul": {"soul_dir": str(home / "soul")},
        "skills_dir": str(home / "skills"),
        "data_dir": str(home),
        "adapters": {"cli": {"enabled": True}},
    }
    if api_key:
        cfg["ai_backend"]["options"] = {"api_key": api_key}
    return cfg


# -- File writers ──────────────────────────────────────────────────────


def _write_soul_files(soul_dir: Path, name: str, language: str) -> None:
    soul_dir.mkdir(parents=True, exist_ok=True)

    core_path = soul_dir / "soul_core.yaml"
    if not core_path.exists():
        core_path.write_text(
            yaml.dump(_DEFAULT_SOUL_CORE, default_flow_style=False),
            encoding="utf-8",
        )
        print(f"  ✓ {core_path}")

    soul_path = soul_dir / "soul.yaml"
    if not soul_path.exists():
        soul_data = _build_soul_yaml(name, language)
        soul_path.write_text(
            yaml.dump(soul_data, default_flow_style=False),
            encoding="utf-8",
        )
        print(f"  ✓ {soul_path}")

    notes_path = soul_dir / "soul_notes.md"
    if not notes_path.exists():
        notes_path.write_text(
            f"# Soul Notes — {name}\n\nFree-form notes about this character.\n",
            encoding="utf-8",
        )
        print(f"  ✓ {notes_path}")


# -- Main wizard ───────────────────────────────────────────────────────


def run_wizard(home: Path | None = None) -> Path:
    """Run the interactive setup wizard.

    Returns the path to the created ``config.yaml``.
    """
    home = home or cordbeat_home()
    print(_BANNER)

    # ── Auto-detect AI backend ───────────────────────────────────
    print("  Detecting AI backend...")
    detected_model = _probe_ollama()

    if detected_model:
        # Zero-question path — Ollama
        print(f"  ✓ Ollama detected — model: {detected_model}\n")
        provider = "ollama"
        base_url = "http://localhost:11434"
        model = detected_model
        api_key = ""
    else:
        # Try llama.cpp server
        llama_model = _probe_llama_cpp()
        if llama_model:
            print(f"  ✓ llama.cpp detected — model: {llama_model}\n")
            provider = "openai_compat"
            base_url = "http://localhost:8080/v1"
            model = llama_model
            api_key = ""
        else:
            # Fallback: ask minimal questions
            print("  ✗ No local AI backend detected\n")
            print("  Supported providers: ollama, openai, openai_compat (llama.cpp)")
            provider = _ask("AI provider", "ollama")
            preset = _PROVIDERS.get(provider, _PROVIDERS["ollama"])
            base_url = _ask("API base URL", preset["base_url"])
            model = _ask("Model name", "llama3")
            provider = preset["config_provider"]
            api_key = ""
            if provider in ("openai", "openai_compat"):
                api_key = _ask("API key (leave empty to skip)", "")

    # ── Character name (optional) ─────────────────────────────────
    name = _ask("Character name", "CordBeat")
    language = _ask("Response language (en, ja, zh, ko, ...)", "en")

    # ── Create directory structure ────────────────────────────────
    print("\n  Creating ~/.cordbeat/ ...")
    home.mkdir(parents=True, exist_ok=True)
    (home / "chroma").mkdir(parents=True, exist_ok=True)
    (home / "skills").mkdir(parents=True, exist_ok=True)

    # ── Write soul files ──────────────────────────────────────────
    _write_soul_files(home / "soul", name, language)

    # ── Write config.yaml ─────────────────────────────────────────
    config_path = home / "config.yaml"
    cfg = _build_config(
        home,
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
    config_path.write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    print(f"  ✓ {config_path}")

    # ── Connection test ───────────────────────────────────────────
    print(f"\n  Testing {provider} at {base_url}...")
    if _probe_provider(provider, base_url):
        print("  ✓ AI backend is reachable!")
    else:
        print("  ✗ Could not reach AI backend (you can start it later)")

    # ── Done ──────────────────────────────────────────────────────
    print(f"\n  ✨ {name} is ready to come alive!")
    print("  Starting CordBeat...\n")
    return config_path


def cordbeat_init_cli() -> None:
    """CLI entry point for ``cordbeat-init``."""
    run_wizard()
