"""Zero-friction setup wizard for CordBeat.

Automatically detects a running Ollama instance and bootstraps
``~/.cordbeat/`` with sensible defaults.  Asks questions only
when auto-detection fails.
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from cordbeat.config import cordbeat_home

# -- ANSI color helpers ────────────────────────────────────────────────


def _supports_color() -> bool:
    """Return True when stdout likely supports ANSI escape codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    """Wrap *text* in an ANSI escape sequence when color is supported."""
    if not _supports_color():
        return text
    return f"{code}{text}\033[0m"


_CYAN = "\033[96m"
_BOLD = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"

_BANNER_ART = r"""
   ██████╗ ██████╗ ██████╗ ██████╗ ██████╗ ███████╗ █████╗ ████████╗
  ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
  ██║     ██║   ██║██████╔╝██║  ██║██████╔╝█████╗  ███████║   ██║
  ██║     ██║   ██║██╔══██╗██║  ██║██╔══██╗██╔══╝  ██╔══██║   ██║
  ╚██████╗╚██████╔╝██║  ██║██████╔╝██████╔╝███████╗██║  ██║   ██║
   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝
"""

_BANNER_SUBTITLE = "  A local-first autonomous AI agent that stays by your side.\n"

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

# Mapping from Python import name → pip package name for dependency checks.
_REQUIRED_HEAVY_DEPS: dict[str, str] = {
    "sqlite_vec": "sqlite-vec",
    "sentence_transformers": "sentence-transformers",
}


# -- Pre-flight checks ─────────────────────────────────────────────────


def _check_required_deps() -> None:
    """Verify that heavy runtime dependencies are importable.

    Called once at the very start of ``cordbeat_init_cli`` so the user
    gets a clear install instruction **before** the interactive wizard
    collects any input — rather than a cryptic traceback after they have
    already typed everything in.
    """
    missing: list[str] = []
    for import_name in _REQUIRED_HEAVY_DEPS:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(import_name)

    if not missing:
        return

    pip_names = [_REQUIRED_HEAVY_DEPS[n] for n in missing]
    print(f"\n  {_c('[ERROR] Missing required dependencies:', _RED)}")
    for imp, pip in zip(missing, pip_names):
        print(f"      {imp!r}  ->  pip install {pip}")
    print(
        "\n  Install them and then re-run cordbeat-init:\n"
        f"    pip install {' '.join(pip_names)}\n"
        "    # or: uv add " + " ".join(pip_names)
    )
    sys.exit(1)


# -- Prompt helpers ────────────────────────────────────────────────────


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"  {_c(prompt, _BOLD)}{suffix}: ").strip()
    return answer or default


def _ok(msg: str) -> None:
    print(f"  {_c('✓', _GREEN)} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_c('⚠', _YELLOW)} {msg}")


def _err(msg: str) -> None:
    print(f"  {_c('✗', _RED)} {msg}")


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


# -- Adapter selection ─────────────────────────────────────────────────

# (key, display_name, pip_packages, import_probe)
_ADAPTER_SPECS: list[tuple[str, str, list[str], str]] = [
    ("cli", "CLI only (no extra adapter)", [], ""),
    ("discord", "Discord", ["discord.py>=2.0"], "discord"),
    ("telegram", "Telegram", ["python-telegram-bot>=21.0"], "telegram"),
    ("slack", "Slack", ["slack-sdk>=3.27", "aiohttp>=3.9"], "slack_sdk"),
    ("line", "LINE", ["line-bot-sdk>=3.11", "aiohttp>=3.9"], "linebot"),
    ("whatsapp", "WhatsApp", ["aiohttp>=3.9"], "aiohttp"),
]

_ADAPTER_TOKEN_PROMPT: dict[str, str] = {
    "discord":  "Discord bot token",
    "telegram": "Telegram bot token (from @BotFather)",
    "slack":    "Slack bot token (xoxb-...)",
    "line":     "LINE channel access token",
    "whatsapp": "WhatsApp API token",
}


def _is_importable(module: str) -> bool:
    """Return True if *module* can be imported without error."""
    if not module:
        return True
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def _install_packages(packages: list[str]) -> bool:
    """Install *packages* using pip in the current interpreter.

    Returns True on success.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def _select_adapter() -> tuple[str, str | None]:
    """Prompt the user to choose a platform adapter.

    Returns ``(adapter_key, token_or_none)``.
    """
    print("\n  Which platform adapter would you like to use?")
    for i, (key, name, _, _) in enumerate(_ADAPTER_SPECS, 1):
        print(f"    {i}. {name}")

    raw = _ask("Choice", "1")
    try:
        idx = int(raw) - 1
        if not 0 <= idx < len(_ADAPTER_SPECS):
            idx = 0
    except ValueError:
        idx = 0

    key, name, packages, probe = _ADAPTER_SPECS[idx]

    if key == "cli":
        return key, None

    # Check / install dependencies
    if not _is_importable(probe):
        _warn(f"{name} requires additional packages: {', '.join(packages)}")
        do_install = _ask("Install now?", "Y")
        if do_install.strip().upper() in ("Y", "YES", ""):
            print("  Installing packages...")
            if _install_packages(packages):
                _ok(f"{name} packages installed successfully")
            else:
                _err(
                    f"Installation failed — install manually:\n"
                    f"    pip install {' '.join(packages)}"
                )

    # Ask for credentials
    prompt = _ADAPTER_TOKEN_PROMPT.get(key, f"{name} token")
    token = _ask(prompt)
    return key, token or None


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
    adapter: str = "cli",
    adapter_token: str | None = None,
) -> dict[str, Any]:
    """Build a config dict with paths anchored to *home*."""
    import secrets

    cfg: dict[str, Any] = {
        "gateway": {
            "host": "127.0.0.1",
            "port": 8765,
            "auth_token": secrets.token_urlsafe(32),
        },
        "log": {"level": "INFO"},
        "ai_backend": {
            "provider": provider,
            "base_url": base_url,
            "model": model,
        },
        "memory": {
            "sqlite_path": str(home / "cordbeat.db"),
        },
        "soul": {"soul_dir": str(home / "soul")},
        "skills_dir": str(home / "skills"),
        "data_dir": str(home),
        "adapters": {"cli": {"enabled": True}},
    }
    if api_key:
        cfg["ai_backend"]["options"] = {"api_key": api_key}
    if adapter != "cli":
        adapter_cfg: dict[str, Any] = {"enabled": True}
        if adapter_token:
            adapter_cfg["options"] = {"token": adapter_token}
        cfg["adapters"][adapter] = adapter_cfg
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
    print(_c(_BANNER_ART, _CYAN + _BOLD) + _BANNER_SUBTITLE)

    # ── Auto-detect AI backend ───────────────────────────────────
    print("  Detecting AI backend...")
    detected_model = _probe_ollama()

    if detected_model:
        # Zero-question path — Ollama
        _ok(f"Ollama detected — model: {detected_model}\n")
        provider = "ollama"
        base_url = "http://localhost:11434"
        model = detected_model
        api_key = ""
    else:
        # Try llama.cpp server
        llama_model = _probe_llama_cpp()
        if llama_model:
            _ok(f"llama.cpp detected — model: {llama_model}\n")
            provider = "openai_compat"
            base_url = "http://localhost:8080/v1"
            model = llama_model
            api_key = ""
        else:
            # Fallback: ask minimal questions
            _err("No local AI backend detected\n")
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

    # ── Adapter selection ─────────────────────────────────────────
    adapter, adapter_token = _select_adapter()

    # ── Create directory structure ────────────────────────────────
    print("\n  Creating ~/.cordbeat/ ...")
    home.mkdir(parents=True, exist_ok=True)
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
        adapter=adapter,
        adapter_token=adapter_token,
    )
    config_path.write_text(
        yaml.dump(cfg, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    _ok(str(config_path))

    # ── Connection test ───────────────────────────────────────────
    print(f"\n  Testing {provider} at {base_url}...")
    if _probe_provider(provider, base_url):
        _ok("AI backend is reachable!")
    else:
        _warn("Could not reach AI backend (you can start it later)")

    # ── Done ──────────────────────────────────────────────────────
    print(f"\n  {_c('✨', _YELLOW)} {name} is ready to come alive!")
    print("  Starting CordBeat...\n")
    return config_path


def cordbeat_init_cli() -> None:
    """CLI entry point for ``cordbeat-init``.

    Checks for required heavy dependencies first (before asking the user
    anything), then runs the setup wizard and starts CordBeat in
    combined server + interactive CLI chat mode.
    """
    import asyncio

    from cordbeat.exceptions import MemorySubsystemError
    from cordbeat.main import main_with_cli

    _check_required_deps()  # exits with friendly message if deps are missing

    config_path = run_wizard()
    try:
        asyncio.run(main_with_cli(str(config_path)))
    except MemorySubsystemError as exc:
        # Safety net: any dep check we missed surfaces here with a clear message.
        print(f"\n  [ERROR] Failed to start: {exc}")
        print("  Make sure all required packages are installed and re-run cordbeat.")
        sys.exit(1)
