"""Zero-friction setup wizard for CordBeat.

Automatically detects a running Ollama instance and bootstraps
``~/.cordbeat/`` with sensible defaults.  Asks questions only
when auto-detection fails.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from cordbeat.config import cordbeat_home

# ANSI color helpers
_P = "\033[95m"   # bright/light pink (magenta bright)
_M = "\033[35m"   # magenta/dark pink
_PD = "\033[91m"  # hot pink (bright red)
_Y = "\033[33m"   # yellow
_G = "\033[32m"   # green
_R = "\033[31m"   # red
_W = "\033[37m"   # white
_BOLD = "\033[1m"
_RESET = "\033[0m"
# aliases kept for internal use
_C = _P
_B = _M


def _colored(text: str) -> str:
    """Apply alternating pink gradient to each line of banner text."""
    colors = [_P, _M, _PD, _M, _P, _PD]
    lines = text.splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        result.append(colors[i % len(colors)] + _BOLD + line + _RESET)
    return "".join(result)


_BANNER_RAW = r"""
   ██████╗ ██████╗ ██████╗ ██████╗ ██████╗ ███████╗ █████╗ ████████╗
  ██╔════╝██╔═══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
  ██║     ██║   ██║██████╔╝██║  ██║██████╔╝█████╗  ███████║   ██║
  ██║     ██║   ██║██╔══██╗██║  ██║██╔══██╗██╔══╝  ██╔══██║   ██║
  ╚██████╗╚██████╔╝██║  ██║██████╔╝██████╔╝███████╗██║  ██║   ██║
   ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝

  A local-first autonomous AI agent that stays by your side.
"""
_BANNER = _colored(_BANNER_RAW)

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
    print("\n  [ERROR] Missing required dependencies:")
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
    answer = input(f"  {_W}{prompt}{_RESET}{suffix}: ").strip()
    return answer or default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"  {_W}{prompt}{_RESET} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


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


# -- Optional extras ──────────────────────────────────────────────────

_EXTRAS: list[dict[str, Any]] = [
    {
        "key": "stt-local",
        "label": "STT (local)  — Voice→text via Whisper on CPU/GPU",
        "packages": ["faster-whisper"],
        "config_hint": "stt.backend: whisper_local",
    },
    {
        "key": "tts-edge",
        "label": "TTS (Edge)   — Text→voice via Microsoft Edge TTS (free)",
        "packages": ["edge-tts"],
        "config_hint": "tts.backend: edge_tts",
    },
    {
        "key": "stt-openai",
        "label": "STT (OpenAI) — Voice→text via OpenAI Whisper API",
        "packages": ["openai"],
        "config_hint": "stt.backend: whisper_openai",
    },
    {
        "key": "tts-openai",
        "label": "TTS (OpenAI) — Text→voice via OpenAI TTS API",
        "packages": ["openai"],
        "config_hint": "tts.backend: openai",
    },
]

# -- i18n strings ──────────────────────────────────────────────────────

_T: dict[str, dict[str, str]] = {
    "en": {
        "builtin_skills": "Built-in skills:",
        "enable_skills": "Enable skills (comma-separated numbers / 'all' / 'none')",
        "optional_features": "Optional features:",
        "install_extras": "Install extras (comma-separated numbers / 'none')",
        "installing": "Installing",
        "installed": "Packages installed",
        "install_failed": "Install failed (you can run it manually later)",
        "char_name": "Character name",
        "lang_prompt": "Response language (en, ja, zh, ko, ...)",
        "creating_dir": "Creating ~/.cordbeat/ ...",
    },
    "ja": {
        "builtin_skills": "ビルトインスキル:",
        "enable_skills": "有効にするスキルを選択 (番号をカンマ区切り / 'all' / 'none')",
        "optional_features": "オプション機能:",
        "install_extras": "追加パッケージをインストール (番号をカンマ区切り / 'none')",
        "installing": "インストール中",
        "installed": "パッケージをインストールしました",
        "install_failed": "インストール失敗 (後から手動で実行してください)",
        "char_name": "キャラクター名",
        "lang_prompt": "応答言語 (en, ja, zh, ko, ...)",
        "creating_dir": "~/.cordbeat/ を作成中 ...",
    },
}


def _t(lang: str, key: str) -> str:
    """Return a translated wizard UI string, falling back to English."""
    return _T.get(lang, _T["en"]).get(key, _T["en"][key])


_SAFE_SKILLS = {
    "file_read", "file_write", "timer",
    "web_search", "weather", "api_call", "read_diary",
}
_DANGEROUS_SKILLS = {"shell_exec"}


def _discover_builtin_skills(bundled: Path, lang: str = "en") -> list[dict[str, str]]:
    """Return list of {name, description, safety_level} for bundled skills."""
    skills: list[dict[str, str]] = []
    if not bundled.is_dir():
        return skills
    for d in sorted(bundled.iterdir()):
        if not d.is_dir():
            continue
        yaml_path = d / "skill.yaml"
        if not yaml_path.exists():
            continue
        try:
            with yaml_path.open(encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            # Prefer description_{lang} (e.g. description_en), then description
            desc = raw.get(f"description_{lang}") or raw.get("description", "")
            skills.append({
                "name": d.name,
                "description": desc,
                "safety_level": (raw.get("safety") or {}).get("level", "safe"),
            })
        except (OSError, yaml.YAMLError):
            skills.append({"name": d.name, "description": "", "safety_level": "safe"})
    return skills


def _select_skills(bundled: Path, skills_dir: Path, lang: str = "en") -> None:
    """Interactive skill selection; copies selected skills to skills_dir."""
    skills = _discover_builtin_skills(bundled, lang)
    if not skills:
        return

    print(f"\n  {_BOLD}{_t(lang, 'builtin_skills')}{_RESET}")
    for i, s in enumerate(skills, 1):
        safety = s["safety_level"]
        if safety == "requires_confirmation":
            color = _Y
        elif safety == "dangerous":
            color = _R
        else:
            color = _G
        name_field = s["name"].ljust(14)
        desc = s["description"]
        print(f"  {_W}[{i}]{_RESET} {name_field} {color}({safety}){_RESET}  {desc}")

    default_nums = ",".join(
        str(i) for i, s in enumerate(skills, 1)
        if s["name"] not in _DANGEROUS_SKILLS
    )
    raw = _ask(_t(lang, "enable_skills"), default_nums)
    raw = raw.strip().lower()

    if raw in ("all", "a"):
        selected = {s["name"] for s in skills}
    elif raw in ("none", "n", ""):
        selected = set()
    else:
        selected = set()
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part) - 1
                if 0 <= idx < len(skills):
                    selected.add(skills[idx]["name"])

    skills_dir.mkdir(parents=True, exist_ok=True)
    for s in skills:
        dst = skills_dir / s["name"]
        src = bundled / s["name"]
        if s["name"] in selected:
            if not dst.exists():
                shutil.copytree(src, dst)
            print(f"  {_G}✓{_RESET} {s['name']}")
        # Skills not selected are simply not copied; main.py won't add them
        # unless the user manually copies them later.


def _select_extras(lang: str = "en") -> None:
    """Ask which optional features to install via uv pip."""
    if not _EXTRAS:
        return

    if shutil.which("uv") is None:
        return  # can't install without uv

    print(f"\n  {_BOLD}{_t(lang, 'optional_features')}{_RESET}")
    for i, ext in enumerate(_EXTRAS, 1):
        print(f"  {_W}[{i}]{_RESET} {ext['label']}")
        print(f"        {_W}→ {ext['config_hint']}{_RESET}")

    raw = _ask(_t(lang, "install_extras"), "none")
    raw = raw.strip().lower()
    if raw in ("none", "n", ""):
        return

    to_install: set[str] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(_EXTRAS):
                to_install.update(_EXTRAS[idx]["packages"])

    if not to_install:
        return

    pkg_list = sorted(to_install)
    print(f"\n  {_t(lang, 'installing')}: {', '.join(pkg_list)} ...")
    try:
        subprocess.run(  # noqa: S603
            ["uv", "pip", "install", "--quiet", *pkg_list],  # noqa: S607
            check=True,
        )
        print(f"  {_G}✓{_RESET} {_t(lang, 'installed')}")
    except subprocess.CalledProcessError as exc:
        print(f"  {_Y}⚠{_RESET} {_t(lang, 'install_failed')}: {exc}")


# -- Main wizard ───────────────────────────────────────────────────────


def run_wizard(home: Path | None = None) -> Path:
    """Run the interactive setup wizard.

    Returns the path to the created ``config.yaml``.
    """
    home = home or cordbeat_home()
    print(_BANNER)

    # ── Language selection (first — affects all subsequent prompts) ──
    language = _ask("Response language (en, ja, zh, ko, ...)", "en")

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

    # ── Character name ────────────────────────────────────────────
    name = _ask(_t(language, "char_name"), "CordBeat")

    # ── Create directory structure ────────────────────────────────
    print(f"\n  {_t(language, 'creating_dir')}")
    home.mkdir(parents=True, exist_ok=True)
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)

    # ── Built-in skill selection ──────────────────────────────────
    _here = Path(__file__).resolve()
    bundled_skills = _here.parent.parent.parent / "skills"
    _select_skills(bundled_skills, skills_dir, language)

    # ── Optional feature extras ───────────────────────────────────
    _select_extras(language)

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
    """CLI entry point for ``cordbeat-init``.

    Checks for required heavy dependencies first (before asking the user
    anything), then runs the setup wizard and starts CordBeat.
    """
    import asyncio

    from cordbeat.exceptions import MemorySubsystemError
    from cordbeat.main import main

    _check_required_deps()  # exits with friendly message if deps are missing

    config_path = run_wizard()
    try:
        asyncio.run(main(str(config_path)))
    except MemorySubsystemError as exc:
        # Safety net: any dep check we missed surfaces here with a clear message.
        print(f"\n  [ERROR] Failed to start: {exc}")
        print("  Make sure all required packages are installed and re-run cordbeat.")
        sys.exit(1)
