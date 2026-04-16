"""Configuration loader for CordBeat."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PREFIX = "CORDBEAT_"


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    handshake_timeout: float = 10.0


@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"


@dataclass
class AdapterConfig:
    core_ws_url: str = "ws://localhost:8765"
    enabled: bool = True
    reconnect_max_backoff: int = 60
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeartbeatConfig:
    default_interval_minutes: int = 60
    min_interval_minutes: int = 5
    max_interval_minutes: int = 1440
    quiet_hours_start: str = "01:00"
    quiet_hours_end: str = "07:00"
    timezone: str = "UTC"


@dataclass
class MemoryConfig:
    sqlite_path: str = "data/cordbeat.db"
    chroma_path: str = "data/chroma"
    decay_rate: float = 0.1
    archive_threshold: float = 0.05
    conversation_history_limit: int = 20
    memory_search_results: int = 3
    diary_max_tokens: int = 512
    facts_per_message_limit: int = 5
    extraction_temperature: float = 0.2
    flashbulb_intensity_threshold: float = 0.8
    max_user_input_len: int = 2000
    diary_temperature: float = 0.5
    consolidation_temperature: float = 0.2
    consolidation_episode_results: int = 10
    consolidation_facts_limit: int = 5
    chain_link_episode_results: int = 5
    chain_link_related_results: int = 3
    recall_keyword_search_results: int = 2
    emotion_recall_search_results: int = 2
    chain_recall_max_depth: int = 2
    recall_hints_limit: int = 20
    message_trim_keep: int = 100
    token_expiry_minutes: int = 10
    chain_link_query_limit: int = 100
    chain_link_max_results: int = 10
    recall_hints_retention_days: int = 2
    chain_links_retention_days: int = 2
    proposal_expiry_days: int = 7
    chain_recall_depth_penalty: float = 0.5


@dataclass
class SoulConfig:
    soul_dir: str = "data/soul"
    emotion_decay_rate: float = 0.05
    emotion_baseline_intensity: float = 0.3
    emotion_secondary_clear_threshold: float = 0.1


@dataclass
class AIBackendConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    timeout: float = 120.0
    max_tokens: int = 1024
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    adapters: dict[str, AdapterConfig] = field(default_factory=dict)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ai_backend: AIBackendConfig = field(default_factory=AIBackendConfig)
    soul: SoulConfig = field(default_factory=SoulConfig)
    log: LogConfig = field(default_factory=LogConfig)
    skills_dir: str = "skills"
    data_dir: str = "data"

    @property
    def soul_dir(self) -> str:
        """Backward-compatible accessor for soul directory path."""
        return self.soul.soul_dir


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def _load_dotenv(path: Path) -> None:
    """Load .env file into os.environ if it exists.

    Supports simple KEY=VALUE lines. Ignores comments and blank lines.
    Strips optional surrounding quotes from values.
    """
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            os.environ.setdefault(key, value)


def _apply_env_overrides(raw: dict[str, Any]) -> None:
    """Overlay CORDBEAT_* environment variables onto the raw config dict.

    Mapping convention:
        CORDBEAT_GATEWAY__HOST  → raw["gateway"]["host"]
        CORDBEAT_AI_BACKEND__MODEL → raw["ai_backend"]["model"]
        CORDBEAT_ADAPTERS__DISCORD__OPTIONS__TOKEN
            → raw["adapters"]["discord"]["options"]["token"]

    Double-underscore (__) separates nesting levels.
    The first segment after the prefix uses lowercase for the top-level key.
    """
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(_ENV_PREFIX):
            continue
        remainder = env_key[len(_ENV_PREFIX) :]
        parts = [p.lower() for p in remainder.split("__")]
        if not parts:
            continue

        target = raw
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            child = target[part]
            if not isinstance(child, dict):
                break
            target = child
        else:
            target[parts[-1]] = _coerce_value(env_value)


def _coerce_value(value: str) -> Any:
    """Convert string env var to an appropriate Python type.

    Only the words ``true/false/yes/no`` are treated as booleans so that
    numeric strings like ``"0"`` and ``"1"`` are correctly coerced to
    integers rather than booleans.
    """
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML file with env var overrides.

    Loading order (later wins):
        1. Dataclass defaults
        2. YAML file values
        3. .env file (same directory as config, or CWD)
        4. CORDBEAT_* environment variables
    """
    path = Path(path)

    # Load .env from config directory or CWD
    env_path = path.parent / ".env" if path.parent != Path() else Path(".env")
    _load_dotenv(env_path)

    if not path.exists():
        raw: dict[str, Any] = {}
    else:
        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    # Apply environment variable overrides
    _apply_env_overrides(raw)

    gateway = _build_dataclass(
        GatewayConfig,
        raw.get("gateway", {}),
    )
    heartbeat = _build_dataclass(
        HeartbeatConfig,
        raw.get("heartbeat", {}),
    )
    memory = _build_dataclass(
        MemoryConfig,
        raw.get("memory", {}),
    )
    ai_backend = _build_dataclass(
        AIBackendConfig,
        raw.get("ai_backend", {}),
    )

    adapters: dict[str, AdapterConfig] = {}
    for name, adapter_raw in raw.get("adapters", {}).items():
        if isinstance(adapter_raw, dict):
            adapters[name] = _build_dataclass(AdapterConfig, adapter_raw)

    # Handle soul config: support both legacy "soul_dir" and new "soul" section
    soul_raw = raw.get("soul", {})
    if isinstance(soul_raw, str):
        soul_raw = {"soul_dir": soul_raw}
    elif not isinstance(soul_raw, dict):
        soul_raw = {}
    if "soul_dir" in raw and "soul_dir" not in soul_raw:
        soul_raw["soul_dir"] = raw["soul_dir"]
    soul = _build_dataclass(SoulConfig, soul_raw)

    log = _build_dataclass(LogConfig, raw.get("log", {}))

    return Config(
        gateway=gateway,
        adapters=adapters,
        heartbeat=heartbeat,
        memory=memory,
        ai_backend=ai_backend,
        soul=soul,
        log=log,
        skills_dir=raw.get("skills_dir", "skills"),
        data_dir=raw.get("data_dir", "data"),
    )
