"""Configuration loader for CordBeat."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 8765


@dataclass
class AdapterConfig:
    core_ws_url: str = "ws://localhost:8765"
    enabled: bool = True
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


@dataclass
class AIBackendConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    adapters: dict[str, AdapterConfig] = field(default_factory=dict)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ai_backend: AIBackendConfig = field(default_factory=AIBackendConfig)
    soul_dir: str = "data/soul"
    skills_dir: str = "skills"
    data_dir: str = "data"


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML file."""
    path = Path(path)
    if not path.exists():
        return Config()

    with path.open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

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

    return Config(
        gateway=gateway,
        adapters=adapters,
        heartbeat=heartbeat,
        memory=memory,
        ai_backend=ai_backend,
        soul_dir=raw.get("soul_dir", "data/soul"),
        skills_dir=raw.get("skills_dir", "skills"),
        data_dir=raw.get("data_dir", "data"),
    )
