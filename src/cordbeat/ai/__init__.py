"""AI subsystem: backend, cache, extraction, validation, prompt, STT/TTS.

Imports are done lazily via ``__getattr__`` to avoid circular-import issues
arising from the cross-package dependency graph:
  ai.extraction → agent.soul → agent.heartbeat → core.engine → ai.extraction
"""

from __future__ import annotations

from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "AIBackend": ("cordbeat.ai.backend", "AIBackend"),
    "ConversationCompressor": ("cordbeat.ai.compression", "ConversationCompressor"),
    "create_backend": ("cordbeat.ai.backend", "create_backend"),
    "CachingBackend": ("cordbeat.ai.cache", "CachingBackend"),
    "MemoryExtractor": ("cordbeat.ai.extraction", "MemoryExtractor"),
    "build_context": ("cordbeat.ai.prompt", "build_context"),
    "build_soul_system_prompt": ("cordbeat.ai.prompt", "build_soul_system_prompt"),
    "sanitize": ("cordbeat.ai.prompt", "sanitize"),
    "STTBackend": ("cordbeat.ai.stt", "STTBackend"),
    "create_stt_backend": ("cordbeat.ai.stt", "create_stt_backend"),
    "TTSBackend": ("cordbeat.ai.tts", "TTSBackend"),
    "create_tts_backend": ("cordbeat.ai.tts", "create_tts_backend"),
    "validate_heartbeat_decision": (
        "cordbeat.ai.validation",
        "validate_heartbeat_decision",
    ),
    "validate_heartbeat_triage": (
        "cordbeat.ai.validation",
        "validate_heartbeat_triage",
    ),
    "validated_ai_json": ("cordbeat.ai.validation", "validated_ai_json"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_path, attr = _EXPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'cordbeat.ai' has no attribute {name!r}")
