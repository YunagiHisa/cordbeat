"""Configuration loader for CordBeat."""

from __future__ import annotations

import ipaddress
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PREFIX = "CORDBEAT_"
_LOOPBACK_HOSTS = {"localhost"}


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().lower().strip("[]")
    if normalized in _LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def gateway_connect_host(host: str) -> str:
    """Return a client-connectable host for a configured gateway bind host."""
    normalized = host.strip().lower().strip("[]")
    if normalized in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def gateway_ws_url(host: str, port: int) -> str:
    return f"ws://{gateway_connect_host(host)}:{port}"


def cordbeat_home() -> Path:
    """Return the CordBeat home directory.

    Resolution order:
        1. ``CORDBEAT_HOME`` environment variable
        2. ``~/.cordbeat/``
    """
    env = os.environ.get("CORDBEAT_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cordbeat"


@dataclass
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    handshake_timeout: float = 10.0
    max_message_bytes: int = 64 * 1024 * 1024
    auth_token: str = ""


@dataclass
class LogConfig:
    level: str = "INFO"
    format: str = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    file: str = ""  # empty = stderr only; set a path to enable file logging
    max_bytes: int = 10_485_760  # 10 MiB — triggers rotation
    backup_count: int = 5


@dataclass
class AdapterConfig:
    core_ws_url: str = "ws://localhost:8765"
    enabled: bool = True
    reconnect_max_backoff: int = 60
    auth_token: str = ""  # populated from gateway.auth_token by runner
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeartbeatConfig:
    default_interval_minutes: int = 60
    min_interval_minutes: int = 5
    max_interval_minutes: int = 1440
    proactive_user_cooldown_minutes: int = 1440
    proactive_destination_cooldown_minutes: int = 360
    max_proactive_messages_per_tick: int = 1
    discovery_share_cooldown_minutes: int = 240
    max_discovery_shares_per_day: int = 3
    max_actions_per_tick: int = 3
    self_review_interval_ticks: int = 6
    quiet_hours_start: str = "01:00"
    quiet_hours_end: str = "07:00"
    timezone: str = "UTC"


@dataclass
class MemoryConfig:
    sqlite_path: str = "cordbeat.db"
    decay_rate: float = 0.1
    archive_threshold: float = 0.05
    conversation_history_limit: int = 20
    voice_conversation_history_limit: int = 6
    image_summary_enabled: bool = True
    memory_search_results: int = 3
    recalled_episode_context_limit: int = 4
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
    voice_recall_keywords_enabled: bool = False
    voice_memory_extraction_enabled: bool = False
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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    # Device for the sentence-transformers embedder ("cpu" | "cuda" | "").
    # Default CPU keeps the small MiniLM embedder off the GPU so it does not
    # compete for VRAM with STT/LLM models (which caused CUDA OOM in recall).
    embedding_device: str = "cpu"
    # Near-duplicate memories within this vector distance are merged
    # (strength reinforced) instead of stored again. 0.0 disables dedup.
    # 0.15 ≈ cosine similarity 0.99 on normalized embeddings — merges only
    # near-identical statements.
    dedup_distance_threshold: float = 0.15
    # Context compression: summarise old conversation chunks with LLM
    # instead of silently dropping them.  Runs during the sleep phase.
    context_compression_enabled: bool = True
    # Real-time: summarise in-memory if conversation history exceeds this many chars.
    context_compression_chars_threshold: int = 4000
    # Sleep: summarise when a user has more messages than this.
    context_compression_threshold: int = 60
    # Number of oldest messages to summarise per compression pass.
    context_compression_chunk: int = 20
    # LLM generation parameters for the compression / summarisation calls.
    context_compression_temperature: float = 0.3
    context_compression_max_tokens: int = 256


@dataclass
class SoulConfig:
    soul_dir: str = "soul"
    emotion_decay_rate: float = 0.05
    emotion_baseline_intensity: float = 0.3
    emotion_secondary_clear_threshold: float = 0.1
    emotion_style: str = "full"
    absence_note_days: int = 2


@dataclass
class LLMCacheConfig:
    """Optional response cache for LLM ``generate()`` calls.

    Off by default. When enabled, cache only deterministic-ish calls
    (temperature below ``max_temperature``) keyed on
    (system, prompt, model, temperature, max_tokens). LRU eviction on
    ``max_entries``; entries also expire after ``ttl_seconds``.
    """

    enabled: bool = False
    max_entries: int = 256
    ttl_seconds: int = 3600
    max_temperature: float = 0.2


@dataclass
class AIBackendConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "llama3"
    # Read timeout for LLM HTTP requests.  Increase this on slow hardware or
    # when using thinking models (Qwen3/DeepSeek-R1) that spend several minutes
    # in their chain-of-thought phase before producing a response.
    timeout: float = 300.0
    # Max tokens per generation request.
    # None (default) = let the server decide (uses the server's --n-predict / default).
    # Set explicitly (e.g. 4096) to override the server default.
    # ⚠ For thinking models (Qwen3/DeepSeek-R1): if your server's default is low
    #   (< 2048), set max_tokens: 4096 or higher so the thinking phase doesn't
    #   consume all tokens before the model generates a response.
    max_tokens: int | None = None
    options: dict[str, Any] = field(default_factory=dict)
    cache: LLMCacheConfig = field(default_factory=LLMCacheConfig)
    vision_enabled: bool = False
    # Send video attachments to the model as-is.  Only some backends decode
    # video (Gemini's OpenAI-compatible endpoint does; a local llama.cpp
    # vision model generally does not), so this is off by default and is
    # separate from vision_enabled.  Video is delivered through the same
    # image_url content part, so it also requires a vision-capable path.
    video_enabled: bool = False


@dataclass
class SkillSandboxConfig:
    timeout_seconds: float = 30.0
    memory_limit_mb: int = 256
    max_output_bytes: int = 1_048_576  # 1 MiB
    allow_network_by_default: bool = False


@dataclass
class SkillsConfig:
    sandbox: SkillSandboxConfig = field(default_factory=SkillSandboxConfig)
    default_rate_limit_per_minute: int = 0


@dataclass
class STTConfig:
    """Speech-to-text configuration.

    Set ``enabled = true`` in ``config.yaml`` to activate voice transcription.
    ``backend`` selects the implementation:

    * ``whisper_local``  — local *faster-whisper* model (extra: ``stt-local``)
    * ``whisper_openai`` — OpenAI cloud Whisper API
    * ``openai_compat``  — any OpenAI-compatible transcription endpoint
    """

    enabled: bool = False
    backend: str = "whisper_openai"
    model: str = "base"
    language: str = ""
    api_url: str = ""
    api_key: str = ""
    # Override for whisper_openai (set to a self-hosted OpenAI-compatible
    # endpoint).  Empty → use the official OpenAI API base URL.
    base_url: str = ""
    # HTTP request timeout for cloud STT calls (whisper_openai / openai_compat).
    timeout: float = 60.0
    # Inference device for whisper_local. "cpu", "cuda", "auto", or a specific
    # index like "cuda:1" to pin to one GPU. Empty → cpu. Real-time VC
    # transcription benefits greatly from a GPU.
    device: str = ""
    # faster-whisper compute type for whisper_local, e.g. "int8_float16" or
    # "int8" to fit large models into limited VRAM (~1.6 GB / ~1.5 GB for
    # large-v3 vs ~3 GB at float16). Empty → CTranslate2 default for the
    # device (float16 on CUDA, int8 on CPU).
    compute_type: str = ""
    # Maximum number of audio chunks decoded in parallel by faster-whisper's
    # batched inference pipeline. 1 keeps voice transcription serial and uses
    # the regular WhisperModel path, which is the lowest-memory option.
    batch_size: int = 1


@dataclass
class VoiceProfileConfig:
    """Stable speaker identity shared by prompt-capable TTS backends."""

    description: str = "A calm, approachable adult voice with a natural tone."
    # soul: use Soul.identity.language; fixed: use ``language``; auto: omit a
    # language instruction and let the synthesis backend infer it from text.
    language_source: str = "soul"
    language: str = ""


@dataclass
class SpeechDirectionConfig:
    """Content-aware delivery direction generated independently from replies."""

    enabled: bool = True
    history_turns: int = 3
    max_tokens: int = 128
    temperature: float = 0.2
    timeout: float = 3.0
    max_direction_chars: int = 180
    fallback: str = "Speak in a natural, conversational manner."


@dataclass
class TTSStreamingConfig:
    """Chunked synthesis and playback controls."""

    enabled: bool = True
    chunk_min_chars: int = 15
    chunk_max_chars: int = 45
    max_queue_size: int = 20


@dataclass
class TTSConfig:
    """Text-to-speech configuration.

    Set ``enabled = true`` in ``config.yaml`` to activate voice replies.
    ``backend`` selects the implementation:

    * ``edge_tts``      — Microsoft Edge TTS (free, extra: ``tts-edge``)
    * ``openai``        — OpenAI TTS API
    * ``openai_compat`` — any OpenAI-compatible speech endpoint
    * ``voice_design``  — prompt-capable OpenAI-compatible speech endpoint
    """

    enabled: bool = False
    backend: str = "edge_tts"
    voice: str = "en-US-AriaNeural"
    speed: float = 1.0
    model: str = "tts-1"
    api_url: str = ""
    api_key: str = ""
    # Override for the official OpenAI TTS backend (set to a self-hosted
    # OpenAI-compatible endpoint).  Empty → use the official OpenAI API URL.
    base_url: str = ""
    # HTTP request timeout for cloud TTS calls (openai / openai_compat).
    timeout: float = 60.0
    response_format: str = ""
    voice_profile: VoiceProfileConfig = field(default_factory=VoiceProfileConfig)
    speech_direction: SpeechDirectionConfig = field(
        default_factory=SpeechDirectionConfig
    )
    streaming: TTSStreamingConfig = field(default_factory=TTSStreamingConfig)
    # Provider-specific knobs are deliberately kept behind a generic mapping.
    # The voice_design backend currently understands num_steps, retries, and voice.
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class RVCConfig:
    """Voice conversion (RVC) configuration.

    When enabled, TTS audio is post-processed through an RVC model to apply
    voice conversion. Requires the ``rvc`` extra: ``uv sync --extra rvc``.

    Set ``model_path`` to the .pth checkpoint file.
    Set ``index_path`` to the .index file (optional, improves quality).
    ``f0_up_key`` shifts pitch (semitones, e.g. 0 = no shift, 12 = +1 octave).
    ``device`` selects compute device (``cuda``, ``cpu``, or empty for auto).
    """

    enabled: bool = False
    model_path: str = ""
    index_path: str = ""
    f0_up_key: int = 0
    device: str = ""


@dataclass
class MetricsConfig:
    """In-process metrics collection.

    When ``enabled`` is False, all observations become no-ops and
    :func:`cordbeat.metrics.render_prometheus` returns an empty string.
    The optional Prometheus HTTP endpoint is **off by default**;
    set ``prometheus_port`` to a non-zero value to enable it.
    Bind address defaults to loopback for the same security posture
    as the gateway WebSocket.
    """

    enabled: bool = True
    prometheus_host: str = "127.0.0.1"
    prometheus_port: int = 0


@dataclass
class ReActConfig:
    """Configuration for the ReAct multi-step skill execution loop."""

    enabled: bool = True
    max_iterations: int = 3
    max_actions_per_turn: int | None = None
    max_tool_output_chars: int = 4000
    continuation_max_tokens: int = 4000
    expose_trace_to_user: bool = False


@dataclass
class Config:
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    adapters: dict[str, AdapterConfig] = field(default_factory=dict)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ai_backend: AIBackendConfig = field(default_factory=AIBackendConfig)
    # Opt-in lightweight backend for ``respond_mode: ai_decision_llm``.
    # When ``None`` (default), the LLM judgement mode falls back to the
    # keyword-based ``ai_decision`` behaviour.  Configure with a small / fast
    # model (e.g. ``gemma2:2b``, ``qwen2.5:0.5b`` or BitNet b1.58) — it is
    # invoked per public-channel message and must answer yes/no quickly.
    ai_decision: AIBackendConfig | None = None
    soul: SoulConfig = field(default_factory=SoulConfig)
    log: LogConfig = field(default_factory=LogConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    rvc: RVCConfig = field(default_factory=RVCConfig)
    react: ReActConfig = field(default_factory=ReActConfig)
    skills_dir: str = "sandbox/skills"
    data_dir: str = "."

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


_SECRET_YAML_PATHS: tuple[tuple[str, ...], ...] = (
    ("gateway", "auth_token"),
    ("ai_backend", "api_key"),
    ("ai_backend", "options", "api_key"),
    ("ai_decision", "api_key"),
    ("ai_decision", "options", "api_key"),
    ("stt", "api_key"),
    ("tts", "api_key"),
)

_SECRET_YAML_ADAPTER_KEYS = ("token", "api_key", "bot_token", "webhook_secret")


_warned_paths: set[str] = set()


def _warn_secrets_in_yaml(raw: dict[str, Any], path: Path) -> None:
    """Emit a warning when secret-like keys are found directly in config YAML.

    Secrets should live in a ``.env`` file next to ``config.yaml`` and be
    referenced via ``CORDBEAT_*`` environment variables, not stored in YAML.
    Warnings are deduplicated — if ``load_config`` is called multiple times
    for the same file (e.g. during combined server + CLI startup), only the
    first call emits warnings.
    """
    if str(path) in _warned_paths:
        return
    _warned_paths.add(str(path))
    import logging as _logging

    _log = _logging.getLogger(__name__)

    def _check(d: Any, key_path: str) -> None:
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            full = f"{key_path}.{k}" if key_path else k
            if k in _SECRET_YAML_ADAPTER_KEYS and isinstance(v, str) and v:
                _log.warning(
                    "Secret key '%s' found in %s — "
                    "prefer env var CORDBEAT_%s "
                    "(or run `cordbeat setup` to auto-manage).",
                    full,
                    path.name,
                    full.upper().replace(".", "__"),
                )
            elif isinstance(v, dict):
                _check(v, full)

    for key_path in _SECRET_YAML_PATHS:
        node: Any = raw
        for part in key_path[:-1]:
            if not isinstance(node, dict):
                break
            node = node.get(part, {})
        else:
            leaf_key = key_path[-1]
            leaf_val = node.get(leaf_key) if isinstance(node, dict) else None
            if isinstance(leaf_val, str) and leaf_val:
                dotted = ".".join(key_path)
                env_name = "CORDBEAT_" + "__".join(k.upper() for k in key_path)
                _log.warning(
                    "Secret key '%s' found in %s — move it to .env: %s=<value>",
                    dotted,
                    path.name,
                    env_name,
                )

    adapters_raw = raw.get("adapters", {})
    if isinstance(adapters_raw, dict):
        _check(adapters_raw, "adapters")


def _load_dotenv(path: Path) -> None:
    """Load .env file into os.environ if it exists.

    Supports simple KEY=VALUE lines. Ignores comments and blank lines.
    Strips optional surrounding quotes from values and ignores comments after
    quoted values.
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
            value = _parse_dotenv_value(value)
            os.environ.setdefault(key, value)


def _parse_dotenv_value(value: str) -> str:
    """Parse the value half of a simple dotenv ``KEY=VALUE`` line."""
    value = value.strip()
    if not value:
        return ""

    quote = value[0]
    if quote in ("'", '"'):
        escaped = False
        out: list[str] = []
        for char in value[1:]:
            if escaped:
                out.append(char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                return "".join(out)
            else:
                out.append(char)
        return "".join(out)

    for marker in (" #", "\t#"):
        idx = value.find(marker)
        if idx != -1:
            value = value[:idx].rstrip()
            break
    return value


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


class ConfigValidationError(ValueError):
    """Raised when ``config.yaml`` contains structural errors that would
    cause silent crashes at runtime (e.g. options field declared but parsed
    as a string due to missing space after a YAML colon)."""


def validate_config(cfg: Config) -> None:
    """Validate a loaded :class:`Config` for structural correctness.

    Catches common YAML authoring mistakes that would otherwise cause a
    silent crash deep inside subsystem initialisation.  The most frequent
    culprit is a missing space after a colon (``key:value`` instead of
    ``key: value``), which YAML happily parses as a single bare scalar
    string and turns the parent mapping into a string — leading to an
    ``AttributeError: 'str' object has no attribute 'get'`` from inside an
    AI backend constructor.

    Raises :class:`ConfigValidationError` listing **all** detected problems
    at once so the user can fix them in a single edit pass.
    """
    errors: list[str] = []

    def _check_options(label: str, value: Any) -> None:
        if not isinstance(value, dict):
            errors.append(
                f"{label}.options must be a mapping, got "
                f"{type(value).__name__}={value!r}. "
                "Common cause: missing space after a colon (e.g. "
                "'enable_thinking:false' instead of 'enable_thinking: false')."
            )

    _check_options("ai_backend", cfg.ai_backend.options)
    if cfg.ai_decision is not None:
        _check_options("ai_decision", cfg.ai_decision.options)
    for name, adapter in cfg.adapters.items():
        _check_options(f"adapters.{name}", adapter.options)

    if cfg.gateway.max_message_bytes < 1_048_576:
        errors.append(
            "gateway.max_message_bytes must be at least 1048576 "
            "(1 MiB). Increase it for image attachments."
        )
    if not cfg.gateway.auth_token and not _is_loopback_host(cfg.gateway.host):
        errors.append(
            "gateway.auth_token is required when gateway.host is not a "
            "loopback address"
        )
    if cfg.heartbeat.proactive_user_cooldown_minutes < 0:
        errors.append("heartbeat.proactive_user_cooldown_minutes must be >= 0")
    if cfg.heartbeat.proactive_destination_cooldown_minutes < 0:
        errors.append("heartbeat.proactive_destination_cooldown_minutes must be >= 0")
    if cfg.heartbeat.max_proactive_messages_per_tick < 1:
        errors.append("heartbeat.max_proactive_messages_per_tick must be >= 1")
    if cfg.heartbeat.self_review_interval_ticks < 1:
        errors.append("heartbeat.self_review_interval_ticks must be >= 1")

    if errors:
        msg = "Invalid config.yaml:\n  - " + "\n  - ".join(errors)
        raise ConfigValidationError(msg)


def _resolve_relative_paths(config: Config, config_dir: Path) -> None:
    """Resolve relative paths in config relative to the config file directory.

    When the config lives outside CWD (e.g. ``~/.cordbeat/config.yaml``),
    paths like ``cordbeat.db`` must be anchored to the config
    directory so that data files are found regardless of the working
    directory the process was started from.
    """

    def _resolve(p: str) -> str:
        pp = Path(p)
        if not pp.is_absolute():
            return str((config_dir / pp).resolve())
        return p

    config.memory.sqlite_path = _resolve(config.memory.sqlite_path)
    config.soul.soul_dir = _resolve(config.soul.soul_dir)
    config.data_dir = _resolve(config.data_dir)
    config.skills_dir = _resolve(config.skills_dir)
    if config.log.file:
        config.log.file = _resolve(config.log.file)
    if config.rvc.model_path:
        config.rvc.model_path = _resolve(config.rvc.model_path)
    if config.rvc.index_path:
        config.rvc.index_path = _resolve(config.rvc.index_path)


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML file with env var overrides.

    Loading order (later wins):
        1. Dataclass defaults
        2. YAML file values
        3. .env file (same directory as config, or CWD)
        4. CORDBEAT_* environment variables

    Relative paths in the resulting config are resolved relative to the
    config file's parent directory so that ``~/.cordbeat/config.yaml``
    correctly anchors ``data/`` paths to ``~/.cordbeat/data/``.
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
        _warn_secrets_in_yaml(raw, path)

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
    if heartbeat.discovery_share_cooldown_minutes < 0:
        logging.getLogger(__name__).warning(
            "Invalid negative heartbeat.discovery_share_cooldown_minutes %r; "
            "disabling the discovery share cooldown",
            heartbeat.discovery_share_cooldown_minutes,
        )
        heartbeat.discovery_share_cooldown_minutes = 0
    if heartbeat.max_discovery_shares_per_day < 0:
        logging.getLogger(__name__).warning(
            "Invalid negative heartbeat.max_discovery_shares_per_day %r; "
            "disabling discovery sharing",
            heartbeat.max_discovery_shares_per_day,
        )
        heartbeat.max_discovery_shares_per_day = 0
    memory = _build_dataclass(
        MemoryConfig,
        raw.get("memory", {}),
    )
    ai_backend = _build_dataclass(
        AIBackendConfig,
        raw.get("ai_backend", {}),
    )
    ai_cache_raw = raw.get("ai_backend", {}).get("cache")
    if isinstance(ai_cache_raw, dict):
        ai_backend.cache = _build_dataclass(LLMCacheConfig, ai_cache_raw)

    ai_decision: AIBackendConfig | None = None
    ai_decision_raw = raw.get("ai_decision")
    if isinstance(ai_decision_raw, dict):
        ai_decision = _build_dataclass(AIBackendConfig, ai_decision_raw)
        decision_cache_raw = ai_decision_raw.get("cache")
        if isinstance(decision_cache_raw, dict):
            ai_decision.cache = _build_dataclass(LLMCacheConfig, decision_cache_raw)

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
    if soul.emotion_style not in {"full", "subtle", "off"}:
        logging.getLogger(__name__).warning(
            "Invalid soul.emotion_style %r; falling back to 'full'",
            soul.emotion_style,
        )
        soul.emotion_style = "full"
    if soul.absence_note_days < 0:
        logging.getLogger(__name__).warning(
            "Invalid negative soul.absence_note_days %r; disabling absence notes",
            soul.absence_note_days,
        )
        soul.absence_note_days = 0

    log_raw = raw.get("log", {})
    # Track whether log.file was explicitly set (even to "") in the YAML / env
    log_file_explicit = isinstance(log_raw, dict) and "file" in log_raw
    log = _build_dataclass(LogConfig, log_raw if isinstance(log_raw, dict) else {})

    metrics_raw = raw.get("metrics", {})
    if not isinstance(metrics_raw, dict):
        metrics_raw = {}
    metrics = _build_dataclass(MetricsConfig, metrics_raw)

    skills_raw = raw.get("skills", {})
    if not isinstance(skills_raw, dict):
        skills_raw = {}
    sandbox_raw = skills_raw.get("sandbox", {})
    if not isinstance(sandbox_raw, dict):
        sandbox_raw = {}
    skills = SkillsConfig(
        sandbox=_build_dataclass(SkillSandboxConfig, sandbox_raw),
    )

    stt_raw = raw.get("stt", {})
    if not isinstance(stt_raw, dict):
        stt_raw = {}
    stt = _build_dataclass(STTConfig, stt_raw)

    tts_raw = raw.get("tts", {})
    if not isinstance(tts_raw, dict):
        tts_raw = {}
    tts = _build_dataclass(TTSConfig, tts_raw)
    voice_profile_raw = tts_raw.get("voice_profile", {})
    if isinstance(voice_profile_raw, dict):
        tts.voice_profile = _build_dataclass(VoiceProfileConfig, voice_profile_raw)
    speech_direction_raw = tts_raw.get("speech_direction", {})
    if isinstance(speech_direction_raw, dict):
        tts.speech_direction = _build_dataclass(
            SpeechDirectionConfig, speech_direction_raw
        )
    streaming_raw = tts_raw.get("streaming", {})
    if isinstance(streaming_raw, dict):
        tts.streaming = _build_dataclass(TTSStreamingConfig, streaming_raw)
    if not isinstance(tts.backend_options, dict):
        tts.backend_options = {}

    rvc_raw = raw.get("rvc", {})
    if not isinstance(rvc_raw, dict):
        rvc_raw = {}
    rvc = _build_dataclass(RVCConfig, rvc_raw)

    react_raw = raw.get("react", {})
    if not isinstance(react_raw, dict):
        react_raw = {}
    react = _build_dataclass(ReActConfig, react_raw)

    cfg = Config(
        gateway=gateway,
        adapters=adapters,
        heartbeat=heartbeat,
        memory=memory,
        ai_backend=ai_backend,
        ai_decision=ai_decision,
        soul=soul,
        log=log,
        skills=skills,
        metrics=metrics,
        stt=stt,
        tts=tts,
        rvc=rvc,
        react=react,
        skills_dir=raw.get("skills_dir", "sandbox/skills"),
        data_dir=raw.get("data_dir", "."),
    )

    _resolve_relative_paths(cfg, path.resolve().parent)

    # Default log file: <data_dir>/logs/cordbeat.log
    # Only set if NOT explicitly configured in YAML/env (empty string = user-disabled)
    if not log_file_explicit and not cfg.log.file:
        cfg.log.file = str(Path(cfg.data_dir) / "logs" / "cordbeat.log")

    validate_config(cfg)

    return cfg
