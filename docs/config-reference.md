# Configuration Reference

CordBeat is configured via a single `config.yaml` file. All fields have
sensible defaults — you only need to set what you want to change.

Secrets such as gateway tokens and bot tokens should be supplied via `.env`
(`CORDBEAT_*` nested environment variables). The examples below intentionally
omit token values.

---

## Full Example

```yaml
gateway:
  host: "127.0.0.1"
  port: 8765
  # auth_token is normally supplied by .env:
  # CORDBEAT_GATEWAY__AUTH_TOKEN=...

ai_backend:
  provider: ollama
  base_url: "http://localhost:11434"
  model: "qwen3.5:9b"
  timeout: 120.0
  max_tokens: 1024
  options:
    num_predict: 512
    temperature: 0.8

heartbeat:
  default_interval_minutes: 60
  min_interval_minutes: 5
  max_interval_minutes: 1440
  quiet_hours_start: "01:00"
  quiet_hours_end: "07:00"

memory:
  sqlite_path: "cordbeat.db"
  decay_rate: 0.1
  archive_threshold: 0.05
  conversation_history_limit: 20
  memory_search_results: 3

soul_dir: "soul"
skills_dir: "sandbox/skills"
data_dir: "."

adapters:
  discord:
    enabled: true
    core_ws_url: "ws://localhost:8765"
    options:
      # token is normally supplied by .env:
      # CORDBEAT_ADAPTERS__DISCORD__OPTIONS__TOKEN=...
  telegram:
    enabled: true
    core_ws_url: "ws://localhost:8765"
    options:
      # token is normally supplied by .env:
      # CORDBEAT_ADAPTERS__TELEGRAM__OPTIONS__TOKEN=...
  cli:
    enabled: true
    core_ws_url: "ws://localhost:8765"
```

---

## Field Reference

### `log`

| Field | Type | Default | Description |
|---|---|---|---|
| `level` | string | `"INFO"` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `format` | string | `"%(asctime)s [%(name)s] %(levelname)s: %(message)s"` | Python `logging` format string |
| `file` | string | `""` | Path to a log file. Empty = stderr only. When set, a `RotatingFileHandler` is attached. |
| `max_bytes` | int | `10485760` | Maximum log file size in bytes before rotation (10 MiB) |
| `backup_count` | int | `5` | Number of rotated backup files to keep |

### `gateway`

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | string | `"127.0.0.1"` | Bind address for the WebSocket server |
| `port` | int | `8765` | WebSocket server port |
| `auth_token` | string | `""` | HMAC token for WebSocket authentication. `cordbeat-init` writes it to `.env` as `CORDBEAT_GATEWAY__AUTH_TOKEN`. Empty = auth disabled. |

### `ai_backend`

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | string | `"ollama"` | AI provider (`ollama`, `openai`, or `openai_compat`) |
| `base_url` | string | `"http://localhost:11434"` | API base URL |
| `model` | string | `"llama3"` | Model name |
| `timeout` | float | `120.0` | HTTP request timeout in seconds |
| `max_tokens` | int | `1024` | Maximum tokens for AI generation |
| `options` | dict | `{}` | Provider-specific options (passed directly to API) |
| `vision_enabled` | bool | `false` | Enable image attachments for vision-capable models |
| `video_enabled` | bool | `false` | Enable video attachments. Requires a video-capable model plus `ffmpeg` and `ffprobe` on `PATH`. |
| `video_content_part` | string | `"auto"` | Video request shape: `auto`, `image_url`, or `input_video`. `auto` uses `input_video` for llama.cpp compatibility and `image_url` otherwise. |
| `video_quality` | str | `balanced` | Frame budget preset: `conservative` (6 frames @ 448px), `balanced` (8 @ 448), `detail` (12 @ 448). A frame costs roughly 2000 prompt tokens, so these are sized to leave room for the conversation inside a 32k context. The three values below override it individually |
| `video_max_seconds` | float | `8.0` | Frame budget of the sampled clip (`video_max_seconds * video_fps`) |
| `video_fps` | float | `1.0` | Playback rate of the sampled clip |
| `video_max_edge_px` | int | `448` | Maximum width or height of the sampled frames |
| `video_scene_threshold` | float | `0.3` | How different two frames must look to count as a cut worth keeping. Frames are chosen as: first and last, then scene changes, then even spacing |
| `video_max_input_bytes` | int | `20971520` | Maximum source attachment size (20 MiB); larger files are rejected before download/Core transfer |
| `video_max_input_seconds` | float | `600.0` | Maximum accepted source duration (10 minutes) |
| `video_transcribe_audio` | bool | `true` | Extract video audio and transcribe it using the configured STT backend |
| `video_audio_chunk_seconds` | float | `60.0` | Audio chunk duration; each transcript chunk receives an approximate timestamp range |

Common options for Ollama:

| Option | Type | Description |
|---|---|---|
| `num_predict` | int | Max tokens to generate (512+ recommended for thinking models) |
| `temperature` | float | Creativity (0.0 = deterministic, 1.0+ = creative) |
| `top_p` | float | Nucleus sampling threshold |

Common options for `openai_compat`:

| Option | Type | Description |
|---|---|---|
| `api_key` | string | API key; prefer environment expansion such as `${GEMINI_API_KEY}` |
| `compatibility_mode` | string | `strict_openai`, `llama_cpp`, or `vllm`. Defaults to `strict_openai` for remote URLs and infers `llama_cpp` only for local host URLs. |
| `enable_thinking` | bool | Sent only in compatibility modes that support thinking control. `llama_cpp` sends both top-level `enable_thinking` and `chat_template_kwargs.enable_thinking`; `vllm` sends only `chat_template_kwargs.enable_thinking`; `strict_openai` sends neither. |
| `reasoning_effort` | string | Optional top-level OpenAI-compatible reasoning control (`none`, `minimal`, `low`, `medium`, or `high`). Sent only when `compatibility_mode` is `strict_openai`. |
| `reasoning_content_keys` | list[string] | Message fields treated as internal reasoning, for example `reasoning_content` or `reasoning` |
| `reasoning_strip_tags` | list[string] | Inline reasoning tags to strip from user-facing content |

### `heartbeat`

| Field | Type | Default | Description |
|---|---|---|---|
| `default_interval_minutes` | int | `60` | Default time between HEARTBEAT cycles |
| `min_interval_minutes` | int | `5` | Minimum allowed interval |
| `max_interval_minutes` | int | `1440` | Maximum allowed interval (24h) |
| `quiet_hours_start` | string | `"01:00"` | Start of quiet hours (UTC, HH:MM) |
| `quiet_hours_end` | string | `"07:00"` | End of quiet hours (UTC, HH:MM) |

### `memory`

| Field | Type | Default | Description |
|---|---|---|---|
| `sqlite_path` | string | `"cordbeat.db"` | Path to SQLite database |
| `decay_rate` | float | `0.1` | Ebbinghaus forgetting curve decay rate |
| `archive_threshold` | float | `0.05` | Memory strength threshold for archival |
| `conversation_history_limit` | int | `20` | Max messages included in prompt context |
| `image_summary_enabled` | bool | `true` | Store structured text-only visual observations for later conversation context when vision is enabled |
| `memory_search_results` | int | `3` | Max semantic/episodic search results per query |
| `recalled_episode_context_limit` | int | `4` | Max deduplicated episodic memories included per prompt |

### `adapters.<name>`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `true` | Whether the adapter is active |
| `core_ws_url` | string | `"ws://localhost:8765"` | WebSocket URL to CordBeat Core |
| `options` | dict | `{}` | Adapter-specific options |

Adapter-specific options:

| Adapter | Option | Description |
|---|---|---|
| `discord` | `options.token` | Discord bot token |
| `telegram` | `options.token` | Telegram bot token from @BotFather |

### Top-level

| Field | Type | Default | Description |
|---|---|---|---|
| `soul_dir` | string | `"soul"` | Directory for SOUL YAML files |
| `skills_dir` | string | `"sandbox/skills"` | Directory for skill plugins |
| `data_dir` | string | `"."` | Base data directory |

Formal Soul files are intentionally outside the sandbox. Use sandbox-local
files for drafts, reflections, and experiments; approved system flows apply
changes to `soul_dir`.
