# Configuration Reference

CordBeat is configured via a single `config.yaml` file. All fields have
sensible defaults — you only need to set what you want to change.

---

## Full Example

```yaml
gateway:
  host: "0.0.0.0"
  port: 8765

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
  sqlite_path: "data/cordbeat.db"
  chroma_path: "data/chroma"
  decay_rate: 0.1
  archive_threshold: 0.05
  conversation_history_limit: 20
  memory_search_results: 3

soul_dir: "data/soul"
skills_dir: "skills"
data_dir: "data"

adapters:
  discord:
    enabled: true
    core_ws_url: "ws://localhost:8765"
    options:
      token: "YOUR_DISCORD_BOT_TOKEN"
  telegram:
    enabled: true
    core_ws_url: "ws://localhost:8765"
    options:
      token: "YOUR_TELEGRAM_BOT_TOKEN"
  cli:
    enabled: true
    core_ws_url: "ws://localhost:8765"
```

---

## Field Reference

### `gateway`

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | string | `"0.0.0.0"` | Bind address for the WebSocket server |
| `port` | int | `8765` | WebSocket server port |

### `ai_backend`

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | string | `"ollama"` | AI provider (`ollama` or `openai`) |
| `base_url` | string | `"http://localhost:11434"` | API base URL |
| `model` | string | `"llama3"` | Model name |
| `timeout` | float | `120.0` | HTTP request timeout in seconds |
| `max_tokens` | int | `1024` | Maximum tokens for AI generation |
| `options` | dict | `{}` | Provider-specific options (passed directly to API) |

Common options for Ollama:

| Option | Type | Description |
|---|---|---|
| `num_predict` | int | Max tokens to generate (512+ recommended for thinking models) |
| `temperature` | float | Creativity (0.0 = deterministic, 1.0+ = creative) |
| `top_p` | float | Nucleus sampling threshold |

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
| `sqlite_path` | string | `"data/cordbeat.db"` | Path to SQLite database |
| `chroma_path` | string | `"data/chroma"` | Path to ChromaDB storage |
| `decay_rate` | float | `0.1` | Ebbinghaus forgetting curve decay rate |
| `archive_threshold` | float | `0.05` | Memory strength threshold for archival |
| `conversation_history_limit` | int | `20` | Max messages included in prompt context |
| `memory_search_results` | int | `3` | Max semantic/episodic search results per query |

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
| `soul_dir` | string | `"data/soul"` | Directory for SOUL YAML files |
| `skills_dir` | string | `"skills"` | Directory for skill plugins |
| `data_dir` | string | `"data"` | Base data directory |
