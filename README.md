# 💗 CordBeat

> **A local-first autonomous AI agent that thinks, acts, and stays by your side — powered by your own hardware.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/YunagiHisa/cordbeat/actions/workflows/ci.yml/badge.svg)](https://github.com/YunagiHisa/cordbeat/actions/workflows/ci.yml)
[![Local AI First](https://img.shields.io/badge/AI-Local%20First-green)]()
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

---

## What is CordBeat?

CordBeat is an open-source autonomous AI agent framework designed from the ground up for **local AI backends**.

Unlike other agent frameworks that treat cloud APIs as the default and local models as an afterthought, CordBeat is built with local inference (llama.cpp, Ollama) as a **first-class citizen**. It runs a true autonomous HEARTBEAT loop — the AI evaluates its own situation and decides what to do next, without waiting to be asked.

Think of it as a companion that lives on your machine, knows you, and quietly acts on your behalf — 24/7.

---

## Why CordBeat?

| | Other Agents | CordBeat |
|---|---|---|
| AI Backend | Cloud-first | **Local-first** (llama.cpp / Ollama) |
| Autonomy | Checklist-driven | **AI self-evaluates & decides** |
| Memory | Shared / flat | **Per-user isolation** |
| Platforms | Single or tightly coupled | **Pluggable adapter layer** |
| Character | None / minimal | **SOUL system built-in** |

---

## Core Concepts

### 🫀 HEARTBEAT — True Autonomous Loop
The AI evaluates its own situation and decides: *"What should I do right now?"*  
No external triggers needed. It can choose to act, rest, or reach out to you.

### 🧠 SOUL — Identity & Persona
A structured definition of who the agent is: values, personality, goals, and constraints.  
Consistent behavior across all interactions and platforms.

### 💾 MEMORY — Per-User Isolated Context
Each user gets their own memory space. Long-term facts, preferences, and conversation history are stored independently.

### 🛠 SKILL — Pluggable Actions
Discrete capabilities the agent can invoke: web search, file manipulation, API calls, and more.  
Self-contained and easy to add or remove.

---

## Architecture

```
┌─────────────────────────────────────────┐
│              CordBeat Core              │
│                                         │
│  ┌─────────┐  ┌────────┐  ┌─────────┐   │
│  │  SOUL   │  │ MEMORY │  │  SKILL  │   │
│  └─────────┘  └────────┘  └─────────┘   │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │        HEARTBEAT Loop           │    │
│  │  evaluate → decide → act → rest │    │
│  └─────────────────────────────────┘    │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴────────┐
       │   AI Backend   │
       │  Abstraction   │
       └───────┬────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
 llama.cpp  Ollama   OpenAI-compatible
 (primary) (primary)   (optional)

┌─────────────────────────────────────────────────────────┐
│                   Platform Adapters                     │
│  Discord │ Telegram │ Slack │ LINE │ WhatsApp │ Signal  │
│                          │ CLI                          │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- A local AI backend — one of:
  - [Ollama](https://ollama.com/) (auto-detected at `localhost:11434`)
  - [llama.cpp](https://github.com/ggerganov/llama.cpp) server (auto-detected at `localhost:8080`)
  - Any OpenAI-compatible API

### Installation

```bash
# Clone the repository
git clone https://github.com/YunagiHisa/cordbeat.git
cd cordbeat

# Install dependencies
uv sync

# Install with Discord support
uv sync --extra discord

# Install with development tools
uv sync --extra dev
```

### Running CordBeat

The fastest way to get started is the **setup wizard**:

```bash
# Run the setup wizard — auto-detects Ollama or llama.cpp
cordbeat-init

# Start CordBeat
cordbeat
```

The wizard automatically detects your local AI backend and generates
`~/.cordbeat/config.yaml` with sensible defaults. No questions asked if
Ollama or llama.cpp is running.

<details>
<summary>Manual setup (without wizard)</summary>

```bash
# 1. Pull an AI model (e.g. qwen3.5:9b)
ollama pull qwen3.5:9b

# 2. Copy and edit the config
cp config.example.yaml config.yaml
# Edit config.yaml — set your model name and preferences

# 3. Start CordBeat Core
uv run cordbeat

# 4. In another terminal, connect the CLI adapter
uv run python -m cordbeat.cli_adapter
```

</details>

### Diagnostics

Run the built-in doctor to check your setup:

```bash
cordbeat doctor
```

### Docker

```bash
# Start Core only
docker compose up core -d

# Start Core + Discord adapter
docker compose --profile discord up -d

# Start Core + Telegram adapter
docker compose --profile telegram up -d

# Start everything
docker compose --profile discord --profile telegram up -d
```

> **Note:** Ollama runs on the host machine. The container reaches it via
> `host.docker.internal`. Set `ai_backend.base_url` to
> `http://host.docker.internal:11434` in your `config.yaml`.

### Platform Adapters

Adapters run as separate processes and connect to Core via WebSocket.

```bash
# Discord (requires discord.py)
uv sync --extra discord
cordbeat-discord

# Telegram (requires python-telegram-bot)
uv sync --extra telegram
cordbeat-telegram

# Slack (requires slack-sdk; Socket Mode — no public webhook needed)
uv sync --extra slack
cordbeat-slack

# LINE (requires line-bot-sdk; exposes a webhook HTTP server)
uv sync --extra line
cordbeat-line

# WhatsApp (Meta Cloud API; exposes a webhook HTTP server)
uv sync --extra whatsapp
cordbeat-whatsapp

# Signal (requires a local signal-cli JSON-RPC daemon)
uv sync --extra signal
cordbeat-signal
```

Configure tokens in `config.yaml`:
```yaml
adapters:
  discord:
    enabled: true
    options:
      token: "YOUR_DISCORD_BOT_TOKEN"
  telegram:
    enabled: true
    options:
      token: "YOUR_TELEGRAM_BOT_TOKEN"
  slack:
    enabled: true
    options:
      bot_token: "xoxb-..."    # Bot User OAuth Token
      app_token: "xapp-..."    # App-Level Token (connections:write)
  line:
    enabled: true
    options:
      channel_access_token: "..."
      channel_secret: "..."
      webhook_host: "0.0.0.0"
      webhook_port: 8080
      webhook_path: "/webhook"
  whatsapp:
    enabled: true
    options:
      access_token: "..."
      phone_number_id: "..."
      verify_token: "your-verify-token"
      webhook_host: "0.0.0.0"
      webhook_port: 8081
      webhook_path: "/webhook"
  signal:
    enabled: true
    options:
      rpc_url: "http://localhost:8088/api/v1/rpc"
      phone_number: "+1234567890"
      poll_interval: 2
```

> **Note:** Slack/LINE/WhatsApp/Signal adapters are v1.0+ scaffolds —
> platform-side plumbing is complete and reconnects to Core via
> WebSocket automatically, but each requires platform-specific setup
> (tokens, webhook registration, or a running `signal-cli` daemon).

### Development

```bash
# Run linter
uv run ruff check src/ tests/

# Run formatter check
uv run ruff format --check src/ tests/

# Run type checker
uv run mypy src/

# Run tests
uv run pytest
```

---

## Project Structure

```
cordbeat/
├── .github/workflows/     # CI (ruff, mypy, pytest on 3.11-3.13)
├── src/cordbeat/          # Source code
│   ├── models.py          # Core data models & enums
│   ├── config.py          # YAML config loader
│   ├── soul.py            # SOUL — identity, personality, emotion
│   ├── memory.py          # 4-layer memory (aiosqlite + ChromaDB)
│   ├── ai_backend.py      # AI abstraction (Ollama / OpenAI-compat)
│   ├── validation.py      # AI output validation & retry
│   ├── prompt.py          # Prompt building & input sanitization
│   ├── extraction.py      # AI-driven memory extraction
│   ├── skills.py          # Pluggable skill registry (with integrity check)
│   ├── gateway.py         # WebSocket gateway & adapter base
│   ├── heartbeat.py       # Autonomous HEARTBEAT loop (2-layer)
│   ├── heartbeat_proposals.py # Proposal execution (skills, traits)
│   ├── heartbeat_sleep.py # Sleep phase (diary, decay, promotion)
│   ├── engine.py          # Message processing engine
│   ├── setup_wizard.py    # Zero-friction setup wizard
│   ├── doctor.py          # Diagnostic health checks
│   ├── main.py            # Entry point — boots all subsystems
│   ├── cli_adapter.py     # Interactive CLI client
│   ├── discord_adapter.py # Discord bot adapter
│   ├── telegram_adapter.py# Telegram bot adapter
│   ├── slack_adapter.py   # Slack adapter (Socket Mode)
│   ├── line_adapter.py    # LINE adapter (webhook)
│   ├── whatsapp_adapter.py# WhatsApp Cloud API adapter (webhook)
│   ├── signal_adapter.py  # Signal adapter (signal-cli JSON-RPC)
│   └── adapter_runner.py  # Standalone adapter launcher
├── tests/                 # Test suite (536+ tests)
├── docs/                  # Design & architecture documentation
├── Dockerfile             # Core container image
├── Dockerfile.adapter     # Adapter container image
├── docker-compose.yml     # Full stack deployment
├── config.example.yaml    # Reference configuration
├── pyproject.toml         # Python project configuration (uv)
└── LICENSE                # MIT License
```

---

## Documentation

See the [docs/](docs/) directory for detailed design and architecture documentation.

---

## Contributing

CordBeat is in early design stage. Contributions, ideas, and feedback are welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and
pull request guidelines.

Please note that this project follows a [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Security

See [SECURITY.md](SECURITY.md) for our security policy, skill execution risks,
and how to report vulnerabilities.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*CordBeat is alive — local AI, your rules.*
