# Deployment Guide

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- A local AI backend — one of:
  - [Ollama](https://ollama.com/) (auto-detected at `localhost:11434`)
  - [llama.cpp](https://github.com/ggerganov/llama.cpp) server (auto-detected at `localhost:8080`)
  - Any OpenAI-compatible API

---

## Local Deployment

### Quick Start (Setup Wizard)

The fastest way to get started:

```bash
git clone https://github.com/YunagiHisa/cordbeat.git
cd cordbeat
uv sync

# Run the setup wizard — auto-detects Ollama or llama.cpp
cordbeat-init

# Start CordBeat
cordbeat
```

The wizard generates `~/.cordbeat/config.yaml` with sensible defaults.
No questions asked if Ollama or llama.cpp is already running.

Run `cordbeat doctor` to verify your setup at any time.

### Manual Setup

```bash
git clone https://github.com/YunagiHisa/cordbeat.git
cd cordbeat
uv sync
ollama pull qwen3.5:9b

# Copy and edit config
cp config.example.yaml config.yaml

# Start CordBeat Core
uv run cordbeat
```

### 2. Connect the CLI Adapter

```bash
uv run python -m cordbeat.cli_adapter
```

### 3. Platform Adapters

Each adapter runs as a separate process and connects to Core via WebSocket.

```bash
# Discord
uv sync --extra discord
cordbeat-discord

# Telegram
uv sync --extra telegram
cordbeat-telegram
```

Or run via module:

```bash
python -m cordbeat.adapter_runner discord config.yaml
python -m cordbeat.adapter_runner telegram config.yaml
```

---

## Docker Deployment

### Core Only

```bash
docker compose up core -d
```

### Core + Adapters

```bash
# With Discord
docker compose --profile discord up -d

# With Telegram
docker compose --profile telegram up -d

# Both
docker compose --profile discord --profile telegram up -d
```

### Ollama Connection

Ollama runs on the host machine. The container reaches it via
`host.docker.internal`. Set your `config.yaml`:

```yaml
ai_backend:
  provider: ollama
  base_url: "http://host.docker.internal:11434"
  model: "qwen3.5:9b"
```

### Volumes

| Path | Purpose |
|---|---|
| `./data:/app/data` | SQLite database + ChromaDB vectors |
| `./config.yaml:/app/config.yaml:ro` | Configuration (read-only) |

### Custom Build

```bash
# Core
docker build -t cordbeat-core .

# Discord adapter
docker build -t cordbeat-discord \
  --build-arg ADAPTER_EXTRA=discord \
  -f Dockerfile.adapter .

# Telegram adapter
docker build -t cordbeat-telegram \
  --build-arg ADAPTER_EXTRA=telegram \
  -f Dockerfile.adapter .
```

---

## Architecture Overview

```
┌──────────────────┐
│   CordBeat Core  │  ← cordbeat / docker: core
│   (port 8765)    │
│                  │
│  Gateway ←─── WebSocket ───→ Ollama (host:11434)
│    ↕                        │
│  Engine → AI Backend        │
│    ↕                        │
│  SOUL / MEMORY / SKILL      │
└──────────────────┘
        ↕ WebSocket
┌───────┼───────┐
│       │       │
Discord Telegram CLI
Adapter Adapter Adapter
  ↕       ↕       ↕
Discord Telegram Terminal
  API     API
```

Each adapter is a standalone process. They can run on the same
machine or on different hosts — just point `core_ws_url` to the
Core's WebSocket address.
