# 💗 CordBeat

> **A local-first autonomous AI agent that thinks, acts, and stays by your side — powered by your own hardware.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Early Design](https://img.shields.io/badge/Status-Early%20Design-blue)]()
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
│  ┌─────────┐  ┌────────┐  ┌─────────┐  │
│  │  SOUL   │  │ MEMORY │  │  SKILL  │  │
│  └─────────┘  └────────┘  └─────────┘  │
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

┌─────────────────────────────────────────┐
│          Platform Adapters              │
│  Discord │ Telegram │ CLI │ (more...)   │
└─────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- llama.cpp or Ollama (for local AI inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/cordbeat.git
cd cordbeat

# Install dependencies
uv sync

# Install with Discord support
uv sync --extra discord

# Install with development tools
uv sync --extra dev
```

### Development

```bash
# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/

# Run tests
uv run pytest
```

---

## Project Structure

```
cordbeat/
├── .claude/               # Claude Code configuration
│   ├── settings.json      # Shared project settings
│   └── rules/             # Path-specific coding rules
├── src/cordbeat/          # Source code
├── tests/                 # Test suite
├── docs/                  # Documentation
├── CLAUDE.md              # Claude Code project instructions
├── pyproject.toml         # Python project configuration (uv)
└── LICENSE                # MIT License
```

---

## Documentation

See the [docs/](docs/) directory for detailed design and architecture documentation.

---

## Contributing

CordBeat is in early design stage. Contributions, ideas, and feedback are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Status: Early Design Phase — Architecture finalized, implementation starting*
