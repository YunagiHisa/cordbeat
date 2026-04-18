# CordBeat Documentation

## Design & Architecture

| Document | Description |
|---|---|
| [Philosophy](philosophy.md) | Core design philosophy — why CordBeat exists |
| [Architecture](architecture.md) | System architecture, message flow, and deployment |
| [Heartbeat](heartbeat.md) | The autonomous HEARTBEAT loop |
| [Soul](soul.md) | Identity, persona, and emotion system |
| [Memory](memory.md) | 4-layer memory architecture with forgetting model |
| [Skills](skills.md) | Pluggable skill system |
| [Gateway](gateway.md) | WebSocket adapter gateway for multi-platform support |
| [Validation](validation.md) | AI output validation and safety |

## Operations

| Document | Description |
|---|---|
| [Engine](engine.md) | CoreEngine message processing flow |
| [AI Backends](ai-backends.md) | Provider configuration (Ollama, llama.cpp, OpenAI-compatible) |
| [Config Reference](config-reference.md) | All `config.yaml` fields and defaults |
| [Deployment](deployment.md) | Local and Docker deployment guide |

## Getting Started

1. Run `cordbeat-init` — the setup wizard auto-detects your local AI backend
2. Run `cordbeat` to start
3. Run `cordbeat doctor` to verify your setup

See the [main README](../README.md) for more details, or
[Deployment](deployment.md) for Docker instructions.
