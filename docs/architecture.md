# Architecture

## Overview

```
┌─────────────────────────────────────────────────┐
│                CordBeat Core Server              │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │   SOUL   │  │  MEMORY  │  │     SKILL     │  │
│  │  Persona │  │ Per-user  │  │   Actions     │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │            HEARTBEAT Loop                  │  │
│  │  evaluate → decide → act → update memory   │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │         AI Backend Abstraction             │  │
│  │   llama.cpp / Ollama / OpenAI-compat (opt) │  │
│  └────────────────────────────────────────────┘  │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │  Global Queue (single)                     │  │
│  │  Messages processed one at a time          │  │
│  └───────────────┬────────────────────────────┘  │
│                  │                               │
│  ┌───────────────▼────────────────────────────┐  │
│  │         Adapter Gateway                    │  │
│  │  Unified send/receive interface            │  │
│  └───────────────┬────────────────────────────┘  │
└──────────────────┼──────────────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
  [Discord]   [Telegram]    [CLI]
   Adapter     Adapter     Adapter
```

## Design Principles

### Runs Anywhere
- Core is a simple Python process
- Docker-ready (Proxmox, local PC, VPS — anywhere)
- Single YAML configuration file
- SQLite by default (no heavy DB server required)

### Adapter Gateway
Each platform bot doesn't need to know about CordBeat Core internals. The Gateway provides a unified interface:

```
Bot → Gateway → Queue → Core → AI → Core → Gateway → Bot
```

New platforms are added by creating an adapter — no Core changes needed.

## Message Processing

### Single Global Queue

```
Discord ──┐
Telegram ─┼──→ [Global Queue] → AI processes one at a time → Reply to each platform
CLI ──────┘
```

**Why a single queue?**
- Local AI (llama.cpp / Ollama) uses GPU/CPU fully for one request at a time
- Parallelism wouldn't improve speed — it would waste resources
- Eliminates memory access conflicts entirely
- Simpler implementation
- CordBeat is "a being by your side" — massive concurrent requests are not the use case

## User Management

### Internal User Model

Users are identified by a CordBeat-internal ID. Multiple platform identities can be linked to a single user:

```
User: cb_0001 (display: "Alex")
  ├── discord: 123456789
  └── telegram: 987654321
```

Platform linking uses a time-limited token flow, verified through the CLI.

## Security Architecture

CordBeat applies defense-in-depth across multiple layers:

```
┌──────────────────────────────────────────────────┐
│              Security Layers                      │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Skill Execution Sandbox                    │  │
│  │  AST validator → subprocess isolation       │  │
│  │  Resource limits, network/FS guards         │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Gateway Authentication                     │  │
│  │  HMAC token (hmac.compare_digest)           │  │
│  │  Default bind: 127.0.0.1 (localhost only)   │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  SSRF Hardening (api_call skill)            │  │
│  │  DNS resolution → private IP blocking       │  │
│  │  DNS rebinding mitigation → redirect block  │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Prompt Injection Defense                   │  │
│  │  Memory delimiters, content truncation      │  │
│  │  System prompt directive                    │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Data Integrity                             │  │
│  │  Atomic conditional state transitions       │  │
│  │  AI output validation (3 layers)            │  │
│  │  Skill integrity hash verification          │  │
│  └─────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

See individual documents for details:
- [Skills — Sandbox Architecture](skills.md#sandbox-architecture)
- [Gateway — Authentication](gateway.md#authentication)
- [Memory — Prompt Injection Defense](memory.md#prompt-injection-defense)
- [Validation — AI Output Validation](validation.md)
