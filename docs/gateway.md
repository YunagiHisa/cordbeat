# Adapter Gateway

## Concept

The Adapter Gateway connects CordBeat Core to platform bots.  
Core runs as a WebSocket server; each adapter connects as a client.

---

## Overview

```
CordBeat Core
  └ Gateway Server (WebSocket)
        ↑↓ ws://
  ┌─────┼─────┐
Discord  Telegram  CLI
Adapter  Adapter  Adapter
  │        │        │
Discord  Telegram  Terminal
  Bot      Bot
```

---

## Communication

**WebSocket only** — always-connected, real-time, bidirectional.

- Supports HEARTBEAT-initiated messages (Core → Bot)
- Same mechanism for both directions
- Adapters don't need to know Core internals; Core doesn't need to know platform APIs

---

## Message Format

### Bot → Core (incoming)
```json
{
  "type": "message",
  "adapter_id": "discord",
  "platform_user_id": "123456789",
  "content": "Good morning!",
  "timestamp": "2026-03-31T08:00:00"
}
```

### Core → Bot (outgoing)
```json
{
  "type": "message",
  "adapter_id": "discord",
  "platform_user_id": "123456789",
  "content": "Good morning! How did yesterday's work go?",
  "timestamp": "2026-03-31T08:00:01"
}
```

### Message Types

| Type | Direction | Description |
|---|---|---|
| `message` | Bidirectional | Normal message |
| `heartbeat_message` | Core → Bot | HEARTBEAT-initiated message |
| `ack` | Bot → Core | Delivery acknowledgment |
| `error` | Bidirectional | Error notification |
| `link_request` | Bot → Core | Account linking token request |
| `link_confirm` | Bot → Core | Account linking token confirmation |

---

## Configuration

```yaml
gateway:
  host: "0.0.0.0"
  port: 8765

adapters:
  discord:
    enabled: true
    core_ws_url: "ws://localhost:8765"       # Same machine
    # core_ws_url: "ws://192.168.1.10:8765"  # Remote
  telegram:
    enabled: true
    core_ws_url: "ws://localhost:8765"
  cli:
    enabled: true
    core_ws_url: "ws://localhost:8765"
```
