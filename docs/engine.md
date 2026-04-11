# CoreEngine — Message Processing

## Overview

CoreEngine is the central processing unit of CordBeat. It consumes messages
from the global queue, resolves users, builds AI prompts using SOUL and
MEMORY context, and routes responses back through the gateway.

Processing is split into clear phases with dedicated helper modules:

- **Prompt building** is handled by `prompt.py` (`build_soul_system_prompt`,
  `build_context`, `sanitize`)
- **Memory extraction** (emotion inference, fact/episode extraction) is handled
  by `extraction.py` (`MemoryExtractor`)
- **User resolution** and **response generation** are internal engine phases

---

## Message Flow

```
Adapter
  → WebSocket
    → GatewayServer (queue)
      → CoreEngine.handle_message()
        ├── Phase 1: _resolve_user() — resolve/create user (MEMORY)
        ├── Phase 2: _generate_response() — build prompt → AI generate
        ├── Phase 3: store conversation (MEMORY)
        └── Phase 4: extract memories (MemoryExtractor, background)
```

## Step-by-Step Processing

### 1. Message Filtering

`LINK_REQUEST` and `LINK_CONFIRM` messages are routed to dedicated
handlers (see [Account Linking](#account-linking) below). `MESSAGE`
types proceed through the standard AI pipeline. All other types are
silently dropped.

### 2. User Resolution

The engine resolves the platform user to an internal CordBeat user:

1. Calls `memory.resolve_user(adapter_id, platform_user_id)` to look up
   an existing user via platform link.
2. If not found, creates a new user with ID `cb_{adapter_id}_{platform_user_id}`
   and links the platform identity.

### 3. Prompt Building

The system prompt is assembled via `prompt.build_soul_system_prompt()` from the
SOUL snapshot:

- **Identity**: Name and personality traits
- **Emotion**: Current primary emotion and intensity
- **Immutable rules**: Always included, cannot be overridden

The user context is assembled via `prompt.build_context()`:

- **User context**: Display name and known profile data
- **Semantic memories**: Known preferences and facts (via ChromaDB search)
- **Episodic memories**: Related past moments (via ChromaDB search)
- **Conversation history**: Last N messages (configurable via
  `memory.conversation_history_limit`, default 20)
- **Current message**: The new user input (sanitized)

### 4. AI Generation

The assembled prompt is sent to the configured `AIBackend`. The timeout is
configurable via `ai_backend.timeout` (default 120 seconds).

### 5. Error Handling

If AI generation fails for any reason:

- The error is logged via `logger.exception()`
- A `MessageType.ERROR` response is sent back to the adapter
- The failed exchange is **not** saved to conversation history

### 6. Conversation Storage

On success, both the user message and AI response are stored in the
conversation history via `memory.add_message()`.

### 7. Response Routing

The response is wrapped in a `GatewayMessage` with `MessageType.MESSAGE`
and sent back through the gateway to the originating adapter.

### 8. Memory Extraction (Background)

After the response is sent, `MemoryExtractor` runs two AI-powered tasks:

- **Emotion inference**: Asks the AI what emotion the exchange conveyed,
  updates the SOUL emotion state. High-intensity emotions (≥0.8) trigger
  a flashbulb memory.
- **Fact & episode extraction**: Extracts user preferences, facts, and
  notable episode summaries from the conversation. Stores them in the
  semantic and episodic memory layers.

These operations are non-blocking — failures are logged and silently skipped.

---

## Dependencies

| Component | Role |
|---|---|
| `AIBackend` | Text generation |
| `Soul` | Identity and personality snapshot |
| `MemoryStore` | User data, profiles, conversation history |
| `PromptBuilder` | System prompt and context assembly (`prompt.py`) |
| `MemoryExtractor` | Emotion inference and memory extraction (`extraction.py`) |
| `SkillRegistry` | Available skills (future: tool-use integration) |
| `GatewayServer` | Message routing to/from adapters |

---

## Account Linking

Users can link their accounts across multiple platforms (e.g. Discord +
Telegram) using a secure token flow. This bypasses the AI pipeline entirely.

### Flow

```
New Platform                          Existing Platform
    │                                       │
    ├─ LINK_REQUEST ──────→ Engine          │
    │                        │              │
    │   ←── ACK (token) ────┘              │
    │                                       │
    │   "Send this token from               │
    │    your other platform"               │
    │                                       │
    │                        ├── LINK_CONFIRM (token) ──┤
    │                        │              │
    │                        │  verify      │
    │                        │  link_platform()
    │                        │              │
    │                        └── ACK ──────→│
```

### Security Properties

- Tokens are generated with `secrets.token_urlsafe(16)`
- Default expiry: 10 minutes
- Single-use: tokens are marked as used after first verification
- The confirmer must have an existing linked account
- No AI inference is triggered during the linking process
