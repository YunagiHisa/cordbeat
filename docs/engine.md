# CoreEngine — Message Processing

## Overview

CoreEngine is the central processing unit of CordBeat. It consumes messages
from the global queue, resolves users, builds AI prompts using SOUL and
MEMORY context, and routes responses back through the gateway.

---

## Message Flow

```
Adapter
  → WebSocket
    → GatewayServer (queue)
      → CoreEngine.handle_message()
        ├── resolve/create user (MEMORY)
        ├── update user summary
        ├── build prompt (SOUL + MEMORY + history)
        ├── AI generate (AIBackend)
        ├── store conversation (MEMORY)
        └── send response → Gateway → Adapter
```

## Step-by-Step Processing

### 1. Message Filtering

Only `MESSAGE` and `LINK_REQUEST` types are processed; all others are
silently dropped.

### 2. User Resolution

The engine resolves the platform user to an internal CordBeat user:

1. Calls `memory.resolve_user(adapter_id, platform_user_id)` to look up
   an existing user via platform link.
2. If not found, creates a new user with ID `cb_{adapter_id}_{platform_user_id}`
   and links the platform identity.

### 3. Prompt Building

The system prompt is assembled from the SOUL snapshot:

- **Identity**: Name and personality traits
- **Emotion**: Current primary emotion and intensity
- **Immutable rules**: Always included, cannot be overridden

The user prompt includes:

- **User context**: Display name and known profile data
- **Conversation history**: Last 20 messages (user and assistant turns)
- **Current message**: The new user input

### 4. AI Generation

The assembled prompt is sent to the configured `AIBackend`. A 120-second
timeout is applied per request.

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

---

## Dependencies

| Component | Role |
|---|---|
| `AIBackend` | Text generation |
| `Soul` | Identity and personality snapshot |
| `MemoryStore` | User data, profiles, conversation history |
| `SkillRegistry` | Available skills (future: tool-use integration) |
| `GatewayServer` | Message routing to/from adapters |
