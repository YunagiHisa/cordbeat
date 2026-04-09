# Changelog

All notable changes to CordBeat will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- Core framework: SOUL, MEMORY, HEARTBEAT, SKILL, Engine, Gateway
- AI backend abstraction (Ollama, OpenAI-compatible)
- Platform adapters: Discord, Telegram, CLI
- 4-layer memory system (semantic, episodic, flashbulb, certain)
- AI output validation with retry logic
- Skill registry with SHA-256 integrity verification and sandbox support
- Conversation history and context management
- Automatic memory extraction from conversations
- Flashbulb memory for emotionally significant moments
- Sleep phase memory consolidation
- SOUL emotion engine with transitions, decay, and inference
- Environment variable and .env file config support
- Docker and docker-compose deployment
- CI pipeline (ruff, mypy, pytest on Python 3.11/3.12/3.13)
- Full documentation suite in docs/
- `prompt.py` module — centralized prompt building and input sanitization
- `extraction.py` module — AI-driven emotion inference and memory extraction
- `RetryableConnection` base class for adapter WebSocket reconnection
- Configurable `timeout` and `max_tokens` in `AIBackendConfig`
- Configurable `conversation_history_limit` and `memory_search_results`
  in `MemoryConfig`
- CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, CHANGELOG.md
- Docker HEALTHCHECK for core container
- 215+ tests with 80% coverage

### Changed
- Migrated to aiosqlite for async database operations
- `MemoryStore` refactored into facade with 4 internal classes
  (`_UserStore`, `_VectorMemory`, `_ConversationStore`, `_RecordStore`)
- `CoreEngine.handle_message()` split into `_resolve_user()` and
  `_generate_response()` phases
- Discord and Telegram adapters now extend `RetryableConnection`

### Fixed
- Redundant `except (json.JSONDecodeError, Exception)` in engine.py
- `dict(result)` TypeError when skill returns non-dict value
- `CancelledError` not collected after `queue_task.cancel()` in main.py
- Async safety, security, and performance issues (PR #16)
