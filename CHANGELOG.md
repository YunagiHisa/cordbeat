# Changelog

All notable changes to CordBeat will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **Exception hierarchy.** New `cordbeat.exceptions` module introduces a
  `CordBeatError` root with typed subclasses (`SkillError`,
  `SkillValidationError`, `SkillExecutionError`, `SkillTimeoutError`,
  `SkillSandboxError`, `SoulPermissionError`, `AdapterError`,
  `ConfigurationError`). Internal raise sites now use these instead of bare
  `RuntimeError`/`ValueError`, making catch-by-type cleaner for integrators.
- **Platform adapter scaffolds.** Added `slack_adapter`, `line_adapter`,
  `whatsapp_adapter`, and `signal_adapter` modules wired into
  `adapter_runner` with dedicated CLI entry points
  (`cordbeat-slack-adapter`, `cordbeat-line-adapter`,
  `cordbeat-whatsapp-adapter`, `cordbeat-signal-adapter`) and matching
  `[project.optional-dependencies]` extras. Each scaffold early-returns on
  missing SDK so the base install stays slim.

### Changed
- **SOUL permission matrix is now enforced.** `Soul.add_memory`,
  `set_emotion`, `update_self_image`, `record_decision`, and
  `record_reflection` require a keyword-only `caller` argument matched
  against an internal `_PERMISSIONS` matrix; unauthorized writers raise
  `SoulPermissionError`. **Breaking change** for anyone calling these
  methods directly from external Python code — pass
  `caller="engine"` / `"extraction"` / `"heartbeat_proposals"` (or whatever
  component is legitimately writing).

### Fixed
- Removed unused-import ruff violation in `skills/api_call/main.py`.

### Security
- **Skill sandbox rewritten as subprocess isolation.** All skills now run in a
  separate Python process launched with `-I` (isolated mode), a pruned
  environment, restricted `sys.path`, and a minimum set of runtime guards
  installed before the skill module is imported. In-process monkey-patching
  of `socket`/`open` (easily bypassed) has been removed.
- **AST-based skill validator.** AI-proposed skill source is parsed and
  checked against an allowlist (imports, constructs, builtins) in
  `cordbeat.skill_validator`. The previous substring-regex check is gone.
  Module-level code execution is forbidden; only a small set of pure
  constructors (`frozenset`, `ipaddress.ip_network`, …) is permitted at
  module scope.
- **SSRF hardening in `api_call`.** Requests to private, loopback,
  link-local, multicast, reserved, and cloud-metadata (169.254.169.254 /
  fd00:ec2::254) addresses are rejected after DNS resolution. Redirects are
  disabled. Only `http`/`https` schemes are allowed.
- **Atomic proposal state transitions.** `MemoryStore.update_proposal_status`
  is now a conditional `UPDATE` that requires the caller-observed previous
  state, preventing lost updates under concurrent approval.

### Added
- `skills.sandbox` config block (`timeout_seconds`, `memory_limit_mb`,
  `max_output_bytes`, `allow_network_by_default`).
- `psutil` dependency for recursive subprocess termination.

### Changed
- `Skill.execute` signature: skills no longer run in-process by default.
  The `_test_callable` hook exists purely for unit tests that exercise
  skill interactions without subprocess overhead.
- `SkillRegistry` never imports skill code at load time; it only parses
  and validates it. Import happens inside the subprocess worker.

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
- Configurable `diary_max_tokens` and `facts_per_message_limit`
  in `MemoryConfig`
- Proposal approval system — structured pending/approved/rejected/executed
  lifecycle for skill executions, trait changes, and general improvements
- `requires_confirmation` skills now create approval proposals instead of
  being silently skipped
- SOUL trait change approval flow — AI can propose personality changes via
  `propose_trait_change` action; changes only apply after user approval
- Cross-platform account linking with secure tokens
  (`secrets.token_urlsafe`, single-use, 10-minute expiry)
- `LINK_REQUEST` / `LINK_CONFIRM` message handlers in CoreEngine
- CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, CHANGELOG.md
- Docker HEALTHCHECK for core container
- Built-in skills: file_read, file_write, timer, read_diary, shell_exec,
  web_search, weather, api_call
- AI-generated skill proposal feature (`PROPOSE_SKILL` action)
- Externalized hardcoded constants into `MemoryConfig` (46 settings total)
- `/link` and `/unlink` text commands with audit logging
- `/name`, `/quiet`, `/prefer` user commands and `preferred_platform` field
- Configurable logging via `config.yaml` (`log.level`, `log.format`)
- Timezone-aware datetimes throughout codebase
- Setup wizard (`cordbeat-init`) — zero-question bootstrapping with
  auto-detection of Ollama and llama.cpp
- `cordbeat doctor` diagnostic command — checks config, AI connectivity,
  data directories, memory DB, and skills
- SOUL `language` property for multilingual response support
- llama.cpp server auto-detection at `localhost:8080`
- 536+ tests with 92%+ coverage

### Changed
- Migrated to aiosqlite for async database operations
- `MemoryStore` refactored into facade with 4 internal classes
  (`_UserStore`, `_VectorMemory`, `_ConversationStore`, `_RecordStore`)
- `CoreEngine.handle_message()` split into `_resolve_user()` and
  `_generate_response()` phases
- Discord and Telegram adapters now extend `RetryableConnection`
- `heartbeat.py` split into three focused modules: `heartbeat.py`,
  `heartbeat_proposals.py`, `heartbeat_sleep.py`
- Eliminated all `type: ignore` comments (39 → 0) via typed accessors
- Replaced `dict(result)` coercion with proper type checks in engine
- Improved code quality: ABC base classes, routing tables, proposal filters

### Fixed
- Redundant `except (json.JSONDecodeError, Exception)` in engine.py
- `dict(result)` TypeError when skill returns non-dict value
- `CancelledError` not collected after `queue_task.cancel()` in main.py
- Async safety, security, and performance issues (PR #16)
- YAML injection in AI-generated skill proposals
- Japanese comments in codebase replaced with English
