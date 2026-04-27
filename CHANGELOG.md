# Changelog

All notable changes to CordBeat will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Changed
- **BREAKING: Skills now run inside per-skill ``uv``-managed Python
  environments.** Each builtin skill ships its own ``pyproject.toml``;
  on first execution the dependencies declared there are installed
  into ``~/.cordbeat/skill-envs/<skill_name>/`` (cached and re-used
  on subsequent runs, with SHA-256 hash invalidation when the
  ``pyproject.toml`` changes). The skill subprocess is spawned with
  the env's interpreter, the ``-I`` isolated flag, and
  ``PYTHONNOUSERSITE=1`` so the host's ``site-packages`` no longer
  leak into skills. The runner script (``skill_runner.py``) was made
  self-contained — it no longer imports anything from the
  ``cordbeat`` package — so skill envs do not need cordbeat itself
  installed. Skills without a ``pyproject.toml`` continue to run
  under the host Python (used by tests and AI-generated skills with
  no extra deps). Requires ``uv`` to be available on ``PATH``.
- **BREAKING: Memory decay is now lazy.** The nightly
  ``decay_and_archive_memories()`` batch (and the ``MemoryStore`` /
  ``HeartbeatSleep`` hooks that called it) has been removed. Memory strength
  is computed on every read inside ``search_semantic`` / ``search_episodic``
  using the same Ebbinghaus formula as before, and entries that fall below
  ``archive_threshold`` are physically deleted from sqlite-vec on access
  (flashbulb entries are exempt). Search now oversamples (``n_results * 3``)
  and ranks by composite ``score = strength / (1 + distance)`` so weak-but-
  close hits no longer dominate. Net behaviour for callers is unchanged
  except that ``metadata["strength"]`` reflects the freshly computed value.
  ``MemoryStore.calculate_strength()`` is retained as a diagnostic helper.
- **BREAKING: ``user_id`` is now a random 32-char UUID hex.** New users are
  assigned ``uuid4().hex`` instead of the previous
  ``cb_<adapter>_<platform_user_id>`` format. This decouples the internal
  primary key from external account identifiers so a leaked database dump
  cannot be linked back to Discord/Slack/etc. accounts directly. The mapping
  is preserved via the existing ``platform_links`` table. Legacy
  ``cb_…``-style IDs are no longer generated; existing databases must be
  recreated (destructive upgrade — pre-release, no migration provided).
- **BREAKING: Vector backend switched from ChromaDB to sqlite-vec.** Semantic
  and episodic memories are now stored in the same SQLite database via the
  ``sqlite-vec`` loadable extension (``vec0`` virtual tables with
  ``user_id PARTITION KEY``), and embeddings are produced locally by
  ``sentence-transformers`` (``all-MiniLM-L6-v2``, 384 dims). The ``chromadb``
  dependency is removed; ``chroma_path`` has been removed from
  ``MemoryConfig`` / ``config.yaml`` and existing ChromaDB data is **not**
  migrated (destructive upgrade; delete the old ``data/chroma`` directory).
  Search result metadata now preserves native types (e.g. ``"flashbulb":
  True`` rather than ``"True"``).
- **BREAKING: Minimum version bumped to 0.4.0.**

### Added
- **Optional LLM response cache.** New ``cordbeat.llm_cache.CachingBackend``
  wraps any ``AIBackend`` with an in-memory LRU+TTL response cache keyed on
  ``(model, system, prompt, temperature, max_tokens)``. Disabled by default
  (``ai_backend.cache.enabled = false``); when enabled it skips
  high-temperature calls (``> max_temperature``, default 0.2) so only
  deterministic-ish requests are memoized. Eviction on
  ``max_entries`` (default 256) and per-entry ``ttl_seconds`` (default 3600).
  Two new metrics — ``cordbeat_llm_cache_hits_total`` and
  ``cordbeat_llm_cache_misses_total{reason}`` — surface hit rates through the
  Prometheus exporter. ``create_backend()`` transparently wraps the chosen
  provider when caching is on, so existing call sites need no changes.
- **In-process metrics + optional Prometheus exporter.** New
  ``cordbeat.metrics`` registry exposes Counters and Histograms with
  Prometheus 0.0.4 text-format rendering — zero new runtime
  dependencies. Four key paths are now instrumented out of the box:
  HEARTBEAT tick latency/outcome, memory query latency by kind
  (semantic / episodic), skill execution latency/outcome by name and
  safety level, and LLM ``generate()`` latency/outcome by backend and
  model. A new ``MetricsConfig`` (``config.metrics``) gates collection
  globally (``enabled``, default true) and can opt into a tiny stdlib
  HTTP exporter via ``prometheus_port`` (default 0 = disabled) bound
  by default to ``127.0.0.1`` for the same loopback security posture
  as the gateway.
- **Schema migration framework + backup/restore CLI.** New
  ``cordbeat._memory_migrations`` module introduces a versioned,
  append-only migration framework. ``MemoryStore.initialize`` now
  applies pending migrations on every boot and records the latest
  applied version in a new ``schema_version`` table. Pre-framework
  databases (``users`` table present, no ``schema_version``) are
  silently fast-forwarded to version 1. Two new CLI entry points are
  registered: ``cordbeat-backup [destination] [--config PATH]`` runs
  an online SQLite backup via ``sqlite3.Connection.backup`` (no agent
  shutdown required); ``cordbeat-restore <source> [--yes]`` overwrites
  the live database, saving the previous file as ``*.pre-restore`` for
  rollback.
- **Dependency audit job in CI.** New ``audit`` job in ``.github/workflows/ci.yml``
  runs ``pip-audit`` against the resolved runtime dependency set
  (``uv export --no-dev``) on every push and pull request. Fails the
  build on advisory matches. ``pip-audit`` is also added to the
  ``dev`` extra for local use.
- **Updated security documentation.** ``SECURITY.md`` refreshed to match
  the current architecture (sqlite-vec + sentence-transformers, default
  ``127.0.0.1`` Gateway bind, HMAC adapter auth, per-skill ``uv``
  environments, AST validator, ``api_call`` SSRF/DNS-rebinding
  hardening, ``SoulCaller`` permission matrix). New "Threat Model"
  section enumerates assets / threats / mitigations and explicit
  out-of-scope items.
- **Memory integration test suite.** New ``tests/integration/`` directory with
  ``test_memory_vector.py`` exercising the full memory lifecycle (insert →
  semantic search → user isolation → emotional search → flashbulb → lazy
  decay) against a real ``sqlite-vec`` + ``sentence-transformers`` stack.
  Marked with the new ``integration`` pytest marker (informational; runs by
  default in CI). Coverage gate raised from 78% to **85%** (current: 90.85%);
  ``skill_runner.py`` and the optional Slack/LINE/WhatsApp/Signal adapter
  scaffolds are now omitted from coverage measurement since their execution
  paths are not reachable from the default test environment.
- **Composite index on ``certain_records``.** New
  ``idx_certain_user_type_time`` covering ``(user_id, record_type,
  created_at DESC)`` to keep sleep-phase queries fast as record volume
  grows.
- **CI Windows coverage + coverage artifact uploads.** CI now runs on both
  ``ubuntu-latest`` and ``windows-latest`` across Python 3.11/3.12/3.13
  (6 jobs total). ``coverage.xml`` is uploaded as a per-job artifact for
  post-mortem analysis. Coverage gate raised from 70% to 78%.
- **WhatsApp webhook signature verification.** `whatsapp_adapter` now
  verifies the Meta `X-Hub-Signature-256` header using a constant-time HMAC
  comparison against a new `app_secret` option. Missing or mismatched
  signatures are rejected with HTTP 401. When `app_secret` is empty the
  adapter logs a warning and runs in permissive mode (for local dev only).
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
- **Typed exceptions adopted at internal raise sites.** Infrastructural
  failures in `memory.py` (store-not-initialized guards, 5 sites),
  `ai_backend.py` (malformed backend response), `validation.py`
  (post-retry validation failure), and `skill_runner.py` (missing
  `execute()` / memory-proxy RPC failure) now raise the typed
  `CordBeatError` subclasses introduced in the prior release instead of
  bare `RuntimeError` / `ValueError`. New `SkillExecutionError` subclass
  added for the skill-runner cases.
- **SOUL permission matrix is now enforced.** `Soul.add_memory`,
  `set_emotion`, `update_self_image`, `record_decision`, and
  `record_reflection` require a keyword-only `caller` argument matched
  against an internal `_PERMISSIONS` matrix; unauthorized writers raise
  `SoulPermissionError`. **Breaking change** for anyone calling these
  methods directly from external Python code — pass
  `caller="engine"` / `"extraction"` / `"heartbeat_proposals"` (or whatever
  component is legitimately writing).

### Fixed
- **`api_call` skill: clarified SSRF pinning intent.** Removed the dead
  ``transport=httpx.AsyncHTTPTransport(local_address=None)`` shim
  (``local_address`` binds the *source* interface, not the destination
  and was not contributing to pinning). URL rewriting to the
  pre-verified resolved IP with ``Host`` header restoration is now the
  single source of truth for DNS-rebinding mitigation, with an updated
  comment explaining the mechanism.
- **Heartbeat timezone fallback logs a warning.** When
  ``zoneinfo.ZoneInfo(self._config.timezone)`` raises
  ``ZoneInfoNotFoundError`` (missing ``tzdata`` on Windows) or an
  invalid zone name is given, the loop now logs a ``WARNING`` naming
  the offending zone and pointing at the ``tzdata`` package, instead of
  silently dropping to UTC via a bare ``except Exception``.
- **Adapter user→channel caches are now bounded LRUs.**
  ``DiscordAdapter`` and ``SlackAdapter`` previously grew their
  ``_user_channels`` dicts without bound for long-running bot
  processes. Both now use an ``OrderedDict`` capped at 10,000 entries
  with least-recently-used eviction on insertion.
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
