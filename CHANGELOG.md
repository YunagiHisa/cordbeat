# Changelog

All notable changes to CordBeat will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **Grounded same-server continuity.** Public Discord channels automatically
  contribute source-linked decisions, schedules, and announcements to isolated
  server indexes. Related notes are injected three at a time within that server
  or a mutual-member DM; DM content is never published back to a server.
- **Expression layer stage 1.** Conversation prompts now apply emotion-aware
  tone guidance (`soul.emotion_style`, default `full`) and provide a factual
  elapsed-days note after an absence (`soul.absence_note_days`, default `2`).
  Both behaviors can be disabled with `off` and `0`, respectively.
- **Timer reminder delivery.** Timer records now use aware UTC timestamps and
  bounded minute ranges, and HEARTBEAT delivers due reminders through the
  user's last-seen route outside proactive cooldowns. Delivery is deferred
  during quiet hours, limited to five per tick, and stops after five failures;
  malformed legacy timestamps are disabled without blocking later reminders.
- **Per-category thinking control.** OpenAI-compatible backends can override
  thinking for internal extraction, compression, sleep, and heartbeat work,
  while skills can select `auto`, `off`, or `force_on`. Draw DSL generation now
  forces thinking without changing chat or voice latency defaults.
- **Availability-aware web research and visual inspection.** System guidance now
  encourages fresh multi-source research only when the relevant skills are
  enabled. `fetch_url` returns bounded image candidates, and the new
  `inspect_image` skill supports on-demand multimodal ReAct inspection with
  URL-provenance and SSRF defenses.
- **Structured conversation media observations.** Vision-generated summaries are
  stored separately from user-authored conversation text, without persisting raw
  image bytes or feeding visual observations into long-term fact extraction.

### Fixed
- **Evidence-bounded tool responses.** Replies now distinguish the exact scope
  returned by tools (such as titles, snippets, metadata, transcripts, and page
  text) from inference or unavailable media, without forcing mechanical labels.
- **Interaction versus context gaps.** Reunion language now follows the user's
  latest interaction across scopes, while a DM or channel's own idle interval
  is used only to judge whether its previous topic remains active.
- **Conversation temporal grounding.** Platform send times now survive adapter
  delivery and conversation persistence separately from Core receipt times.
  Prompts timestamp history and media observations, show exact elapsed time,
  and avoid carrying short-lived states forward without evidence.
- **Draw DSL coordinate and color normalization.** RECT and ELLIPSE must use
  forward corner coordinates (`x2 > x1`, `y2 > y1`), rejecting the
  width/height-style boxes small models emit, and six-digit hex colors at
  fixed argument positions are repaired with the missing `#` instead of
  failing validation. Prompt guidance spells out both rules.
- **Version metadata consistency.** `cordbeat.__version__` now matches the
  `pyproject.toml` package version instead of a stale `0.1.0`.
- **Remaining audit cleanup.** `file_read` now streams bounded text prefixes,
  vision responses strip reasoning blocks, Telegram skill confirmations use
  plain text, and Discord VC speech keeps one latest pending reply instead of
  synthesizing and dropping overlaps. Web research docs now describe the known
  trust limitation of chained tool-result URLs.
- **Async reloads, identity races, and adapter recovery (audits #6/#8/#10/#11).**
  Skill registry reloads now build complete snapshots off-thread before an atomic
  swap, including proposed-skill file writes and all runtime reload paths. User
  creation and platform linking use conflict-safe inserts instead of
  check-then-act sequences. Signal polling exponentially backs off to 60 seconds
  during outages, resets after recovery, and suppresses repeated exception logs.
  Soul persistence documents the intentional synchronous, atomic write retained
  for its few-kilobyte snapshots and synchronous public mutation API.
- **Operational tooling fixes (audits #7/#8).** `cordbeat-doctor show-config`
  masks any key containing token/secret/key/password, so adapter-specific
  credentials (Slack `app_token`, LINE `channel_secret`, WhatsApp
  `access_token`/`app_secret`/`verify_token`) no longer appear in shareable
  output. `cordbeat-add` backs up `config.yaml` before rewriting it (YAML
  comments are lost on rewrite) and installs skill dependencies via
  `uv pip install` first, falling back to pip, so uv-managed venvs without
  pip work. Metrics counters and histograms take a per-metric lock so
  background-thread updates cannot race the `/metrics` render. The CLI
  adapter gives up waiting for a reply after 120 s instead of hanging on
  "(thinking...)". `stt.device` is now configurable for `whisper_local`
  (empty = cpu) so GPU hosts get real-time VC transcription.
- **Log/data hygiene and multilingual judge (audits #6/#10/#11/#12).**
  Heartbeat skill-execution logs now show redacted, truncated params and a
  bounded result preview instead of full payloads. Diary and
  conversation-compression prompts sanitize each message, cap the total
  transcript while keeping the newest messages, and mark it as data-only;
  their LLM outputs are stripped of reasoning artifacts before being stored.
  The ai_decision yes/no judge also accepts the Japanese affirmative so
  multilingual judge models no longer silently degrade to always-no, while
  unrecognized answers still fail closed.
- **ReAct reply-context URLs and thinking-budget retries.** URLs quoted in a
  replied-to message are now eligible for web tools (replying is an explicit
  user selection of that context), and the OpenAI-compatible backend retries
  without thinking when a chat response spends its whole token budget on
  reasoning and returns empty content.
- **Security-backlog hardening (audits #10/#12/#13/#14).** The sandbox
  runner now trims `sys.path` before deriving filesystem-guard read roots,
  so the runner's own package directory and PYTHONPATH injections are no
  longer readable by `filesystem: false` skills. `web_search` and `weather`
  gained the same SSRF resolution pre-flight as `api_call`/`fetch_url` for
  their fixed domains. Extracted facts, episode summaries, and sleep-promoted
  facts are capped at 500 characters before storage. Skill-confirmation
  proposal records now store a redacted, truncated parameter display in
  their content (full parameters remain in metadata for execution), and the
  SSRF guard docstring no longer overstates DNS-rebinding coverage for
  hostname-literal connects.
- **Heartbeat channel-scope privacy.** Layer-2 heartbeat evaluation now scopes
  recent conversation history to the user's last-seen adapter/channel/DM target
  when that routing metadata is available, preventing DM history from being
  fed into guild-targeted proactive decisions.
- **Low-risk dependency CVE updates.** Bumped `aiohttp`, `starlette`,
  `urllib3`, `idna`, and `msgpack` in `uv.lock` to address the approved
  low-risk vulnerability backlog while leaving `torch` and `pynacl` pinned for
  user-reviewed voice/RVC compatibility.
- **Grounding v2 for verified tool-action honesty.** Conversation ReAct tool
  outcomes are now recorded as bounded verified-action ledger entries and
  recent conversation plus heartbeat tool results are injected into chat
  context as data, while shared-voice context stays free of those private
  records and callers that do not load the ledger emit no ledger section at
  all. System, tool, and heartbeat prompts now separate intentions from
  verified external actions, and the memory-extraction prompt instructs the
  extractor not to store AI claims about hidden progress, files, or completed
  tasks unless the user independently confirmed them.
- **Built-in file-skill path confinement from source audit #14.** `file_read`,
  `file_write`, and `file_search` now reject absolute paths, drive letters,
  `..` traversal, and `~` home expansion, matching `file_delete`/`file_mkdir`,
  and all five file skills additionally reject rooted drive-relative paths
  such as `\foo` (which resolve to the current drive's root on Windows).
  These skills run with `filesystem: true` (bypassing the subprocess
  filesystem guard) and previously could read or overwrite any host file —
  including `~/.cordbeat/.env` and skill source — once approved.
- **Decided audit backlog hardening.** Sandboxed memory RPCs are now scoped to
  the acting user, with system execution left unscoped, a 50-call per-run cap,
  and 16,000-character truncation for `add_certain_record` content. ReAct
  continuation generation now uses its own `continuation_max_tokens` setting
  instead of reusing the tool-output character limit. Memory initialization logs
  `foreign_key_check` orphan counts without enforcing foreign keys. AI backend
  generation parameters now distinguish omitted values from explicit ones, so
  explicit low-temperature calls such as yes/no judges override configured
  defaults while ordinary calls still use config or built-in defaults.
- **Audit backlog cleanup.** Proposals orphaned in EXECUTING by a crash are
  now expired by the nightly sweep, an identical pending trait-change
  proposal is reused instead of piling up duplicates, the Gateway closes
  connections with invalid handshake JSON using a proper close code,
  `/name` validates and caps the new character name at 50 visible
  characters, and a skill whose single stdout line exceeds the sandbox
  limit raises the typed sandbox error instead of a bare ValueError.
- **Soul emotion resilience and prompt hygiene from source audit #13.** An
  invalid persisted emotion value or intensity in soul.yaml (e.g. a
  hand-edit typo) no longer crashes every message and heartbeat tick:
  emotion state is coerced defensively with a warning and heals itself on
  the next legitimate update. The ReAct tool-response name attribute is
  sanitized so a model-authored skill name cannot forge tool-response
  structure.
- **Vector memory resilience from source audit #12.** A memory row with
  corrupt metadata JSON no longer breaks semantic/episodic recall or
  memory writes for that user: corrupt rows are skipped with a warning
  (without archiving them) and a corrupt near-duplicate candidate falls
  back to a normal insert. Vector and metadata inserts roll back together
  on failure so aborted writes leave no orphan vectors.
- **Proposal pipeline and AI-retry robustness from source audit #11.** A
  proposal row with corrupt metadata JSON no longer blocks execution or
  expiry of every other proposal (invalid rows are detected in SQL via
  `json_valid`, surfaced, and expired), approved-skill results that are not
  dictionaries are summarized safely instead of mis-reporting a successful
  run as failed, result notifications and expiry transitions can no longer
  cascade into the executor loop, and the OpenAI-compatible no-think retry
  flattens vision message content to its text parts instead of embedding
  base64 image data into the retry prompt.
- **Draw truncation, Slack DM routing, and model datetime fixes from source
  audit #10.** Draw DSL normalization now closes REPEAT blocks that were cut
  open by the line cap instead of manufacturing a validation issue that
  forced every retry to fail, the Slack adapter respects
  `allow_dm_fallback=False` so channel replies no longer leak to DMs, ReAct
  pre-tool text strips [DRAW: ...] tags before flushing to the user, model
  dataclass timestamps default to timezone-aware UTC, and the exceptions
  module docstring matches the real hierarchy.
- **Telegram delivery and approval-UI hardening from source audit #9.** Telegram
  replies now split at the 4096-character API limit via a shared `split_message`
  helper (also reused by Discord), over-long photo captions are delivered as a
  follow-up text message instead of silently failing, skill-approval buttons on
  both Discord and Telegram verify the owner before acting and use neutral
  wording when ownership is unknown, and the Gateway closes a stale WebSocket
  before accepting a reconnecting adapter with the same ID.
- **AI validation and prompt robustness from source audit #8.** ReAct tool-error
  output is truncated and JSON-escaped like the success path, `validated_ai_json`
  survives validator crashes by retrying, summary/soul validators reject
  malformed types, invalid timezone strings fall back to UTC, corrupt recall-hint
  metadata rows are skipped instead of failing recall, and CLI clients rewrite
  wildcard gateway binds to a connectable loopback address.
- **Setup, service, and restore fixes from source audit #7.** The wizard and
  `cordbeat-add` now collect and write the credential keys each adapter actually
  reads (Slack bot+app tokens, LINE token+secret, WhatsApp token/phone/verify/app
  secret), the runner wires `rvc_config` so RVC voice conversion works in
  production, Windows service uninstall/stop/start also manage per-adapter
  scheduled tasks, restore validates the SQLite header and moves stale sidecar
  files aside, and the reasoning-leak heuristic no longer discards legitimate
  English replies on a single weak match.
- **Adapter input and quiet-hours hardening from source audit #6.** Quiet-hours
  values are range-validated (0-23/0-59) and invalid persisted values disable
  quiet hours with a warning instead of breaking heartbeat ticks, inbound
  adapter text is capped at 16k characters and blank messages are dropped,
  keyword-based respond modes fail closed when no keywords resolve, and skill
  environments build in a temp directory with an atomic rename so failed
  installs leave no partial venv.
- **Linking, proposals, and global-command hardening from source audit #5.**
  Platform links no longer silently repoint to another user, public `/link`
  replies no longer expose tokens, approved proposals are claimed before
  execution, global `/name` and `/quiet` writes require admin linkage, and
  proposal notifications respect adapter DM policy and last-seen channel
  routing metadata.
- **Memory, recall, heartbeat, and voice hardening from source audit #4.**
  Sleep consolidation now uses timezone-aware day windows instead of vector
  searching for `"today"`, duplicate memory merges reinforce from the current
  decay point, heartbeat messages reject unknown AI-supplied adapters, recalled
  memory data is stricter against prompt injection, voice receive tasks are
  bounded and supervised, and link tokens are consumed atomically.
- **Execution, voice, and tool hardening from source audit #3.** RVC checkpoint
  loading now uses PyTorch safe weights-only mode, skill parameters are coerced
  from `skill.yaml` types before execution, `shell_exec` cleans up timed-out
  processes, and shutdown/CLI task errors are bounded and logged.
- **Skill self-improvement hardening.** AI-generated skill metadata is now
  serialized with `yaml.safe_dump`, proposed parameters are schema-validated,
  skill metadata names must match their directory names, duplicate skill loads
  no longer replace the first loaded skill, and Python skill-file updates are
  validated before writing so failed self-repairs do not remove installed skills.
- **Gateway, Telegram, and auto-draw reliability fixes.** Telegram registers
  `/link_confirm` while routing it to Core's `/link-confirm`, Gateway reconnects
  no longer delete a newer connection, invalid Core JSON no longer resets
  adapters, auto-draw failures preserve the text reply, public Gateway binds now
  require auth, Draw DSL validates `REPEAT`/`END` balance and 3-point polygons,
  and Windows skill resource-limit caveats are documented and logged.
- **Draw DSL generation and interpreter consistency.** The generator now receives
  concrete composition/layering guidance, while common shapes consistently parse
  HSL colors and Turtle/gradient/shape validation matches the documented DSL.
  Safe coordinate and numeric corrections render without warnings, while missing
  or unknown DSL commands trigger regeneration before an incomplete image is sent.
- **Messages sent while Core is down are no longer lost.** Adapters dropped
  user messages when the WebSocket to Core was disconnected (e.g. during a
  Core restart). Each adapter now buffers outgoing messages in a bounded
  outbox (last 50) and resends them after reconnecting.
- **Near-duplicate memory dedup is now enabled by default.** The merge
  logic existed but `dedup_distance_threshold` defaulted to `0.0` (off),
  so repeated statements piled up as separate memories and crowded recall.
  The default is now `0.15` (≈ cosine similarity 0.99), merging only
  near-identical memories by reinforcing the existing one.
- **Familiarity now survives history trimming.** The relationship stage was
  derived from `COUNT(conversation_messages)`, but the nightly sleep phase
  trims that table to the most recent 100 rows — so a relationship could
  never grow past "acquaintance" no matter how long you talked. A new
  `users.total_messages` lifetime counter (migration v5, backfilled from
  existing history) now drives the familiarity stage.
- **soul.yaml / soul_notes.md writes are now atomic.** A crash or shutdown
  mid-write could truncate the personality file and corrupt the soul. Saves
  now write to a temp file and rename into place.
- **Message-loop crash no longer leaves a zombie process.** If the queue
  processing task died, the bot stopped responding while the process kept
  running silently. The task is now supervised: a crash logs CRITICAL and
  triggers a clean shutdown so a process manager can restart it.
- **Memory-extraction failures are now visible.** Extraction and emotion
  inference errors were logged at DEBUG, so memories could silently stop
  being recorded for days. They now log at WARNING with the cause.
- **Out-of-range HEARTBEAT intervals are logged.** An AI-proposed
  `next_heartbeat_minutes` outside the configured bounds was silently
  clamped; it now logs a warning so anomalies are noticeable.
- **`user_id` skill parameter no longer exposed to the AI.** The skill
  catalog shown to the model hid nothing, so `timer`/`read_diary` asked the
  AI to supply `user_id` — but the model only sees platform IDs (e.g.
  Discord snowflakes), not internal UUIDs. The catalog now omits `user_id`
  and the engine/heartbeat inject the code-level resolved user id at
  execution time, ignoring any AI-provided value.
- **Conversation summaries no longer preserve AI-failure narratives.**
  Compressed history summaries recalled as episodes could embed
  meta-commentary like "the assistant continued to fail", reinforcing bad
  behaviour in later prompts. The summarizer now focuses on the user's
  topics and requests and omits commentary about the assistant's
  performance.
- **Prompt hardening against phantom tool actions.** Production logs showed
  the model replying "I drew it, what do you think?" without emitting a
  `[DRAW: ...]` tag (so no image was ever rendered) and promising "I'll
  search and report back" without a `[SKILL: ...]` tag. The draw guidance now
  states that no image exists unless the same reply contains the tag, and the
  skill STRICT RULE now also forbids past-tense claims of completed
  searches/fetches when no tag or tool result backs them.
- **Spurious shutdown warning.** A cleanly cancelled `queue_task` re-raises
  `CancelledError` when awaited; the shutdown helper treated this as a
  timeout and logged `Shutdown timeout for queue_task (10s)` on every
  shutdown. Clean cancellation is now logged at DEBUG level and only a real
  `TimeoutError` produces the warning.
- **Contradictory thinking-retry log messages.** When a thinking model
  exhausted its token budget (`content=null` with `reasoning_content`), two
  back-to-back warnings claimed both "retrying with max_tokens=8192" and
  "retrying with thinking disabled". The retry (which disables thinking AND
  raises the token budget) now logs a single accurate warning.
- **Dead optional dependencies removed.** The `anthropic` and `openrouter`
  extras installed packages that no code imports (only `ollama` and
  `openai_compat` providers exist); the matching unused mypy override was
  also removed.
- **Skill approval and safety hardening.** ReAct now creates a
  `skill_confirm` proposal instead of silently skipping non-safe `[SKILL: ...]`
  calls, and adapter approval buttons now include the platform user id required
  by `/approve` / `/reject`. The unsupported `/approve_session` UI was removed.
  `file_read` now requires confirmation before reading local files, and the
  skill validator blocks `asyncio.create_subprocess_*` helpers for all
  non-dangerous skills.
- **Heartbeat draw proposal guard.** HEARTBEAT now refuses `draw` skill
  proposals whose `commands` parameter is natural-language text such as
  `DRAW: ...`; draw proposals must contain executable Draw DSL ending in
  `OUTPUT`.
- **Project cleanup.** Empty placeholder docs were filled as draft material, and
  leftover `.new` / append scratch files were removed.

### Added
- **Config validation at startup.** `load_config()` now runs a structural
  validation pass that detects common YAML authoring mistakes — most notably
  a missing space after a colon (`enable_thinking:false` instead of
  `enable_thinking: false`) which causes YAML to parse the parent `options:`
  mapping as a bare scalar string. Previously such errors caused a silent
  `AttributeError: 'str' object has no attribute 'get'` deep inside AI backend
  initialisation, which surfaced as a hung Core service with no log entry
  past `MEMORY store initialized`. The new `validate_config()` helper raises
  `ConfigValidationError` listing every detected problem at once. As
  belt-and-suspenders, `OllamaBackend` and `OpenAICompatBackend` also
  defensively coerce a non-dict `options` field to `{}` with a WARNING log.
- **ReAct multi-step skill execution loop.** Replaces the single-pass
  `_dispatch_skill_tags` with a full ReAct loop (`_react_loop`). The AI can now
  call multiple `[SKILL: name | key=value]` tags per response; results are fed
  back as `<tool_response>` blocks and the AI generates a fresh reply. Controlled
  by the new `react:` config section (`enabled`, `max_iterations`, `max_tool_output_chars`).
  Pre-tag text is flushed to the adapter immediately (D13); non-safe skills are
  skipped unchanged (D15); errors are returned as `{"error": "..."}` (D6);
  `</tool_response>` in tool output is escaped to prevent prompt injection (OQ2).
  A new `generate_chat()` method was added to all AI backends for multi-turn
  message-list generation.
- **`fetch_url` skill.** New builtin skill for reading a specific URL when the
  AI knows the exact page it wants (e.g. summarising an article the user
  shared). Strips HTML to text, decodes common entities, normalises
  whitespace, and truncates to a configurable `max_length` (default 8000,
  hard cap 64000). Reuses the same SSRF defenses as `api_call`: scheme
  allow-list (http/https), DNS resolution + private/loopback/link-local/
  multicast/metadata IP blocking, redirect disabling, and Host-header pinning
  to prevent DNS rebinding. Safety level `safe`. The `web_search` skill
  description was also clarified to tell the AI to use `fetch_url` for direct
  URL access (previously the AI sometimes passed URLs to `web_search`, which
  silently treated them as query strings).
- **Draw DSL skill — E-3.** New builtin skill `skills/draw/` implements a
  domain-specific language for programmatic image generation with Pillow. Supported
  commands: `SIZE`, `CANVAS`, `CIRCLE`, `RECT`, `LINE`, `TRIANGLE`, `STAR`,
  `SPIRAL`, `LAYER`, `IF`, `REPEAT`, and affine transforms (rotate/scale/translate).
  The `SAVE <filename>` command writes output to `~/.cordbeat/draw_output/` (fixed
  safe directory; user-supplied paths are stripped to basename to prevent traversal).
  Safety level `requires_confirmation`; dependency: `Pillow` only.
- **Configurable LLM/HTTP parameters (audit pass).** Several previously
  hardcoded values are now exposed in `config.yaml`:
  - `ai_decision.options.judge_max_tokens` / `judge_temperature` for the
    yes/no judge call (defaults `8` / `0.0`).
  - `memory.context_compression_temperature` / `context_compression_max_tokens`
    for the conversation summarisation LLM call (defaults `0.3` / `256`).
  - `stt.base_url` / `stt.timeout` (override official OpenAI STT endpoint /
    HTTP timeout). `tts.base_url` / `tts.timeout` likewise for the TTS side.
  - `adapters.signal.options.http_timeout` for signal-cli RPC requests
    (default `30.0`).
  All defaults reproduce the previous behaviour, so this is non-breaking.
- **Adapter respond-mode filtering — E-4.** All adapters now support a
  `respond_mode` option (`all` | `mention_only` | `ai_decision` |
  `ai_decision_llm`) plus `channel_whitelist`, `channel_blacklist`, and
  `user_blocklist` lists configurable in `config.yaml` under
  `adapters.<platform>.options`. In `mention_only` mode the bot replies only
  when directly mentioned or in a DM. `ai_decision` mode matches the soul
  name and `ai_decision_keywords` against the message text (no LLM call).
  `ai_decision_llm` mode (new) consults a small dedicated LLM configured
  under the top-level `ai_decision:` section to decide whether to interject;
  when no backend is configured it falls back to keyword matching.
- **Opt-in `ai_decision` AI backend.** New top-level `ai_decision:` config
  section accepts the same shape as `ai_backend:` (provider, model, host,
  api_key, cache, …) and is used exclusively by adapters running in
  `respond_mode: ai_decision_llm`. Defaults to `None` (disabled). Intended
  for tiny/fast judge models such as `gemma2:2b`, `qwen2.5:0.5b`, or
  BitNet b1.58. DMs always bypass the judge call.
- **DM and channel conversation history are stored separately.** Schema
  migration v4 adds `channel_id` and `is_dm` columns to
  `conversation_messages` (auto-applied on startup; legacy rows default to
  `is_dm=1`). The engine now scopes message retrieval to the originating
  venue, so private DM context no longer leaks into public-channel replies
  and vice versa. Heartbeat / tool paths that don't supply venue metadata
  see the full per-user history (unchanged behaviour).
- **Conversation history is also scoped per adapter.** `get_recent_messages`
  now accepts an `adapter_id` filter, and the engine passes the originating
  adapter (e.g. `discord`, `telegram`, `cli`). A user who chats with the
  bot on multiple platforms with the same canonical user-id no longer sees
  cross-platform context bleed. Heartbeat and tool paths still default to
  the unscoped legacy behaviour.
- **Package structure refactored into subpackages.** The previously flat
  `src/cordbeat/` namespace is now organised into six subpackages mirroring
  architectural domains: `adapters/` (Discord, Telegram, CLI, Slack, LINE,
  WhatsApp, Signal, runner, signing), `ai/` (backend, cache, extraction, prompt,
  STT, TTS, validation), `agent/` (heartbeat, proposals, sleep, soul), `core/`
  (engine, gateway), `memory/` (core, vector, common, conversation, records, users,
  migrations), `skills/` (registry, sandbox, runner, validator, env, rate_limit),
  `tools/` (doctor, wizard, metrics, metrics_server, backup, export, add_cmd).
  All public import paths are preserved via `__init__.py` re-exports so external
  code is unaffected. `CLAUDE.md` file-structure table updated.
- **Install scripts.** `install.sh` (Bash) and `install.ps1` (PowerShell) provide
  one-shot bootstrap: clone repo, verify Python 3.11+/uv, install runtime deps,
  run `cordbeat-init`. Supports optional `--extras` flag (e.g. `--extras stt-local`).
- **Discord VC voice-receive scaffold.** `voice_recv.py` provides a
  `VoiceReceiver` class (Opus decode → PCM buffer → silence detection → WAV
  16kHz mono) for future full Discord VC support. Excluded from default coverage
  measurement since it requires `discord.py[voice]` + `PyNaCl`.
- **RVC voice-conversion scaffold.** `rvc_backend.py` provides a `RVCBackend`
  class (HuBERT + RVC v2, F0 via pyworld/scipy) as an optional TTS post-processor.
  Wired behind `rvc.enabled` config flag; excluded from coverage since it requires
  the `rvc` extra (torch + transformers + torchaudio + pyworld).
- **Voice support (STT + TTS) — E-2.**Opt-in audio processing on Telegram and
  Discord adapters. Set `stt.enabled: true` and/or `tts.enabled: true` in
  `config.yaml` to activate. Telegram voices (`filters.VOICE`) are transcribed
  to text via STT; if TTS is enabled and the user sent a voice message, the
  reply is synthesised and sent as a voice note. Discord audio attachments
  (`audio/*`) are transcribed and appended to the message text before forwarding
  to the core engine.
  Three STT backends: `whisper_local` (faster-whisper, local GPU/CPU),
  `whisper_openai` (OpenAI API), `openai_compat` (whisper.cpp etc.).
  Three TTS backends: `edge_tts` (Microsoft Edge TTS, free), `openai` (OpenAI
  API), `openai_compat` (LocalAI etc.). Optional extras: `uv sync --extra
  stt-local` / `uv sync --extra tts-edge`. Both backends follow the ABC pattern
  of `ai_backend.py` for easy extensibility (VOICEVOX, Google, Azure…).
- **Vision / image support.** When `ai_backend.vision_enabled: true` is set in
  config, image attachments sent by users are forwarded to the LLM. Discord
  attachments with an `image/*` content type and Telegram photos are downloaded
  and base64-encoded before being included in the inference request.
  `OllamaBackend` uses the `/api/chat` endpoint with the `images` field (e.g.
  llava); `OpenAICompatBackend` uses the content-array vision format
  (`image_url` blocks with `data:` URIs). The cache layer passes vision calls
  straight through to the inner backend (no caching). A new
  `_detect_image_mime()` helper infers JPEG/PNG/GIF/WebP from magic bytes.

### Fixed
- **Heartbeat now skips ticks while user-message generation is in flight.**
  `HeartbeatLoop._tick` issued its own LLM calls (Layer 1 triage / Layer 2
  evaluation) on a fixed timer regardless of whether the engine was busy
  handling a real user message. On single-GPU local-inference setups this
  produced two parallel LLM calls competing for the same backend, causing
  timeouts and degraded UX. The default `MessageQueue` now exposes an
  `is_busy()` flag (handler running OR pending messages) and `_tick`
  returns `default_interval_minutes` early when the queue reports busy.
  Custom `MessageQueueProtocol` implementations should implement `is_busy`
  to opt in (returning a constant `False` keeps the previous behaviour).
- **ReAct: leaked `[SKILL: ...]` tags in final reply when iterations exhausted.**
  When `max_iterations` was reached and the AI was still emitting tool tags,
  `_react_loop` returned the raw response with the tag visible to the user
  (e.g. `Let me check [SKILL: web_search | query=...]`). The loop now strips
  any remaining tags from the final response before returning.
- **ReAct: empty final reply when AI emits tag-only responses.** If every
  ReAct continuation contained only `[SKILL: ...]` and no prose, stripping
  tags produced an empty string and the user got nothing back. The engine
  now substitutes a short Japanese acknowledgement when tools were actually
  executed but the model produced no closing text.

### Added
- **`voice_enable_thinking` per-context override for OpenAI-compat backend.**
  Thinking models (Qwen3, etc.) consume the entire token budget on
  `<reasoning>` for short voice replies, producing empty `content` and
  zero-length TTS output. The new `ai_backend.options.voice_enable_thinking`
  setting overrides `enable_thinking` only when the request originates from
  a voice channel/voice-message path. Activated via
  `voice_context_scope(is_voice)` around AI generation in the engine.
  `GatewayMessage.is_voice` is now plumbed through Discord (VC + audio
  attachment STT) and Telegram (voice handler) adapters.

### Changed
- **System prompt strengthened to enforce tool-promise consistency.** The
  prompt now lists the action verbs that MUST be paired with a `[SKILL: ...]`
  tag (look up/check/search/view/fetch/draw) and explicitly allows
  multiple tags per response (matching ReAct semantics). Previous wording
  said "include exactly one tag" which conflicted with multi-step ReAct
  behaviour.

### Fixed
- **`cordbeat server <path>` invocation now actually loads the given config.**
  systemd units installed via `cordbeat-init` use
  `ExecStart=cordbeat server <path>`, but `cli()` only handled `doctor` as
  a subcommand — `server` itself was treated as the config path, and the
  resulting `load_config("server")` call silently returned defaults
  (`provider=ollama`, `base_url=http://localhost:11434`). Production agents
  configured for `openai_compat` (e.g. llama.cpp / LM Studio) ended up
  trying to talk to a non-existent Ollama service. `cli()` now strips a
  leading `server` token from `sys.argv` so the rest of the path is
  resolved correctly. Existing systemd installs continue to work without
  modification.
- **Signal adapter `rpc_url` default URL aligned across config samples.**
  `config.example.yaml` and the bundled `config_template.yaml` previously
  documented `http://localhost:7583` while the adapter default and
  documentation use `http://localhost:8088/api/v1/rpc`.  All three now
  match the adapter default.
- **Heartbeat no longer sends unintended DMs (production incident fix).**
  Previously the Heartbeat loop's outbound path went `Heartbeat → Gateway →
  Discord adapter → DM fallback`: if the Discord adapter had no cached
  `channel_id` for the target user (cache cold-start, restart, or user never
  messaged via channel), `_send_to_discord` silently fell back to opening a
  DM. Combined with the Heartbeat having no notion of "which channel did this
  user last speak in", autonomous beats could surface as DMs to users who had
  only ever interacted in a guild channel — exactly what was observed in
  production for user `syach0323`. Fix:
  - New `user_channels` table (memory schema v3, auto-migrated) records
    `(user_id, adapter_id, channel_id, is_dm, last_seen_at)` on every inbound
    message; the engine writes this in `_resolve_user`.
  - Outbound `GatewayMessage.metadata` now carries `channel_id`, `is_dm`, and
    `allow_dm_fallback`. The Discord adapter prefers
    `metadata.channel_id` over its in-memory cache and only opens a DM when
    `allow_dm_fallback=True` (default true for normal replies, **false** for
    Heartbeat-initiated messages).
  - New `dm_policy` adapter option gates proactive sends from the Heartbeat:
    - `reply_only` (**new default**, *BREAKING*): only beat if we have a
      remembered non-DM channel; otherwise skip silently.
    - `allow_proactive`: send even with no history; allow DM fallback.
    - `never`: never beat at all.
  - The Heartbeat now also records its own outbound message as an `assistant`
    turn in conversation memory (previously it disappeared from history).
  - Deployments that *want* the old behaviour (proactive DMs) must explicitly
    set `adapters.<id>.options.dm_policy: allow_proactive` in `config.yaml`.

- **HEARTBEAT self-heals platform link for legacy users.** Heartbeats
  targeted at users created before commit `310deb4` (which lacked
  `platform_links` rows because the internal `user_id` was set to the
  platform identifier itself, e.g. a Discord snowflake or `cli_user`) now
  fall back to using `target_user_id` directly as the platform id and
  backfill the missing link row so subsequent resolves succeed. Previously
  these heartbeats logged `Cannot resolve platform_user_id` forever and were
  never delivered.
- **Path traversal in Draw skill `SAVE` command.** `SAVE <path>` previously
  accepted arbitrary filenames including `../../evil` and absolute paths. The fix
  strips all directory components with `Path(arg).name`, always writes into the
  fixed safe directory `~/.cordbeat/draw_output/`, and resolves the final path with
  `Path.resolve()` + `relative_to()` to catch symlink-based escapes. Two new
  security regression tests added.
- **Coverage gate restored to 85 %.** `main.py` (event-loop entry point),
  `tools/wizard.py` (interactive setup), and `tools/metrics_server.py` (HTTP
  daemon) added to `[tool.coverage.run] omit` with explanatory comments — standard
  exclusions for modules that cannot be exercised by a unit-test runner.

### Changed
- **Friendlier persona prompt with relationship-stage awareness.** The system
  prompt now includes an English tone block instructing the AI to "talk like a
  friend, not a butler-speak assistant", regardless of the configured response
  language. When prior conversation history is available, a relationship-stage
  hint is added based on the number of stored messages with the user
  (`stranger` <10, `acquaintance` <100, `friend` <500, `close friend` ≥500),
  letting the model calibrate warmth/familiarity automatically. The default
  soul trait set was broadened from `["curious", "playful"]` to
  `["curious", "playful", "warm", "candid", "emotionally expressive"]`;
  existing `soul.json` files are unaffected.
- **`cordbeat-init` ships a fully-annotated `config.yaml`.** The wizard now
  renders config from a bundled commented template (`config_template.yaml`,
  a copy of `config.example.yaml`) instead of `yaml.dump()`. The generated
  file preserves all section headers, defaults, and inline documentation so
  users can discover every option without reading the source. All secrets
  (adapter tokens, gateway `auth_token`, AI `api_key`) are routed exclusively
  through `.env`; on re-run, the wizard reads `auth_token` from either the
  existing `config.yaml` or `.env`.

### Changed

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

### Fixed
- **`cordbeat-init` crashed with ``ModuleNotFoundError: No module named 'sqlite_vec'``**
  even when ``sqlite-vec`` *is* listed as a dependency, because ``memory.py`` and
  ``_memory_vector.py`` imported it at module level. The import is now lazy
  (deferred to the first ``MemoryStore.initialize()`` / ``embed_text()`` call),
  so the setup wizard starts correctly and the missing-module error only surfaces
  when the agent actually tries to open the database — with a clear
  ``MemorySubsystemError`` message including the install command.

### Added
- **Conversation export CLI (`cordbeat-export`).** New entry point that
  reads the SQLite ``conversation_messages`` table directly (read-only,
  agent does **not** need to be stopped) and renders the history as
  JSON or Markdown. Supports per-user export with ``--user-id``,
  inclusive-since / exclusive-until date filtering
  (``--since YYYY-MM-DD --until YYYY-MM-DD``), output to a file
  (``--output PATH``) or stdout, and a no-arg listing mode that prints
  ``user_id / message_count / first / last`` for every user with stored
  messages. JSON output is UTF-8 and preserves non-ASCII characters
  verbatim; Markdown output uses ``> ``-quoted blocks so multi-line user
  messages render cleanly without colliding with section headings. The
  helpers ``fetch_messages`` / ``format_as_json`` / ``format_as_markdown``
  are also exposed from ``cordbeat.export_cli`` for programmatic use.
- **Per-skill rate limiting (token-bucket).** A new
  ``cordbeat.skill_rate_limit.SkillRateLimiter`` enforces a configurable
  per-minute call quota for each skill, defending against runaway loops
  (e.g., a misbehaving heartbeat plan that keeps invoking the same
  skill) and against billed external APIs being hammered. The bucket
  refills continuously at ``capacity / 60`` tokens per second using
  ``time.monotonic`` so wall-clock jumps don't corrupt accounting.
  Two layers of configuration:

  * ``skills.default_rate_limit_per_minute`` in ``config.yaml`` (default
    ``0`` = unlimited, preserving existing behavior for current users).
  * Per-skill override via ``rate_limit.per_minute`` in ``skill.yaml``.
    Negative values fall back to the default; ``0`` is "unlimited" for
    that skill.

  When the bucket is empty, ``Skill.execute()`` raises the new
  :class:`cordbeat.exceptions.SkillRateLimitError` with
  ``retry_after_seconds``, ``skill_name``, and ``limit_per_minute``
  attributes so callers (heartbeat loop, proposal executor) can decide
  whether to defer or skip. ``main.py`` now wires the limiter into
  ``SkillRegistry`` automatically. Ten new tests cover unlimited
  defaults, per-skill overrides, default enforcement, negative-fallback
  semantics, per-skill bucket isolation, ``reset()`` helpers, capacity
  rebuilds, and refill-rate accuracy.
- **Public webhook signature verification helpers for Slack and LINE.**
  New module ``cordbeat.adapter_signing`` exposes
  ``verify_slack_signature()`` (X-Slack-Signature ``v0=…`` HMAC-SHA256
  with built-in 5-minute replay-window enforcement per Slack's official
  guidance) and ``verify_line_signature()`` (X-Line-Signature
  base64-encoded HMAC-SHA256). Both use ``hmac.compare_digest`` and
  return ``False`` on any missing/malformed input so downstream webhook
  handlers can simply reject the request on a falsy return. The bundled
  Slack adapter still defaults to Socket Mode (no public webhook
  required); the LINE adapter still uses ``line-bot-sdk``'s
  ``WebhookParser``. These helpers exist for users opting into HTTP
  Events / webhook deployments and for centralizing the verification
  logic so it can be unit-tested independently. Twelve new tests cover
  positive cases, replay-window rejection (past *and* future skew),
  tampered-body detection, missing-prefix rejection, and missing-input
  edge cases.
- **Memory dedup at insert time + configurable embedding model.**
  ``MemoryConfig`` gains three new knobs: ``embedding_model`` (defaults to
  the existing ``sentence-transformers/all-MiniLM-L6-v2`` — swap for a
  multilingual model like ``paraphrase-multilingual-MiniLM-L12-v2``
  without code changes), ``embedding_dim`` (currently informational; the
  vec0 schema is fixed at 384, so a mismatch logs a warning), and
  ``dedup_distance_threshold``. When the threshold is > 0 (off by
  default), ``add_semantic_memory`` / ``add_episodic_memory`` first look
  up the nearest existing memory for the same user and, if its embedding
  distance is below the threshold, reinforce that record (strength
  bumped, ``last_accessed_at`` refreshed, ``emotion_weight`` taken as
  max) instead of inserting a duplicate row. Returns the existing id so
  callers see merge-as-insert semantics. The sentence-transformers model
  cache is now keyed by name so multiple models can co-exist in tests.
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
