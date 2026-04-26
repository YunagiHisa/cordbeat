# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in CordBeat, please report it
responsibly:

1. **Do NOT open a public issue.**
2. Email the maintainer or use GitHub's
   [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability).
3. Include a clear description, steps to reproduce, and potential impact.

We aim to acknowledge reports within 48 hours and provide a fix or
mitigation plan within 7 days.

---

## Security Model

### Local-First Design

CordBeat is designed to run on your own hardware. By default:

- AI inference runs locally via Ollama (no data leaves your machine)
- All data (SQLite + ``sqlite-vec`` virtual tables) is stored locally
  under ``~/.cordbeat/``
- Embeddings are computed locally via ``sentence-transformers``
  (default model: ``all-MiniLM-L6-v2``, 384 dims)
- No telemetry, analytics, or background callbacks

### Identity & Database

- ``user_id`` is a random 32-character hex (``uuid4().hex``); platform
  account IDs (Discord, Slack, …) are stored only in the ``platform_links``
  table, so a leaked database dump cannot be linked back to external
  accounts directly.
- Each user has their own isolated memory space. ``sqlite-vec`` virtual
  tables are partitioned by ``user_id PARTITION KEY`` to enforce row-level
  separation at query time.

### Skill Execution Risks

CordBeat's skill system executes Python code from skill plugins. This is
the primary attack surface.

| Safety Level | Description | Default | Isolation |
|---|---|---|---|
| ``safe`` | No filesystem or network access by declaration | Enabled | In-process |
| ``requires_confirmation`` | May access filesystem or network; user approval required per call | Enabled | In-process, AST-validated |
| ``dangerous`` | Unrestricted system access | **Disabled by default** | Subprocess + AST validator + ``rlimit`` + timeout |

**Mitigations:**

- Each skill ships its own ``pyproject.toml`` and runs inside a per-skill
  ``uv``-managed virtualenv (``~/.cordbeat/skill-envs/<name>/``).
  Subprocesses are spawned with the env's interpreter, the ``-I``
  isolated flag, and ``PYTHONNOUSERSITE=1`` so the host's
  ``site-packages`` does not leak into skills.
- Skills are loaded only from the configured ``skills_dir`` — never from
  user input or network sources.
- ``skill.yaml`` declares ``safety_level``, ``sandbox`` mode, and required
  permissions (``network``, ``filesystem``).
- ``skill_validator.py`` AST-checks every skill at load time against an
  import allow-list (``ALLOWED_IMPORTS``); ``socket`` / ``ssl`` / direct
  subprocess use are rejected.
- ``skill_runner.py`` is self-contained (no ``cordbeat`` imports) so a
  compromised skill env cannot reach into the host package.
- ``dangerous`` skills additionally run under ``setrlimit`` (CPU, RSS,
  open files) on POSIX and ``asyncio`` ``wait_for`` timeouts on all
  platforms.

**Recommendations:**

- Only install skills from trusted sources.
- Review ``skill.yaml`` and ``main.py`` before enabling any third-party
  skill.
- Keep ``dangerous`` skills disabled unless you understand the risks.
- Run CordBeat in a container (Docker) for additional isolation.

### Outbound HTTP (``api_call`` skill)

The ``api_call`` skill is the only sanctioned path for AI-initiated
outbound HTTP. It hardens against common SSRF vectors:

- Allowed schemes: ``http`` and ``https`` only.
- DNS resolution happens up front; the resolved IP is rejected if it is
  private, loopback, link-local, multicast, reserved, or in any
  cloud-metadata range.
- The resolved IP is **pinned** for the actual connection and the
  ``Host`` header is fixed to the original hostname, mitigating DNS
  rebinding.
- Redirects are disabled (``follow_redirects=False``) so a 3xx cannot
  re-target the request at an internal IP.
- All HTTP is via ``httpx.AsyncClient``; raw ``socket`` / ``urllib`` /
  ``requests`` are forbidden by the validator.

### AI Output Validation

All AI-generated output passes through ``validation.py`` before being
stored or acted upon:

- Output is checked against configurable rules (length, format, blocked
  patterns).
- Invalid output triggers retries with escalating temperature.
- Validation failures are logged and the operation is safely skipped
  (raises ``OutputValidationError``).

### SOUL Permission Model

Identity (``Soul``) mutations are guarded by an explicit caller check.

- Every mutating method on ``Soul`` (``update_emotion``, ``update_name``,
  ``update_notes``, ``apply_trait_change``, ``update_quiet_hours``)
  requires a keyword-only ``caller: SoulCaller`` argument
  (``SYSTEM`` / ``AI`` / ``USER``).
- The permission matrix in ``soul.py`` rejects unauthorised callers by
  raising ``SoulPermissionError``.
- HEARTBEAT-proposed trait changes pass through an atomic
  approve-then-commit flow; AI cannot bypass user confirmation for
  ``USER``-only fields.

### Network Exposure

- The Gateway WebSocket server binds to **``127.0.0.1:8765`` by
  default** — local-only. Opening it to the network requires explicit
  config.
- Adapters authenticate to the Gateway via a shared HMAC token
  (``hmac.compare_digest``). The token is provisioned out-of-band via
  ``config.yaml`` and is never logged.
- For production / multi-host deployments, place the Gateway behind a
  reverse proxy with TLS and continue to bind locally.

### Logging

- Logs go through the standard ``logging`` module; rotation, level, and
  format are configured via ``LogConfig``.
- API keys, HMAC tokens, and Soul private notes are never logged.
- Boundary handlers (``heartbeat`` loop, ``adapter`` loop) use
  ``logger.exception`` to capture tracebacks without re-raising, so a
  single failed iteration does not crash the agent.

### Dependency Hygiene

- CI runs ``pip-audit`` against the resolved runtime dependency set on
  every push and pull request (``audit`` job in ``ci.yml``).
- ``pip-audit`` is also available locally as a dev dependency:
  ``uv run pip-audit -r <(uv export --no-emit-project --no-dev --format
  requirements-txt --quiet)``.
- Optional extras (``slack`` / ``line`` / ``whatsapp`` / ``signal``) are
  installed only when those adapters are enabled.

---

## Threat Model (Summary)

| Asset | Threat | Primary Mitigation |
|---|---|---|
| User memory DB | Local file exfiltration | OS file permissions; database does not contain platform IDs in plaintext (UUID indirection via ``platform_links``). |
| User memory DB | Cross-user leakage via search | ``sqlite-vec`` ``user_id PARTITION KEY``; every search is scoped to the calling user. |
| Host filesystem | Malicious skill writes outside sandbox | ``safe`` skills declare no FS perms; ``requires_confirmation`` requires per-call user approval; ``dangerous`` runs in a subprocess with ``rlimit`` + cwd-restricted temp dir. |
| Host process | Skill exhausts memory/CPU | ``setrlimit`` (POSIX) + ``asyncio.wait_for`` timeout in subprocess sandbox. |
| Internal network | SSRF via ``api_call`` | DNS resolved up front, private/metadata ranges blocked, IP-pinned connect, redirects disabled, ``Host`` header fixed. |
| Internal network | DNS rebinding to private IP after first lookup | IP pinning ensures the second resolution cannot retarget the connection. |
| Gateway port | Unauthenticated remote control | Default bind ``127.0.0.1``; HMAC token required for adapter handshake; ``hmac.compare_digest`` constant-time comparison. |
| Adapter webhooks (WhatsApp) | Spoofed inbound messages | ``X-Hub-Signature-256`` HMAC verification with ``app_secret``; missing/mismatched signatures rejected with HTTP 401. |
| AI output | Prompt injection causing forbidden actions | Memory and user content sanitised + length-capped before insertion; system prompt isolates instructions; output validation rejects out-of-spec responses. |
| Soul identity | AI self-modifies user-only fields | ``SoulCaller`` keyword-only enforcement; permission matrix raises ``SoulPermissionError``; trait changes require atomic approve-then-commit through HEARTBEAT proposal queue. |
| Secrets in config | Accidental commit | ``config.yaml`` is in ``.gitignore``; only ``config.example.yaml`` is tracked; never logged. |
| Dependencies | Known CVE in transitive dep | ``pip-audit`` job in CI fails the build on advisory matches. |

### Out of Scope

- Hardening against an attacker with **local OS-level access** to the
  user's machine (encrypt at rest with full-disk encryption if needed).
- Sandboxing **AI model output** itself: the model is assumed to be
  potentially adversarial — that is exactly what the validator,
  permission model, and skill sandbox defend against. We do **not**
  promise correctness or safety of LLM-generated text.
- Side-channels (timing, Spectre/Meltdown-class). The ``hmac.compare_digest``
  guard removes the most common timing leak; deeper micro-architectural
  attacks are not in scope for a local agent.

---

## Supported Versions

CordBeat is in early development. Security patches are applied to the
latest ``main`` branch only.

| Version | Supported |
|---|---|
| ``main`` (HEAD) | Yes |
| Older commits | No |
