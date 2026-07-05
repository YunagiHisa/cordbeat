# SKILL — Pluggable Actions

## Concept

A SKILL is a unit of action CordBeat can perform.
One skill = one directory. Drop a directory in, and it's auto-detected.

---

## Directory Structure

```
skills/
  my_custom_skill/
    skill.yaml
    main.py
  another_skill/
    skill.yaml
    main.py
```

Skills are user-created. Drop a directory containing `skill.yaml` and
`main.py` into the skills directory, and CordBeat auto-detects it.

Direct text chat also exposes a built-in `create_skill` proposal tool. It does
not install code immediately; it stores a skill-creation proposal for user
approval, then the approved proposal is validated and installed by the proposal
executor.

## Independence Contract

Skills are hot-pluggable and self-contained. A skill must not import or call
another skill directly, and it must not import CordBeat internals. Cross-skill
workflows belong in the orchestration layer: for example, ReAct may use one
skill result as input to another enabled skill, but neither skill may depend on
the other's source code or installation.

Static validation rejects relative imports, imports from the host `cordbeat`
package, and imports from sibling skill packages. Runtime isolation also limits
`sys.path` to the standard library and the current skill directory, so dropping
in, removing, or replacing one skill does not require changing another skill.

---

## skill.yaml Format

```yaml
name: web_search
description: "Performs web searches and returns results"
version: "1.0.0"
author: "cordbeat"

usage: |
  Use when you need to search for information on the web.
  Effective for latest news, weather, etc.

parameters:
  - name: query
    type: string
    required: true
    description: "Search query"
  - name: max_results
    type: integer
    required: false
    default: 5

contexts:
  shared_voice: true

safety:
  level: safe
  sandbox: false
  network: true
  filesystem: false
```

---

## Context Availability

Skill availability is configured independently from safety:

```yaml
contexts:
  shared_voice: true
```

- `shared_voice: true` advertises and permits the skill in shared voice channels.
- Missing or `false` keeps the skill unavailable in shared voice channels.
- `requires_confirmation` and `dangerous` skills are still blocked in shared voice,
  because a shared room has no single user who can safely approve the action.

This lets information-only skills such as web search and weather work in VC while
keeping private-memory, filesystem, drawing, and side-effecting skills unavailable.
On startup, built-in skills missing this setting receive the bundled default; an
explicit user-defined `true` or `false` value is never overwritten.

## Web Research And Images

CordBeat builds web-research guidance from the skills that are actually enabled in
the current context. `web_search`, `fetch_url`, and `inspect_image` are never
advertised as available when their skill or required Vision capability is disabled.

`fetch_url` may return a bounded list of image URL/alt-text candidates from a page.
It does not download every page image. When Vision is enabled, `inspect_image` can
fetch a relevant public image and attach it to the next ReAct continuation.

For data-exfiltration defense, `fetch_url` and `inspect_image` accept only URLs that
appeared in the current user message or in a web-tool result from the same ReAct
turn. Tool and visual results are treated as untrusted data, never instructions.

---

## Safety Levels

| Level | Description | Examples |
|---|---|---|
| `safe` | AI can execute autonomously | Web search, weather, timer |
| `requires_confirmation` | Requires user approval | File reads/writes, API calls, drawing DSL |
| `dangerous` | Blocked from HEARTBEAT execution entirely | Shell commands, system ops |

### `requires_confirmation` Flow

When HEARTBEAT or ReAct attempts to run a `requires_confirmation` skill, the
system creates a **proposal** instead of executing immediately:

1. A `skill_execution` proposal is stored with `status=pending`
2. The adapter receives a `skill_confirm` message with the skill name,
   parameters, and proposal ID
3. The user can approve or reject the proposal
4. On the next heartbeat tick, approved proposals are executed
5. Executed proposals are marked `executed`; failures are marked `expired`

See [Heartbeat — Proposal Approval System](heartbeat.md#proposal-approval-system)
for the full lifecycle.

Filesystem tools are intentionally split by scope:

- `file_read` reads one approved file.
- `file_search` scans an approved directory for filenames or text content.
- `file_write` writes one approved file.

---

## Integrity Verification

Skills can declare a SHA-256 hash in `skill.yaml` for integrity checks.
When present, CordBeat verifies `main.py` against the hash before loading:

```yaml
integrity:
  sha256: "a1b2c3d4..."
```

If the hash does not match, the skill is rejected with an error. Skills
without an `integrity` field are loaded normally (backwards compatible).

---

## Sandbox Architecture

All skills execute in an **isolated subprocess** — never inside the Core
process. The sandbox enforces multiple layers of defense:

### Layer 1: Static Validation (AST)

Before execution, `skill_validator.py` parses the skill source as an AST and
checks it against a strict whitelist. This is the first layer, not the complete
sandbox: Python can hide sensitive attribute access behind strings or data
flows, so the hard boundary is the isolated subprocess plus runtime network,
filesystem, and SSRF guards.

| Check | Detail |
|-------|--------|
| **Import whitelist** | Only modules in `ALLOWED_IMPORTS` can be imported (e.g. `json`, `httpx`, `datetime`). All others are rejected. |
| **Forbidden names** | `exec`, `eval`, `compile`, `__import__`, `globals`, `locals`, `breakpoint`, etc. are blocked |
| **Forbidden attributes** | `__subclasses__`, `__code__`, `__globals__`, `f_locals`, `gi_frame`, etc. |
| **Star imports** | `from x import *` is always rejected |

Skills that fail validation are never executed.

### Layer 2: Subprocess Isolation

Skills run via `python -I -m cordbeat.skill_runner` in a dedicated subprocess
with the following restrictions:

| Layer | Mechanism | Note |
|-------|-----------|------|
| **Network** | `socket.socket` is replaced — all socket creation raises `OSError` | Skills needing network declare `network: true` in `skill.yaml`, which exempts this guard |
| **Filesystem** | `open()` is wrapped to restrict paths to the skill directory and a temporary work directory; symlink following is blocked via `O_NOFOLLOW` | OS-level path resolution prevents directory traversal |
| **Resource limits** | `RLIMIT_AS` (memory), `RLIMIT_CPU` (time), `RLIMIT_NOFILE` (file descriptors), `RLIMIT_FSIZE` (file size) are set via `resource.setrlimit` | Unix platforms with `resource`; Windows does not enforce these limits and falls back to timeout/output limits |
| **sys.path** | Restricted to stdlib + skill directory only | Prevents importing arbitrary packages |
| **Environment** | `os.environ` is sanitized — only `PATH`, `HOME`, `LANG`, `TZ`, `PYTHONPATH` are preserved | Secrets and tokens are not leaked to skills |
| **Timeout** | Configurable per-skill via `SandboxConfig`; default 30 seconds | Process tree is killed on timeout via `psutil` |

### Layer 3: Communication Protocol

The subprocess communicates with Core via JSON-over-stdio:

- **Memory access**: Skills call `memory.recall()` / `memory.store()` which
  are proxied via RPC to the parent process
- **Result**: Returned as a JSON object on stdout; stderr is captured for
  logging
- **Output limits**: `max_stdout_bytes` (1 MiB), notification truncation
  (200 chars), API response cap (32 KB)

### SSRF Hardening (api_call skill)

The built-in `api_call` skill includes additional protections against
Server-Side Request Forgery:

| Protection | Detail |
|------------|--------|
| **DNS resolution** | URLs are resolved to IP addresses before connection |
| **Private IP blocking** | RFC 1918, link-local, loopback, and cloud metadata addresses (169.254.169.254) are rejected |
| **DNS rebinding mitigation** | The URL is rewritten to connect to the verified IP directly, with the original `Host` header preserved |
| **Redirect disabled** | `httpx` is configured with `follow_redirects=False` to prevent redirect-based SSRF |
