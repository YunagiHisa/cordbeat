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
- All data (SQLite, ChromaDB) is stored locally
- No telemetry or external analytics

### Skill Execution Risks

CordBeat's skill system executes Python code from skill plugins. This is
the primary attack surface:

| Safety Level | Description | Default State |
|---|---|---|
| `safe` | No filesystem or network access | Enabled |
| `requires_confirmation` | May access filesystem or network | Enabled |
| `dangerous` | Unrestricted system access | **Disabled by default** |

**Mitigations:**

- Skills are loaded from a configured directory (`skills_dir`), not from
  user input or network sources
- Each skill declares its safety level, sandbox mode, and required
  permissions (`network`, `filesystem`) in `skill.yaml`
- Sandboxed skills run in a temporary working directory
- Skill file integrity is verified via SHA-256 hash on load
- `dangerous` skills are disabled by default and must be explicitly enabled

**Recommendations:**

- Only install skills from trusted sources
- Review `skill.yaml` and `main.py` before enabling any third-party skill
- Keep `dangerous` skills disabled unless you understand the risks
- Run CordBeat in a container (Docker) for additional isolation

### AI Output Validation

All AI-generated output passes through the validation module before being
stored or acted upon:

- Output is checked against configurable rules (length, format, blocked patterns)
- Invalid output triggers retries with escalating temperature
- Validation failures are logged and the operation is safely skipped

### Memory Isolation

- Each user has their own isolated memory space
- Platform identities are linked to internal user IDs
- No cross-user data leakage by design

### Network Exposure

- The Gateway WebSocket server binds to `0.0.0.0:8765` by default
- In production, place it behind a reverse proxy with TLS
- Adapter connections are authenticated by adapter ID in the WebSocket handshake

---

## Supported Versions

CordBeat is in early development. Security patches are applied to the
latest `main` branch only.

| Version | Supported |
|---|---|
| `main` (HEAD) | Yes |
| Older commits | No |
