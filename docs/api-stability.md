# API Stability

This document is a draft stability target for future stable CordBeat releases.
The current branch does not declare a final stable version.

- CLI entry points declared in `pyproject.toml`
- `config.yaml` keys documented in `config.example.yaml`
- Gateway WebSocket message fields in `cordbeat.models.GatewayMessage`
- Skill metadata shape in `skill.yaml`
- Public `MemoryStore`, `SkillRegistry`, and adapter runner entry points

Internal modules may still change when needed for safety or maintainability.
Breaking changes to documented surfaces should be called out in `CHANGELOG.md`
and, when possible, include a migration note.
