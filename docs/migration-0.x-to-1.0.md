# Migration Guide Draft: 0.x to Future Stable Release

This draft tracks migration notes for the planned stable release. This branch
does not declare the final stable version yet, but several defaults have already
been tightened for safety.

## Recommended Steps

1. Back up `~/.cordbeat/config.yaml`, `.env`, and the SQLite database.
2. Update the repository and run `uv sync`.
3. Run `uv run cordbeat-doctor`.
4. Review `config.example.yaml` for new sections such as `react`,
   `ai_decision`, `metrics`, `stt`, `tts`, and adapter response filters.
5. Start CordBeat with `uv run cordbeat server ~/.cordbeat/config.yaml`.

## Notable Changes

- Gateway samples now bind to `127.0.0.1` by default.
- Adapter secrets should live in `.env` via `CORDBEAT_*` variables.
- DM and public-channel conversation history are scoped separately.
- ReAct tool use is enabled by default, but non-safe skills require approval.
- `file_read` now requires approval before reading local files.

Database migrations run automatically on startup.
