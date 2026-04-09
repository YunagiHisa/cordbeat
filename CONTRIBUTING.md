# Contributing to CordBeat

Thank you for your interest in contributing to CordBeat! This guide covers the
development workflow and conventions.

---

## Development Setup

```bash
# Clone and install
git clone https://github.com/YunagiHisa/cordbeat.git
cd cordbeat
uv sync --extra dev

# Verify everything works
uv run ruff check src/ tests/
uv run mypy src/
uv run pytest
```

## Code Style

- **Formatter/Linter**: ruff (line-length=88)
- **Type checker**: mypy (strict mode)
- **Type annotations**: Required on all public functions
- **Imports**: `from __future__ import annotations` at the top of every module
- **Import order**: stdlib → third-party → local (enforced by ruff `I` rules)
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Paths**: Prefer `pathlib.Path` over `os.path`

## Git Workflow

### Branch Naming

- `feature/short-description` — new features
- `fix/short-description` — bug fixes
- `docs/short-description` — documentation changes
- `refactor/short-description` — code refactoring

### Commit Messages

- English, imperative mood (e.g., "Add heartbeat loop", not "Added heartbeat loop")
- Keep the first line under 72 characters
- Add a blank line and detailed description for complex changes

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run the full check suite:
   ```bash
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   uv run mypy src/
   uv run pytest
   ```
5. Push and open a Pull Request against `main`
6. Keep PRs focused on a single concern

## Architecture Guidelines

- **Adapter Gateway pattern**: Adapters must not know about Core internals
- **AI Backend abstraction**: Never import llama.cpp/Ollama/OpenAI SDK in core modules
- **Per-user memory isolation**: Never mix user contexts across operations
- **Single global queue**: No parallel AI inference
- **Config from YAML**: Never hardcode settings; add new fields to `config.py`
- **Skills are self-contained**: Each skill is a directory with `skill.yaml` + `main.py`

See [docs/architecture.md](docs/architecture.md) for the full system overview.

## Testing

- Tests live under `tests/`, mirroring the source structure
- Use `pytest` with `anyio` for async tests
- ChromaDB is automatically mocked via `FakeChromaClient` (conftest.py)
- Target: 70%+ code coverage (enforced in CI)
- Run a single test: `uv run pytest tests/test_specific.py::test_name -v`

## Project Structure

```
src/cordbeat/     # Source code (all modules)
tests/            # Test suite
docs/             # Public documentation
```

See the [README](README.md) for the full file listing.

---

## Questions?

Open an issue on GitHub if you have questions or ideas. We're happy to help!
