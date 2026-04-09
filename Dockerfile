# ── Build stage ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for caching
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

# Install dependencies (no dev extras)
RUN uv sync --no-dev --frozen

# ── Runtime stage ─────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/

# Ensure venv is on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Data directory for SQLite and ChromaDB
VOLUME /app/data

# Config file mount point
VOLUME /app/config.yaml

# Gateway WebSocket port
EXPOSE 8765

# Health check: verify the WebSocket server is accepting connections
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import socket; s=socket.create_connection(('localhost',8765),2); s.close()"

ENTRYPOINT ["python", "-m", "cordbeat.main"]
CMD ["config.yaml"]
