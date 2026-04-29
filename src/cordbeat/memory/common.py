"""Shared helpers and SQL schema for the memory subsystem.

This module is private (``_memory_*`` naming). Public consumers should import
from :mod:`cordbeat.memory`.
"""

from __future__ import annotations

from datetime import UTC, datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id       TEXT PRIMARY KEY,
    display_name  TEXT NOT NULL,
    last_talked_at TEXT,
    last_platform TEXT,
    last_topic    TEXT DEFAULT '',
    emotional_tone TEXT DEFAULT '',
    attention_score REAL DEFAULT 0.5,
    preferred_platform TEXT
);

CREATE TABLE IF NOT EXISTS platform_links (
    user_id          TEXT NOT NULL,
    adapter_id       TEXT NOT NULL,
    platform_user_id TEXT NOT NULL,
    linked_at        TEXT NOT NULL,
    PRIMARY KEY (adapter_id, platform_user_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS core_profiles (
    user_id  TEXT NOT NULL,
    key      TEXT NOT NULL,
    value    TEXT NOT NULL,
    PRIMARY KEY (user_id, key),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS certain_records (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    content    TEXT NOT NULL,
    record_type TEXT DEFAULT 'log',
    created_at TEXT NOT NULL,
    metadata   TEXT DEFAULT '{}',
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    adapter_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_conv_user_time
    ON conversation_messages (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_certain_user_type_time
    ON certain_records (user_id, record_type, created_at DESC);
"""


VECTOR_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS semantic_vectors USING vec0(
    user_id TEXT PARTITION KEY,
    embedding float[384]
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodic_vectors USING vec0(
    user_id TEXT PARTITION KEY,
    embedding float[384]
);

CREATE TABLE IF NOT EXISTS semantic_memory (
    id             TEXT PRIMARY KEY,
    vec_rowid      INTEGER NOT NULL UNIQUE,
    user_id        TEXT NOT NULL,
    content        TEXT NOT NULL,
    trust_level    TEXT NOT NULL,
    strength       REAL NOT NULL,
    emotion_weight REAL NOT NULL,
    created_at     TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    metadata_json  TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_semantic_user ON semantic_memory(user_id);

CREATE TABLE IF NOT EXISTS episodic_memory (
    id             TEXT PRIMARY KEY,
    vec_rowid      INTEGER NOT NULL UNIQUE,
    user_id        TEXT NOT NULL,
    content        TEXT NOT NULL,
    trust_level    TEXT NOT NULL,
    strength       REAL NOT NULL,
    emotion_weight REAL NOT NULL,
    created_at     TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    metadata_json  TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_memory(user_id);
"""


def ensure_aware(dt: datetime) -> datetime:
    """Return *dt* with UTC timezone attached if it was naive."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
