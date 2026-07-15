"""Shared helpers and SQL schema for the memory subsystem.

This module is private (``_memory_*`` naming). Public consumers should import
from :mod:`cordbeat.memory`.
"""

from __future__ import annotations

from datetime import UTC, datetime

SCHEMA = """
-- Foreign keys document relationships but are not enforced by default:
-- CordBeat leaves PRAGMA foreign_keys OFF for legacy database compatibility.
-- Startup observes orphan rows via PRAGMA foreign_key_check in MemoryStore.
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
    received_at TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_conv_user_time
    ON conversation_messages (user_id, created_at DESC);

CREATE TRIGGER IF NOT EXISTS trg_conv_received_at_fallback
AFTER INSERT ON conversation_messages
WHEN NEW.received_at = ''
BEGIN
    UPDATE conversation_messages
    SET received_at = NEW.created_at
    WHERE id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS conversation_media_observations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id     INTEGER NOT NULL,
    media_kind     TEXT NOT NULL DEFAULT 'image',
    relation       TEXT NOT NULL DEFAULT 'attached',
    mime_type      TEXT NOT NULL DEFAULT '',
    content_sha256 TEXT NOT NULL DEFAULT '',
    summary        TEXT NOT NULL,
    source_ref     TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    FOREIGN KEY (message_id) REFERENCES conversation_messages(id)
);

CREATE INDEX IF NOT EXISTS idx_conv_media_message
    ON conversation_media_observations (message_id);

CREATE INDEX IF NOT EXISTS idx_certain_user_type_time
    ON certain_records (user_id, record_type, created_at DESC);
"""


CHANNEL_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_channels (
    user_id     TEXT NOT NULL,
    adapter_id  TEXT NOT NULL,
    channel_id  TEXT NOT NULL,
    is_dm       INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (user_id, adapter_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
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
    metadata_json  TEXT NOT NULL DEFAULT '{}',
    archived_at    TEXT,
    archive_reason TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_semantic_user ON semantic_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_semantic_user_archive
    ON semantic_memory(user_id, archived_at);

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
    metadata_json  TEXT NOT NULL DEFAULT '{}',
    archived_at    TEXT,
    archive_reason TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_episodic_user ON episodic_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_episodic_user_archive
    ON episodic_memory(user_id, archived_at);
"""


def ensure_aware(dt: datetime) -> datetime:
    """Return *dt* with UTC timezone attached if it was naive."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
