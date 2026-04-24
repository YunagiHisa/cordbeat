"""Shared test fixtures for CordBeat."""

from __future__ import annotations

import hashlib
import math

import pytest
import sqlite_vec

import cordbeat._memory_vector as vector_module

_EMBEDDING_DIM = vector_module.EMBEDDING_DIM


def _fake_encode(text: str) -> list[float]:
    """Deterministic pseudo-embedding derived from a hash of *text*.

    Produces a unit vector so cosine/L2 distance behaves sensibly, giving
    vec0 something to order on in tests without needing a real model.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand the 32-byte digest to EMBEDDING_DIM floats by chaining hashes.
    data = bytearray()
    counter = 0
    while len(data) < _EMBEDDING_DIM * 4:
        data.extend(
            hashlib.sha256(digest + counter.to_bytes(2, "big")).digest()
        )
        counter += 1
    raw = data[: _EMBEDDING_DIM * 4]
    # Interpret each 4 bytes as an unsigned int, map into [-1, 1].
    vec = [
        (int.from_bytes(raw[i : i + 4], "big") / 0xFFFFFFFF) * 2.0 - 1.0
        for i in range(0, _EMBEDDING_DIM * 4, 4)
    ]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the sentence-transformers embedding with a hash-based stub.

    The real model is ~80MB and takes several seconds to load; tests only
    need deterministic, same-text-same-vector behavior on top of sqlite-vec.
    """

    async def fake_embed_text(text: str) -> bytes:
        return sqlite_vec.serialize_float32(_fake_encode(text))

    monkeypatch.setattr(vector_module, "embed_text", fake_embed_text)
