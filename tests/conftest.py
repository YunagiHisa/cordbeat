"""Shared test fixtures for CordBeat."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeChromaCollection:
    """Lightweight in-memory mock of a ChromaDB collection.

    Supports add(), query(), get(), and delete() with simple matching
    instead of real vector embeddings. Enough for unit tests.
    """

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        for i, doc_id in enumerate(ids):
            self._docs[doc_id] = {
                "document": documents[i],
                "metadata": metadatas[i] if metadatas else {},
            }

    def get(
        self,
        ids: list[str] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        if ids:
            entries = {k: v for k, v in self._docs.items() if k in ids}
        else:
            entries = dict(self._docs)

        result: dict[str, Any] = {"ids": list(entries.keys())}
        if not include or "documents" in (include or []):
            result["documents"] = [e["document"] for e in entries.values()]
        if not include or "metadatas" in (include or []):
            result["metadatas"] = [e["metadata"] for e in entries.values()]
        return result

    def delete(self, ids: list[str]) -> None:
        for doc_id in ids:
            self._docs.pop(doc_id, None)

    def query(
        self,
        query_texts: list[str],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        matches = []
        for doc_id, entry in self._docs.items():
            if where:
                skip = False
                for k, v in where.items():
                    if entry["metadata"].get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            matches.append((doc_id, entry))

        matches = matches[:n_results]
        if not matches:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        return {
            "ids": [[m[0] for m in matches]],
            "documents": [[m[1]["document"] for m in matches]],
            "metadatas": [[m[1]["metadata"] for m in matches]],
            "distances": [[0.1 * i for i in range(len(matches))]],
        }


class FakeChromaClient:
    """In-memory mock of chromadb.PersistentClient."""

    def __init__(self, path: str = "") -> None:
        self._collections: dict[str, FakeChromaCollection] = {}

    def get_or_create_collection(self, name: str) -> FakeChromaCollection:
        if name not in self._collections:
            self._collections[name] = FakeChromaCollection()
        return self._collections[name]


@pytest.fixture(autouse=True)
def _mock_chromadb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace chromadb.PersistentClient with FakeChromaClient globally.

    This avoids heavy ChromaDB initialization in tests, making the suite
    faster and removing filesystem side-effects from vector storage.
    """
    fake_module = MagicMock()
    fake_module.PersistentClient = FakeChromaClient
    monkeypatch.setitem(
        __import__("sys").modules,
        "chromadb",
        fake_module,
    )
