"""Regression tests: heavy optional dependencies must NOT be imported at module
level.

If a package like ``sqlite_vec`` or ``sentence_transformers`` is imported at
the top of a module (outside any function), the **entire** import chain that
includes that module will crash with ``ModuleNotFoundError`` on any
environment where the package is absent — including fresh installs, CI jobs
that haven't run ``uv sync --extra …`` yet, and the setup wizard that runs
*before* the agent is fully initialised.

Each test here runs a subprocess with the suspect package name injected into
``sys.modules`` as ``None`` (which makes ``import <pkg>`` raise
``ModuleNotFoundError``) and then imports the CordBeat module under test.
A successful exit (rc=0) proves the module-level import chain is clean.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

# Absolute path to the src tree so the subprocess can locate cordbeat without
# an editable install.
_SRC = str(Path(__file__).parent.parent / "src")


def _run_import(
    block_pkg: str, module: str, symbol: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run a tiny Python snippet that blocks *block_pkg* then imports *module*.

    ``symbol`` is an optional attribute to import (e.g. ``"MemoryStore"``).
    """
    import_stmt = f"from {module} import {symbol}" if symbol else f"import {module}"
    code = textwrap.dedent(f"""\
        import sys
        # Simulate the package being absent by poisoning sys.modules.
        sys.modules[{block_pkg!r}] = None  # type: ignore[assignment]
        {import_stmt}
        print("ok")
    """)
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": _SRC},
    )


# ---------------------------------------------------------------------------
# sqlite_vec — used by memory.py and _memory_vector.py
# ---------------------------------------------------------------------------


class TestNoTopLevelSqliteVec:
    """sqlite_vec must only be imported inside functions, never at module level."""

    def test_memory_store_importable(self) -> None:
        result = _run_import("sqlite_vec", "cordbeat.memory", "MemoryStore")
        assert result.returncode == 0, (
            "cordbeat.memory imported sqlite_vec at module level!\n"
            f"stderr: {result.stderr}"
        )

    def test_memory_vector_importable(self) -> None:
        result = _run_import("sqlite_vec", "cordbeat._memory_vector", "VectorMemory")
        assert result.returncode == 0, (
            "cordbeat._memory_vector imported sqlite_vec at module level!\n"
            f"stderr: {result.stderr}"
        )

    def test_setup_wizard_importable(self) -> None:
        """cordbeat-init must start even without sqlite_vec installed."""
        result = _run_import("sqlite_vec", "cordbeat.setup_wizard")
        assert result.returncode == 0, (
            "cordbeat.setup_wizard (cordbeat-init entry point) crashed because "
            "sqlite_vec was imported at module level in its import chain!\n"
            f"stderr: {result.stderr}"
        )

    def test_engine_importable(self) -> None:
        result = _run_import("sqlite_vec", "cordbeat.engine", "CoreEngine")
        assert result.returncode == 0, (
            "cordbeat.engine imported sqlite_vec at module level!\n"
            f"stderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# sentence_transformers — used by _memory_vector.py for local embeddings
# ---------------------------------------------------------------------------


class TestNoTopLevelSentenceTransformers:
    """sentence_transformers is an optional heavyweight dep; must be lazy."""

    def test_memory_store_importable(self) -> None:
        result = _run_import("sentence_transformers", "cordbeat.memory", "MemoryStore")
        assert result.returncode == 0, (
            "cordbeat.memory imported sentence_transformers at module level!\n"
            f"stderr: {result.stderr}"
        )

    def test_setup_wizard_importable(self) -> None:
        result = _run_import("sentence_transformers", "cordbeat.setup_wizard")
        assert result.returncode == 0, (
            "cordbeat.setup_wizard crashed because sentence_transformers was "
            "imported at module level in its import chain!\n"
            f"stderr: {result.stderr}"
        )
