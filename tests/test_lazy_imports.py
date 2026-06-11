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
from unittest.mock import patch

import pytest

from cordbeat.tools.wizard import _check_required_deps

# Absolute path to the src tree so the subprocess can locate cordbeat without
# an editable install.
_SRC = str(Path(__file__).parent.parent / "src")


_SUBPROCESS_TIMEOUT = 30  # seconds — prevents hangs in CI / on Windows


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
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
            env={**__import__("os").environ, "PYTHONPATH": _SRC},
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"Subprocess hung for >{_SUBPROCESS_TIMEOUT}s while blocking {block_pkg!r} "
            f"and importing {module!r}. "
            "This usually means a module-level import triggered a blocking call "
            "(e.g. network connect, heavy initialisation)."
        )


# ---------------------------------------------------------------------------
# sqlite_vec — used by memory/vector.py
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
        result = _run_import("sqlite_vec", "cordbeat.memory.vector", "VectorMemory")
        assert result.returncode == 0, (
            "cordbeat.memory.vector imported sqlite_vec at module level!\n"
            f"stderr: {result.stderr}"
        )

    def test_setup_wizard_importable(self) -> None:
        """cordbeat-init must start even without sqlite_vec installed."""
        result = _run_import("sqlite_vec", "cordbeat.tools.wizard")
        assert result.returncode == 0, (
            "cordbeat.tools.wizard (cordbeat-init entry point) crashed because "
            "sqlite_vec was imported at module level in its import chain!\n"
            f"stderr: {result.stderr}"
        )

    def test_engine_importable(self) -> None:
        result = _run_import("sqlite_vec", "cordbeat.core.engine", "CoreEngine")
        assert result.returncode == 0, (
            "cordbeat.core.engine imported sqlite_vec at module level!\n"
            f"stderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# sentence_transformers — used by memory/vector.py for local embeddings
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
        result = _run_import("sentence_transformers", "cordbeat.tools.wizard")
        assert result.returncode == 0, (
            "cordbeat.tools.wizard crashed because sentence_transformers was "
            "imported at module level in its import chain!\n"
            f"stderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# _check_required_deps() — friendly error, not a traceback
# ---------------------------------------------------------------------------


class TestCheckRequiredDeps:
    """cordbeat_init_cli must print a human-readable error and exit(1)
    — not crash with a traceback — when a required dep is absent."""

    @staticmethod
    def _import_with_missing(package: str) -> object:
        real_import = __import__
        heavy_dependencies = {"sqlite_vec", "sentence_transformers"}

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == package:
                raise ImportError(name)
            if name in heavy_dependencies:
                return object()
            return real_import(name, *args, **kwargs)

        return fake_import

    @pytest.mark.parametrize(
        ("missing_package", "install_name"),
        [
            ("sqlite_vec", "sqlite-vec"),
            ("sentence_transformers", "sentence-transformers"),
        ],
    )
    def test_missing_dependency_exits_with_install_hint(
        self,
        missing_package: str,
        install_name: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with (
            patch(
                "builtins.__import__",
                side_effect=self._import_with_missing(missing_package),
            ),
            pytest.raises(SystemExit, match="1"),
        ):
            _check_required_deps()

        assert install_name in capsys.readouterr().out

    def test_all_present_returns_normally(self) -> None:
        with patch("builtins.__import__", return_value=object()):
            _check_required_deps()
