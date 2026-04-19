"""Tests for the AST-based skill source validator."""

from __future__ import annotations

import pytest

from cordbeat.skill_validator import SkillValidationError, validate_skill_source

VALID_SKILL = '''"""A simple valid skill."""

from __future__ import annotations

import json
from datetime import datetime

from typing import Any


def _helper(x: int) -> int:
    return x + 1


def execute(**kwargs: Any) -> dict[str, Any]:
    now = datetime.now().isoformat()
    return {"now": now, "data": json.dumps(kwargs)}
'''


def test_valid_skill_passes() -> None:
    validate_skill_source(VALID_SKILL, "ok")


def test_empty_source_rejected() -> None:
    with pytest.raises(SkillValidationError, match="empty"):
        validate_skill_source("", "x")


def test_syntax_error_rejected() -> None:
    with pytest.raises(SkillValidationError, match="syntax error"):
        validate_skill_source("def execute(**kw)\n    pass\n", "bad")


def test_missing_execute_rejected() -> None:
    src = '"""No execute."""\n\ndef other(): pass\n'
    with pytest.raises(SkillValidationError, match="execute"):
        validate_skill_source(src, "no_exec")


def test_execute_with_explicit_params_accepted() -> None:
    # Explicit parameter lists are fine; parent passes kwargs by name.
    src = "def execute(a, b): return {}\n"
    validate_skill_source(src, "explicit_params")


def test_async_execute_accepted() -> None:
    src = "async def execute(**kw): return {}\n"
    validate_skill_source(src, "async_ok")


class TestForbiddenConstructs:
    """Direct use of dangerous names must be rejected."""

    def test_exec_call(self) -> None:
        src = 'def execute(**kw):\n    exec("print(1)")\n    return {}\n'
        with pytest.raises(SkillValidationError, match="exec"):
            validate_skill_source(src, "x")

    def test_eval_call(self) -> None:
        src = 'def execute(**kw):\n    eval("1+1")\n    return {}\n'
        with pytest.raises(SkillValidationError, match="eval"):
            validate_skill_source(src, "x")

    def test_compile_call(self) -> None:
        src = (
            'def execute(**kw):\n'
            '    c = compile("1", "x", "eval")\n'
            '    return {"c": c}\n'
        )
        with pytest.raises(SkillValidationError, match="compile"):
            validate_skill_source(src, "x")

    def test_dunder_import(self) -> None:
        src = (
            'def execute(**kw):\n'
            '    m = __import__("os")\n'
            '    return {"m": m}\n'
        )
        with pytest.raises(SkillValidationError, match="__import__"):
            validate_skill_source(src, "x")

    def test_getattr_call(self) -> None:
        src = (
            'def execute(**kw):\n'
            '    x = getattr(object(), "foo", None)\n'
            '    return {"x": x}\n'
        )
        with pytest.raises(SkillValidationError, match="getattr"):
            validate_skill_source(src, "x")

    def test_direct_open(self) -> None:
        src = 'def execute(**kw):\n    f = open("/etc/passwd")\n    return {}\n'
        with pytest.raises(SkillValidationError, match="open"):
            validate_skill_source(src, "x")

    def test_builtins_access(self) -> None:
        src = (
            'def execute(**kw):\n'
            '    return {"b": __builtins__}\n'
        )
        with pytest.raises(SkillValidationError, match="__builtins__"):
            validate_skill_source(src, "x")

    def test_class_traversal(self) -> None:
        src = (
            'def execute(**kw):\n'
            '    subs = object.__subclasses__()\n'
            '    return {"n": len(subs)}\n'
        )
        with pytest.raises(SkillValidationError, match="__subclasses__"):
            validate_skill_source(src, "x")


class TestImportRestrictions:
    def test_forbidden_module_import(self) -> None:
        src = "import subprocess\ndef execute(**kw): return {}\n"
        with pytest.raises(SkillValidationError, match="subprocess"):
            validate_skill_source(src, "x")

    def test_forbidden_module_import_from(self) -> None:
        src = "from os import system\ndef execute(**kw): return {}\n"
        with pytest.raises(SkillValidationError, match="os"):
            validate_skill_source(src, "x")

    def test_socket_import_allowed_but_runtime_blocked(self) -> None:
        # Importing socket is allowed (e.g. for getaddrinfo use in
        # SSRF-safe skills); actual socket creation is blocked at
        # runtime by the subprocess sandbox unless safety.network=true.
        src = "import socket\ndef execute(**kw): return {}\n"
        validate_skill_source(src, "x")

    def test_ctypes_forbidden(self) -> None:
        src = "import ctypes\ndef execute(**kw): return {}\n"
        with pytest.raises(SkillValidationError, match="ctypes"):
            validate_skill_source(src, "x")

    def test_pickle_forbidden(self) -> None:
        src = "import pickle\ndef execute(**kw): return {}\n"
        with pytest.raises(SkillValidationError, match="pickle"):
            validate_skill_source(src, "x")

    def test_star_import_rejected(self) -> None:
        src = "from json import *\ndef execute(**kw): return {}\n"
        with pytest.raises(SkillValidationError, match="[Ss]tar"):
            validate_skill_source(src, "x")

    def test_relative_import_rejected(self) -> None:
        src = "from . import helpers\ndef execute(**kw): return {}\n"
        with pytest.raises(SkillValidationError, match="[Rr]elative"):
            validate_skill_source(src, "x")

    def test_submodule_allowed_if_parent_allowed(self) -> None:
        src = (
            "from urllib.parse import urlparse\n"
            "def execute(**kw):\n"
            "    return {'u': urlparse('http://x').netloc}\n"
        )
        validate_skill_source(src, "x")

    def test_allowed_imports_do_not_trigger(self) -> None:
        src = (
            "import json\n"
            "import re\n"
            "import math\n"
            "from datetime import datetime, timedelta\n"
            "from typing import Any\n"
            "from collections.abc import Iterable\n"
            "import httpx\n"
            "def execute(**kw):\n"
            "    return {}\n"
        )
        validate_skill_source(src, "ok")


class TestTopLevelRestrictions:
    def test_toplevel_call_rejected(self) -> None:
        src = (
            'import json\n'
            'print("hi")\n'
            'def execute(**kw): return {}\n'
        )
        with pytest.raises(SkillValidationError, match="module level"):
            validate_skill_source(src, "x")

    def test_toplevel_os_system_rejected_via_toplevel_rule(self) -> None:
        """Even if os were allowed, a top-level call fires the module-level rule."""
        src = (
            "import json\n"
            'json.loads("{}")\n'  # json is allowed, but calls at top level are not
            "def execute(**kw): return {}\n"
        )
        with pytest.raises(SkillValidationError, match="module level"):
            validate_skill_source(src, "x")

    def test_toplevel_constant_assign_allowed(self) -> None:
        src = (
            "VERSION = '1.0'\n"
            "CONSTANTS = {'a': 1, 'b': [1, 2, 3]}\n"
            "def execute(**kw): return {'v': VERSION}\n"
        )
        validate_skill_source(src, "ok")

    def test_toplevel_nonconstant_assign_rejected(self) -> None:
        src = (
            "import json\n"
            'X = json.loads("{}")\n'
            "def execute(**kw): return {}\n"
        )
        with pytest.raises(SkillValidationError, match="module level"):
            validate_skill_source(src, "x")


class TestBypassAttempts:
    """Regression tests for previously-documented bypasses of the regex check."""

    def test_string_concatenation_for_import(self) -> None:
        """Before: `__import__('sub'+'process')` slipped past regex."""
        src = (
            "def execute(**kw):\n"
            "    m = __import__('sub' + 'process')\n"
            "    return {'m': m}\n"
        )
        with pytest.raises(SkillValidationError, match="__import__"):
            validate_skill_source(src, "x")

    def test_getattr_via_builtins(self) -> None:
        src = (
            "def execute(**kw):\n"
            "    imp = getattr(__builtins__, '__imp' + 'ort__')\n"
            "    return {'imp': imp}\n"
        )
        with pytest.raises(SkillValidationError):
            validate_skill_source(src, "x")

    def test_whitespace_eval(self) -> None:
        """The old regex `\\beval\\s*\\(` missed `eval  (...)` with extra spaces."""
        src = 'def execute(**kw):\n    return eval ("1+1")\n'
        with pytest.raises(SkillValidationError, match="eval"):
            validate_skill_source(src, "x")

    def test_module_load_time_attack_blocked(self) -> None:
        """Malicious code at import time must be rejected even if execute is OK."""
        src = (
            "import json\n"
            'DATA = json.loads("{}")\n'  # innocent-looking but top-level call
            "def execute(**kw):\n"
            "    return {}\n"
        )
        with pytest.raises(SkillValidationError):
            validate_skill_source(src, "x")
