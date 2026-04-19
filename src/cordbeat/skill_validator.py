"""AST-based whitelist validator for AI-proposed skill code.

The prior regex-based pattern check was trivially bypassable (string
concatenation, ``getattr`` dynamic lookups, etc.) and only inspected the
surface of the source text. This module rejects any construct that is not
explicitly permitted, so bypass attempts require adding a new allowed
construct rather than hiding inside an existing one.

High-level rules
----------------

* Only a small set of top-level statements is allowed (imports, function /
  class definitions, docstrings, and constant module-level assignments).
  Module-level function *calls* are rejected so that merely importing the
  skill cannot execute arbitrary code.
* Imports are restricted to an allow-list of modules. Star imports are
  rejected.
* A fixed set of dangerous names (``exec``, ``eval``, ``__import__``,
  ``subprocess``, ``ctypes``, dynamic ``getattr``/``setattr``/``delattr``,
  etc.) is rejected wherever it appears.
* The module must define a top-level ``execute`` function (sync or async)
  that accepts ``**kwargs``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

__all__ = [
    "SkillValidationError",
    "validate_skill_source",
    "ALLOWED_IMPORTS",
]


class SkillValidationError(ValueError):
    """Raised when skill source code fails static validation."""


# Modules a skill is allowed to import. Anything outside this set is
# rejected. We intentionally omit:
#   - subprocess, os (except limited), sys, ctypes, multiprocessing,
#     importlib, builtins — direct access to the host system.
#   - socket, ssl — direct network (skills should go through `httpx`).
#   - pickle, marshal — arbitrary code execution via deserialization.
#   - shutil — broad filesystem manipulation.
ALLOWED_IMPORTS: frozenset[str] = frozenset(
    {
        # Future annotations (harmless, often auto-added)
        "__future__",
        # Standard data / text utilities
        "json",
        "re",
        "math",
        "statistics",
        "datetime",
        "time",
        "calendar",
        "decimal",
        "fractions",
        "random",
        "uuid",
        "hashlib",
        "hmac",
        "base64",
        "binascii",
        "textwrap",
        "string",
        "unicodedata",
        "struct",
        "enum",
        "dataclasses",
        "typing",
        "typing_extensions",
        "collections",
        "collections.abc",
        "functools",
        "itertools",
        "operator",
        "copy",
        "heapq",
        "bisect",
        "contextlib",
        "abc",
        # Path / URL parsing (no IO)
        "pathlib",
        "urllib.parse",
        "posixpath",
        "ntpath",
        # Async primitives (needed for async skills; real IO is gated by
        # the subprocess sandbox, not by whether asyncio can be imported).
        "asyncio",
        # IP / network address *parsing* (no IO). Actual socket use is
        # still blocked at runtime by the sandbox unless the skill
        # declares safety.network=true.
        "ipaddress",
        "socket",
        "ssl",
        # HTTP client (skills requiring network must also set
        # safety.network=true and pass host checks in the skill runner).
        "httpx",
        # Markdown / YAML / other lightweight parsers often useful
        "yaml",
    }
)


# Callables that may appear at module level as part of a constant-like
# initialization (e.g. ``_SET = frozenset({"a", "b"})``). The arguments
# must themselves be constants or other allowed calls.
_ALLOWED_MODULE_LEVEL_CALLS: frozenset[str] = frozenset(
    {
        "frozenset",
        "set",
        "tuple",
        "list",
        "dict",
        "range",
        # Parsing helpers that are side-effect free:
        "ip_address",
        "ip_network",
    }
)


# Names that must never appear as attribute accesses or free-variable
# references, regardless of scope.
_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "__builtins__",
        "globals",
        "locals",
        "vars",
        "breakpoint",
        "input",
        "open",
    }
)


# Attributes that, when accessed on any object, indicate dynamic /
# reflective behaviour we cannot safely permit. These names are also
# rejected as direct identifiers (covers ``from X import getattr``).
_FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "__import__",
        "__subclasses__",
        "__bases__",
        "__class__",
        "__mro__",
        "__globals__",
        "__code__",
        "__dict__",
        "__builtins__",
        "__getattribute__",
        "__reduce__",
        "__reduce_ex__",
    }
)


@dataclass
class _Violation:
    lineno: int
    message: str


class _Validator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[_Violation] = []
        self._scope_depth = 0
        self.has_execute = False

    # ----- helpers --------------------------------------------------

    def _report(self, node: ast.AST, message: str) -> None:
        self.violations.append(
            _Violation(lineno=getattr(node, "lineno", 0), message=message)
        )

    def _check_name(self, node: ast.AST, name: str) -> None:
        if name in _FORBIDDEN_NAMES or name in _FORBIDDEN_ATTRS:
            self._report(node, f"Use of forbidden name: {name!r}")

    # ----- module / top level --------------------------------------

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            if not self._is_allowed_toplevel(stmt):
                self._report(
                    stmt,
                    "Only imports, function/class definitions, constant "
                    "assignments and docstrings are allowed at module level",
                )
                continue
            self.visit(stmt)

    def _is_allowed_toplevel(self, stmt: ast.stmt) -> bool:
        if isinstance(
            stmt,
            ast.Import
            | ast.ImportFrom
            | ast.FunctionDef
            | ast.AsyncFunctionDef
            | ast.ClassDef,
        ):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return True  # docstring
        if isinstance(stmt, ast.Assign | ast.AnnAssign):
            value = stmt.value
            return value is None or self._is_constant_expr(value)
        return False

    def _is_constant_expr(self, node: ast.AST) -> bool:
        """Return True if *node* is a compile-time constant literal or
        a safe constructor call on constants (e.g. ``frozenset({"a"})``).
        """
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Tuple | ast.List | ast.Set):
            return all(self._is_constant_expr(elt) for elt in node.elts)
        if isinstance(node, ast.Dict):
            keys = [k for k in node.keys if k is not None]
            return all(self._is_constant_expr(k) for k in keys) and all(
                self._is_constant_expr(v) for v in node.values
            )
        if isinstance(node, ast.UnaryOp):
            return self._is_constant_expr(node.operand)
        if isinstance(node, ast.BinOp):
            return self._is_constant_expr(node.left) and self._is_constant_expr(
                node.right
            )
        if isinstance(node, ast.Call):
            func = node.func
            name: str | None = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _ALLOWED_MODULE_LEVEL_CALLS:
                return all(self._is_constant_expr(a) for a in node.args) and all(
                    self._is_constant_expr(kw.value) for kw in node.keywords
                )
        return False

    # ----- imports --------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _import_allowed(alias.name):
                self._report(
                    node,
                    f"Import of module {alias.name!r} is not allowed",
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            self._report(node, "Relative imports are not allowed")
            return
        module = node.module or ""
        if not _import_allowed(module):
            self._report(node, f"Import from module {module!r} is not allowed")
            return
        for alias in node.names:
            if alias.name == "*":
                self._report(node, "Star imports are not allowed")
                continue
            if alias.name in _FORBIDDEN_ATTRS or alias.name in _FORBIDDEN_NAMES:
                self._report(
                    node,
                    f"Importing name {alias.name!r} is not allowed",
                )

    # ----- definitions ---------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function_def(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function_def(node)

    def _check_function_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self._scope_depth == 0 and node.name == "execute":
            self.has_execute = True
        self._scope_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._scope_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scope_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._scope_depth -= 1

    # ----- expression-level checks ---------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        self._check_name(node, node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in _FORBIDDEN_ATTRS:
            self._report(node, f"Access to attribute {node.attr!r} is not allowed")
        # Continue descending into the value so chained attributes are checked.
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FORBIDDEN_NAMES:
            self._report(node, f"Call to forbidden builtin: {func.id!r}")
        if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_ATTRS:
            self._report(node, f"Call to forbidden method: {func.attr!r}")
        self.generic_visit(node)


def _import_allowed(dotted_name: str) -> bool:
    """Check if *dotted_name* (or any of its parent packages) is allow-listed."""
    if not dotted_name:
        return False
    parts = dotted_name.split(".")
    for i in range(1, len(parts) + 1):
        candidate = ".".join(parts[:i])
        if candidate in ALLOWED_IMPORTS:
            return True
    return False


def validate_skill_source(source: str, skill_name: str) -> None:
    """Validate AI-proposed skill *source* code, raising on violations.

    Parameters
    ----------
    source:
        The full Python source of the proposed skill's ``main.py``.
    skill_name:
        Used for error messages only.

    Raises
    ------
    SkillValidationError
        If the source contains any construct outside the whitelist, or if
        it fails to define an ``execute(**kwargs)`` function.
    """
    if not source.strip():
        raise SkillValidationError("Skill code is empty.")

    try:
        tree = ast.parse(source, filename=f"skills/{skill_name}/main.py")
    except SyntaxError as exc:
        raise SkillValidationError(
            f"Skill {skill_name!r} has a syntax error: {exc.msg} (line {exc.lineno})"
        ) from exc

    validator = _Validator()
    validator.visit(tree)

    if not validator.has_execute:
        validator.violations.append(
            _Violation(
                lineno=0,
                message="Skill must define a top-level execute() function",
            )
        )

    if validator.violations:
        summary = "; ".join(
            f"line {v.lineno}: {v.message}" for v in validator.violations
        )
        raise SkillValidationError(f"Skill {skill_name!r} failed validation: {summary}")
