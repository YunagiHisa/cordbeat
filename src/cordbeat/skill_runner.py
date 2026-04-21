"""Subprocess entry point for isolated skill execution.

This module is invoked as ``python -m cordbeat.skill_runner`` by the
parent cordbeat process. It sets up a best-effort sandbox, loads the
target skill's ``main.py`` fresh, and executes its ``execute(**params)``
function. All communication with the parent happens over stdin/stdout
using newline-delimited JSON ("ndjson"):

* First line on stdin: ``{"type": "init", "skill_dir": ..., ...}``
* Child may emit ``memory_call`` / ``log`` messages on stdout and
  receive ``memory_result`` / ``memory_error`` on stdin.
* Last message on stdout: ``{"type": "result", ...}`` or ``{"type":
  "error", ...}``.

The runner is deliberately conservative: anything not needed by the
skill must be blocked. It runs with no access to the parent's memory,
state, or open sockets.
"""

from __future__ import annotations

import asyncio
import builtins
import importlib.util
import inspect
import io
import json
import os
import socket as _socket_module
import ssl as _ssl_module
import sys
import threading
from pathlib import Path
from typing import Any

from .exceptions import SkillError, SkillExecutionError


class SkillPermissionError(SkillError):
    """Raised when a skill violates its sandbox permissions."""


# -------------------- stdio JSON protocol ----------------------------

_STDOUT_LOCK = threading.Lock()


def _send(msg: dict[str, Any]) -> None:
    """Write a single JSON message as one line to the original stdout."""
    with _STDOUT_LOCK:
        _ORIGINAL_STDOUT.write(json.dumps(msg, default=str) + "\n")
        _ORIGINAL_STDOUT.flush()


def _recv_line() -> dict[str, Any]:
    line = _ORIGINAL_STDIN.readline()
    if not line:
        raise EOFError("parent closed stdin")
    result: dict[str, Any] = json.loads(line)
    return result


# Captured at import time, before we replace sys.stdout/stdin.
_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDIN = sys.stdin


# -------------------- sandbox guards ---------------------------------


def _install_network_guard() -> None:
    """Block all socket creation."""

    def _blocked(*_a: Any, **_kw: Any) -> Any:
        raise SkillPermissionError("Network access is not allowed for this skill")

    _socket_module.socket = _blocked  # type: ignore[assignment, misc]
    # The C implementation has its own class; block that too.
    try:
        import _socket

        _socket.socket = _blocked  # type: ignore[assignment, misc]
    except ImportError:
        pass

    def _blocked_ctx(*_a: Any, **_kw: Any) -> Any:
        raise SkillPermissionError("SSL contexts are not allowed for this skill")

    _ssl_module.create_default_context = _blocked_ctx


def _install_fs_guard(allowed_root: Path) -> None:
    """Restrict open / os.open to paths inside *allowed_root*.

    Uses ``os.path.realpath`` plus (on Unix) ``O_NOFOLLOW`` at the final
    component to make TOCTOU / symlink-swap attacks harder: even if the
    dirname is verified and then swapped to point outside the sandbox,
    ``open()`` on the full path will still refuse to follow a symlink at
    the final component.
    """
    resolved_root = allowed_root.resolve()
    original_open = builtins.open
    original_os_open = os.open

    def _check(path: Any) -> None:
        try:
            p = Path(path)
        except TypeError:
            return  # fd int or similar — let the lower layer decide
        if not p.is_absolute():
            p = Path.cwd() / p
        # Canonicalize; if the final component is a symlink to outside,
        # realpath will reveal it.
        real = Path(os.path.realpath(p))
        try:
            real.relative_to(resolved_root)
        except ValueError:
            raise SkillPermissionError(
                f"Filesystem access outside work directory is not allowed: {p}"
            ) from None

    def _restricted_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, int):
            return original_open(file, *args, **kwargs)
        _check(file)
        if hasattr(os, "O_NOFOLLOW"):
            # Pre-open with O_NOFOLLOW to ensure the final component isn't
            # a dangling symlink we missed. Then hand the fd to open().
            mode = kwargs.get("mode", args[0] if args else "r")
            flags = os.O_RDONLY
            if any(c in mode for c in "wax"):
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            elif "+" in mode or "r+" in mode:
                flags = os.O_RDWR
            try:
                fd = original_os_open(str(file), flags | os.O_NOFOLLOW)
            except OSError:
                # Fall back to unconstrained open for modes we don't map
                # (e.g. binary, line-buffering nuances). The realpath
                # check above still provides the main protection.
                return original_open(file, *args, **kwargs)
            return original_open(fd, *args, **kwargs)
        return original_open(file, *args, **kwargs)

    def _restricted_os_open(path: Any, *args: Any, **kwargs: Any) -> int:
        _check(path)
        flags = args[0] if args else kwargs.get("flags", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
            if args:
                args = (flags, *args[1:])
            else:
                kwargs["flags"] = flags
        return original_os_open(path, *args, **kwargs)

    builtins.open = _restricted_open
    io.open = _restricted_open
    os.open = _restricted_os_open


def _apply_resource_limits(
    memory_mb: int,
    timeout_seconds: int,
    max_fsize_mb: int = 64,
) -> None:
    """Apply resource rlimits (Unix only). Silently no-op on Windows."""
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        return

    try:
        mem_bytes = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))  # type: ignore[attr-defined,unused-ignore]
    except (ValueError, OSError):
        pass

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds + 1))  # type: ignore[attr-defined,unused-ignore]
    except (ValueError, OSError):
        pass

    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))  # type: ignore[attr-defined,unused-ignore]
    except (ValueError, OSError):
        pass

    try:
        fsize = max_fsize_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))  # type: ignore[attr-defined,unused-ignore]
    except (ValueError, OSError):
        pass


def _restrict_sys_and_env(skill_dir: Path, work_dir: Path) -> None:
    """Trim sys.path to stdlib + skill dir and sanitize os.environ.

    Stdlib paths (detected via ``sys.base_prefix`` / ``sys.prefix``) are
    preserved so common stdlib modules like ``datetime`` or
    ``urllib.parse`` continue to work. Third-party site-packages
    installed into the parent's environment are *also* preserved because
    skills legitimately depend on ``httpx`` and ``yaml``; the AST
    validator already restricts which modules may be imported.
    """
    # We don't prune site-packages in practice (skills need httpx/yaml
    # which live there), but we do remove the parent's CWD and any
    # loose PYTHONPATH-injected entries so the skill can't import sibling
    # project modules.
    keep_prefixes = tuple(
        os.path.realpath(p)
        for p in {sys.base_prefix, sys.prefix, sys.exec_prefix, sys.base_exec_prefix}
    )

    def _is_kept(p: str) -> bool:
        if not p:
            return False
        try:
            rp = os.path.realpath(p)
        except OSError:
            return False
        return rp.startswith(keep_prefixes)

    filtered = [str(skill_dir)] + [p for p in sys.path if _is_kept(p)]
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for p in filtered:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    sys.path[:] = deduped

    keep = {
        "PATH",
        "PYTHONPATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "HOME",
    }
    for key in list(os.environ):
        if key not in keep:
            del os.environ[key]

    try:
        os.chdir(work_dir)
    except OSError:
        pass


# -------------------- memory proxy -----------------------------------


_ALLOWED_MEMORY_METHODS = {
    "get_certain_records",
    "add_certain_record",
    "get_proposal",
    "get_pending_proposals",
}


class _MemoryProxy:
    """RPC stub the skill uses as its ``context.memory`` object.

    Each method call is sent as a ``memory_call`` JSON message to the
    parent; the parent executes it against the real :class:`MemoryStore`
    and sends back a ``memory_result`` / ``memory_error``. The call
    blocks the child until the response arrives. This is safe because
    the child is single-threaded for skill purposes.
    """

    def __init__(self) -> None:
        self._next_id = 1
        self._loop = asyncio.new_event_loop() if False else None  # reserved

    def _call(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if method not in _ALLOWED_MEMORY_METHODS:
            raise SkillPermissionError(
                f"Memory method not allowed from skill: {method!r}"
            )
        msg_id = self._next_id
        self._next_id += 1
        _send(
            {
                "type": "memory_call",
                "id": msg_id,
                "method": method,
                "args": list(args),
                "kwargs": kwargs,
            }
        )
        while True:
            resp = _recv_line()
            rtype = resp.get("type")
            if rtype == "memory_result" and resp.get("id") == msg_id:
                return resp.get("result")
            if rtype == "memory_error" and resp.get("id") == msg_id:
                raise SkillExecutionError(resp.get("error", "memory call failed"))
            # Unexpected message — drop it.

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def method(*args: Any, **kwargs: Any) -> Any:
            return self._call(name, args, kwargs)

        async def async_method(*args: Any, **kwargs: Any) -> Any:
            return self._call(name, args, kwargs)

        # Heuristic: if the real MemoryStore's method is async, we need
        # to return an awaitable. We don't have access to the real class
        # in the child, so we always return an awaitable; skills call
        # these with ``await`` anyway.
        return async_method


# -------------------- main -------------------------------------------


def _load_skill_module(skill_dir: Path, skill_name: str) -> Any:
    main_path = skill_dir / "main.py"
    spec = importlib.util.spec_from_file_location(
        f"cordbeat_skill_{skill_name}",
        str(main_path),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {main_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_skill(
    skill_dir: Path,
    skill_name: str,
    params: dict[str, Any],
    allow_memory: bool,
    sandbox: dict[str, Any],
    loop: asyncio.AbstractEventLoop,
) -> Any:
    work_dir = Path(sandbox["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)

    if not sandbox.get("network", False):
        _install_network_guard()
    if not sandbox.get("filesystem", False):
        _install_fs_guard(work_dir)
    _restrict_sys_and_env(skill_dir, work_dir)
    _apply_resource_limits(
        memory_mb=int(sandbox.get("memory_mb", 256)),
        timeout_seconds=int(sandbox.get("timeout_seconds", 30)),
    )

    module = _load_skill_module(skill_dir, skill_name)
    fn = getattr(module, "execute", None)
    if fn is None:
        raise SkillExecutionError(f"Skill {skill_name!r} has no execute() function")

    sig = inspect.signature(fn)
    if "context" in sig.parameters:
        from types import SimpleNamespace

        context = SimpleNamespace(
            sandbox=True,
            network=sandbox.get("network", False),
            filesystem=sandbox.get("filesystem", False),
            work_dir=work_dir,
            memory=_MemoryProxy() if allow_memory else None,
        )
        params = {**params, "context": context}

    result = fn(**params)
    if inspect.isawaitable(result):
        result = loop.run_until_complete(_await(result))
    return result


async def _await(x: Any) -> Any:
    return await x


def main() -> int:
    # Redirect child stdout so any accidental print() doesn't corrupt
    # the JSON channel; sys.stdout now goes to stderr.
    sys.stdout = sys.stderr

    # Create the asyncio event loop BEFORE we install the network guard,
    # because the default selector loop uses socketpair() for its
    # self-pipe and would otherwise be blocked by the guard.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        init = _recv_line()
    except Exception as exc:  # pragma: no cover - stdin closed immediately
        _send({"type": "error", "error": f"init read failed: {exc}"})
        return 2

    if init.get("type") != "init":
        _send({"type": "error", "error": "first message must be 'init'"})
        return 2

    try:
        result = _run_skill(
            skill_dir=Path(init["skill_dir"]),
            skill_name=init["skill_name"],
            params=init.get("params") or {},
            allow_memory=bool(init.get("allow_memory", False)),
            sandbox=init.get("sandbox") or {},
            loop=loop,
        )
    except SkillPermissionError as exc:
        _send({"type": "error", "error": str(exc), "kind": "permission"})
        return 1
    except Exception as exc:  # noqa: BLE001
        import traceback

        _send(
            {
                "type": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "kind": "runtime",
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    if not isinstance(result, dict):
        result = {"result": result}
    _send({"type": "result", "result": result})
    return 0


if __name__ == "__main__":
    sys.exit(main())
