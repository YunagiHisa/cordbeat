"""Parent-side orchestration for subprocess-isolated skill execution.

Spawns ``python -m cordbeat.skill_runner`` as a child process, ships the
skill parameters via JSON-over-stdio, honors memory proxy RPCs, and
returns the result. Enforces wall-clock timeout and stdout size limits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import SkillError

logger = logging.getLogger(__name__)


class SkillPermissionError(SkillError):
    """Raised when a sandboxed skill violates its permissions."""


class SkillSandboxError(SkillError):
    """Raised when the sandbox itself fails (timeout, crash, protocol error)."""


@dataclass
class SandboxConfig:
    timeout_seconds: int = 30
    memory_mb: int = 256
    max_stdout_bytes: int = 1 * 1024 * 1024  # 1 MiB


DEFAULT_CONFIG = SandboxConfig()


async def _kill_tree(pid: int) -> None:
    """Kill a process and its descendants, preferring psutil when available."""
    try:
        import psutil  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError:
        try:
            os.kill(pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
        return

    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, ProcessLookupError):
        return
    try:
        children = proc.children(recursive=True)
    except psutil.Error:
        children = []
    for child in children:
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        proc.kill()
    except psutil.Error:
        pass


async def _pump_stderr(proc: asyncio.subprocess.Process, skill_name: str) -> None:
    assert proc.stderr is not None
    while True:
        line = await proc.stderr.readline()
        if not line:
            return
        try:
            text = line.decode("utf-8", errors="replace").rstrip()
        except Exception:  # noqa: BLE001
            continue
        if text:
            logger.debug("[skill:%s] %s", skill_name, text)


async def run_skill_in_subprocess(
    *,
    skill_dir: Path,
    skill_name: str,
    params: dict[str, Any],
    sandbox: dict[str, Any],
    memory: Any | None = None,
    config: SandboxConfig | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Run a skill in an isolated subprocess and return its result.

    Parameters mirror the subprocess init message. ``memory`` is an
    optional object; if provided, the child may request whitelisted
    method calls which are executed here and have their return values
    sent back. ``python_executable`` lets callers spawn the child with
    a skill-private Python (e.g. a ``uv``-managed env from
    :class:`cordbeat.skill_env.SkillEnvManager`); when ``None`` the
    host interpreter (``sys.executable``) is used.
    """
    cfg = config or DEFAULT_CONFIG

    init_msg = {
        "type": "init",
        "skill_dir": str(skill_dir),
        "skill_name": skill_name,
        "params": params,
        "allow_memory": memory is not None,
        "sandbox": {
            "network": bool(sandbox.get("network", False)),
            "filesystem": bool(sandbox.get("filesystem", False)),
            "work_dir": sandbox["work_dir"],
            "memory_mb": cfg.memory_mb,
            "timeout_seconds": cfg.timeout_seconds,
        },
    }

    py_exe = python_executable or sys.executable
    runner_path = Path(__file__).parent / "skill_runner.py"
    proc = await asyncio.create_subprocess_exec(
        py_exe,
        "-I",  # isolated mode: ignore PYTHONPATH/user site
        str(runner_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONNOUSERSITE": "1",
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        },
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    stderr_task = asyncio.create_task(_pump_stderr(proc, skill_name))

    try:
        proc.stdin.write((json.dumps(init_msg) + "\n").encode("utf-8"))
        await proc.stdin.drain()

        result = await asyncio.wait_for(
            _read_loop(proc, memory, cfg),
            timeout=cfg.timeout_seconds,
        )
    except TimeoutError as exc:
        await _kill_tree(proc.pid)
        raise SkillSandboxError(
            f"Skill '{skill_name}' timed out after {cfg.timeout_seconds}s"
        ) from exc
    except BaseException:
        await _kill_tree(proc.pid)
        raise
    finally:
        stderr_task.cancel()
        try:
            await stderr_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except TimeoutError:
            await _kill_tree(proc.pid)

    return result


async def _read_loop(
    proc: asyncio.subprocess.Process,
    memory: Any | None,
    cfg: SandboxConfig,
) -> dict[str, Any]:
    assert proc.stdout is not None
    assert proc.stdin is not None

    bytes_read = 0
    while True:
        line = await proc.stdout.readline()
        if not line:
            code = await proc.wait()
            raise SkillSandboxError(
                f"Skill subprocess exited without result (code={code})"
            )
        bytes_read += len(line)
        if bytes_read > cfg.max_stdout_bytes:
            raise SkillSandboxError("Skill exceeded stdout size limit")

        try:
            msg = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise SkillSandboxError(f"Invalid JSON from skill: {exc}") from exc

        mtype = msg.get("type")
        if mtype == "result":
            return msg.get("result") or {}
        if mtype == "error":
            kind = msg.get("kind", "runtime")
            err = msg.get("error", "unknown error")
            if kind == "permission":
                raise SkillPermissionError(err)
            tb = msg.get("traceback")
            if tb:
                logger.debug("Skill traceback:\n%s", tb)
            raise SkillSandboxError(err)
        if mtype == "memory_call":
            await _handle_memory_call(proc, memory, msg)
            continue
        # Unknown — ignore.


_ALLOWED_MEMORY_METHODS = {
    "get_certain_records",
    "add_certain_record",
    "get_proposal",
    "get_pending_proposals",
}


async def _handle_memory_call(
    proc: asyncio.subprocess.Process,
    memory: Any | None,
    msg: dict[str, Any],
) -> None:
    assert proc.stdin is not None
    msg_id = msg.get("id")
    method = msg.get("method", "")
    args = msg.get("args") or []
    kwargs = msg.get("kwargs") or {}

    async def _send(out: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(out, default=str) + "\n").encode("utf-8"))
        await proc.stdin.drain()

    if memory is None:
        await _send(
            {"type": "memory_error", "id": msg_id, "error": "memory not available"}
        )
        return

    if method not in _ALLOWED_MEMORY_METHODS:
        await _send(
            {
                "type": "memory_error",
                "id": msg_id,
                "error": f"Method not allowed: {method}",
            }
        )
        return

    fn = getattr(memory, method, None)
    if fn is None:
        await _send(
            {
                "type": "memory_error",
                "id": msg_id,
                "error": f"No such memory method: {method}",
            }
        )
        return

    try:
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
    except Exception as exc:  # noqa: BLE001
        await _send({"type": "memory_error", "id": msg_id, "error": str(exc)})
        return

    # Normalize dataclasses / non-JSON types
    await _send({"type": "memory_result", "id": msg_id, "result": _jsonable(result)})


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    if hasattr(obj, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    if isinstance(obj, (list | tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return str(obj)
