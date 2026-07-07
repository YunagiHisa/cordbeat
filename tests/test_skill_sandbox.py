"""Unit tests for the parent-side skill sandbox helpers.

The full subprocess round-trip is exercised in ``test_skills.py``; this
module pins down the parent-side protocol handling in isolation — the
memory-RPC allow-list, the stdout/size and protocol-error branches of the
read loop, and the JSON normalization helper. These are trust-boundary
paths whose *rejection* behaviour the end-to-end tests never reach.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from cordbeat.skills.sandbox import (
    DEFAULT_CONFIG,
    SandboxConfig,
    SkillPermissionError,
    SkillSandboxError,
    _handle_memory_call,
    _jsonable,
    _kill_tree,
    _pump_stderr,
    _read_loop,
)


class _FakeStdin:
    """Captures bytes written by the sandbox and decodes them as JSON lines."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def messages(self) -> list[dict[str, Any]]:
        return [json.loads(chunk.decode("utf-8")) for chunk in self.writes]


class _FakeStdout:
    """Yields pre-seeded lines from ``readline``; empty bytes ends the stream."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeProc:
    def __init__(
        self,
        stdout_lines: list[bytes] | None = None,
        code: int = 0,
        stderr_lines: list[bytes] | None = None,
    ) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(stdout_lines or [])
        self.stderr = _FakeStdout(stderr_lines or [])
        self._code = code
        self.pid = 4321

    async def wait(self) -> int:
        return self._code


def _line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload) + "\n").encode("utf-8")


# --------------------------- _jsonable -------------------------------


def test_jsonable_passes_through_native_types() -> None:
    assert _jsonable({"a": 1, "b": [1, 2]}) == {"a": 1, "b": [1, 2]}


def test_jsonable_serializes_object_via_dict() -> None:
    class Record:
        def __init__(self) -> None:
            self.name = "x"
            self._private = "hidden"

    out = _jsonable(Record())
    assert out == {"name": "x"}  # underscore-prefixed attrs are dropped


def test_jsonable_handles_nested_containers_and_fallback() -> None:
    class Opaque:
        __slots__ = ()  # no __dict__, not natively JSON-able

        def __repr__(self) -> str:
            return "OPAQUE"

    out = _jsonable({1: (Opaque(),)})
    assert out == {"1": ["OPAQUE"]}  # int key -> str, tuple -> list, repr fallback


def test_jsonable_handles_cycles() -> None:
    value: dict[str, Any] = {"name": "loop"}
    value["self"] = value

    out = _jsonable(value)

    assert out == {"name": "loop", "self": "<cycle>"}


# ----------------------- _handle_memory_call -------------------------


async def test_memory_call_rejected_when_memory_unavailable() -> None:
    proc = _FakeProc()
    await _handle_memory_call(proc, None, {"id": 1, "method": "get_certain_records"})
    (msg,) = proc.stdin.messages()
    assert msg == {"type": "memory_error", "id": 1, "error": "memory not available"}


async def test_memory_call_rejects_method_outside_allowlist() -> None:
    proc = _FakeProc()
    memory = object()  # would have no such method anyway
    await _handle_memory_call(proc, memory, {"id": 2, "method": "drop_all_tables"})
    (msg,) = proc.stdin.messages()
    assert msg["type"] == "memory_error"
    assert "Method not allowed" in msg["error"]


async def test_memory_call_reports_missing_allowed_method() -> None:
    class PartialMemory:
        pass  # allow-listed name, but attribute absent

    proc = _FakeProc()
    await _handle_memory_call(
        proc, PartialMemory(), {"id": 3, "method": "get_certain_records"}
    )
    (msg,) = proc.stdin.messages()
    assert "No such memory method" in msg["error"]


async def test_memory_call_awaits_async_method_and_normalizes_result() -> None:
    class AsyncMemory:
        async def get_certain_records(self, user: str) -> list[str]:
            assert user == "u1"
            return ["fact"]

    proc = _FakeProc()
    await _handle_memory_call(
        proc,
        AsyncMemory(),
        {"id": 4, "method": "get_certain_records", "args": ["u1"]},
    )
    (msg,) = proc.stdin.messages()
    assert msg == {"type": "memory_result", "id": 4, "result": ["fact"]}


async def test_memory_call_surfaces_method_exception() -> None:
    class BoomMemory:
        def get_proposal(self, *_a: Any, **_k: Any) -> Any:
            raise RuntimeError("kaboom")

    proc = _FakeProc()
    await _handle_memory_call(proc, BoomMemory(), {"id": 5, "method": "get_proposal"})
    (msg,) = proc.stdin.messages()
    assert msg["type"] == "memory_error"
    assert "kaboom" in msg["error"]


# --------------------------- _read_loop ------------------------------


async def test_read_loop_returns_result_message() -> None:
    proc = _FakeProc([_line({"type": "result", "result": {"ok": True}})])
    out = await _read_loop(proc, None, DEFAULT_CONFIG)
    assert out == {"ok": True}


async def test_read_loop_dispatches_memory_call_then_result() -> None:
    class CountMemory:
        def get_pending_proposals(self) -> int:
            return 7

    proc = _FakeProc(
        [
            _line({"type": "memory_call", "id": 9, "method": "get_pending_proposals"}),
            _line({"type": "result", "result": {"done": 1}}),
        ]
    )
    out = await _read_loop(proc, CountMemory(), DEFAULT_CONFIG)
    assert out == {"done": 1}
    # The parent answered the RPC before the result arrived.
    assert proc.stdin.messages()[0]["result"] == 7


async def test_read_loop_raises_permission_error() -> None:
    proc = _FakeProc(
        [_line({"type": "error", "kind": "permission", "error": "no net"})]
    )
    with pytest.raises(SkillPermissionError, match="no net"):
        await _read_loop(proc, None, DEFAULT_CONFIG)


async def test_read_loop_raises_runtime_error_with_traceback() -> None:
    proc = _FakeProc(
        [_line({"type": "error", "error": "boom", "traceback": "Traceback..."})]
    )
    with pytest.raises(SkillSandboxError, match="boom"):
        await _read_loop(proc, None, DEFAULT_CONFIG)


async def test_read_loop_enforces_stdout_size_limit() -> None:
    cfg = SandboxConfig(max_stdout_bytes=16)
    proc = _FakeProc([_line({"type": "result", "result": {"x": "y" * 100}})])
    with pytest.raises(SkillSandboxError, match="stdout size limit"):
        await _read_loop(proc, None, cfg)


async def test_read_loop_wraps_line_overrun_as_sandbox_error() -> None:
    class _OverrunStdout:
        async def readline(self) -> bytes:
            raise ValueError("Separator is not found, and chunk exceed the limit")

    proc = _FakeProc()
    proc.stdout = _OverrunStdout()  # type: ignore[assignment]
    with pytest.raises(SkillSandboxError, match="stdout size limit"):
        await _read_loop(proc, None, DEFAULT_CONFIG)


async def test_read_loop_errors_when_subprocess_exits_without_result() -> None:
    proc = _FakeProc([], code=3)  # no lines -> EOF
    with pytest.raises(SkillSandboxError, match="exited without result"):
        await _read_loop(proc, None, DEFAULT_CONFIG)


async def test_read_loop_rejects_invalid_json() -> None:
    proc = _FakeProc([b"{not json}\n"])
    with pytest.raises(SkillSandboxError, match="Invalid JSON"):
        await _read_loop(proc, None, DEFAULT_CONFIG)


# --------------------------- _pump_stderr ----------------------------


async def test_pump_stderr_drains_until_eof() -> None:
    # Lines (including a blank one that is skipped) then EOF (b"").
    proc = _FakeProc(stderr_lines=[b"warming up\n", b"\n", b"done\n"])
    await _pump_stderr(proc, "demo")  # returns cleanly at EOF, logs at debug


# --------------------------- _kill_tree ------------------------------


async def test_kill_tree_kills_process_and_children(monkeypatch: Any) -> None:
    killed: list[str] = []

    class _FakeChild:
        def kill(self) -> None:
            killed.append("child")

    class _FakePsutilProc:
        def __init__(self, pid: int) -> None:
            assert pid == 4321

        def children(self, recursive: bool = False) -> list[_FakeChild]:
            assert recursive is True
            return [_FakeChild()]

        def kill(self) -> None:
            killed.append("parent")

    class _FakePsutil:
        Error = RuntimeError
        NoSuchProcess = LookupError
        Process = _FakePsutilProc

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil)
    await _kill_tree(4321)
    assert killed == ["child", "parent"]


async def test_kill_tree_falls_back_to_os_kill_without_psutil(
    monkeypatch: Any,
) -> None:
    # Make ``import psutil`` raise ImportError inside _kill_tree.
    monkeypatch.setitem(sys.modules, "psutil", None)
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        "cordbeat.skills.sandbox.os.kill",
        lambda pid, sig: calls.append((pid, sig)),
    )
    await _kill_tree(1234)
    assert calls == [(1234, 9)]
