"""shell_exec skill — execute shell commands (dangerous, disabled by default)."""

from __future__ import annotations

import asyncio
from typing import Any


async def execute(
    *,
    command: str,
    timeout: int = 30,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Run a shell command and return stdout/stderr."""
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        return {"error": f"Command timed out after {timeout}s", "returncode": -1}

    return {
        "returncode": process.returncode,
        "stdout": stdout.decode("utf-8", errors="replace")[:4096],
        "stderr": stderr.decode("utf-8", errors="replace")[:1024],
    }
