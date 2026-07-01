"""CLI adapter — interactive terminal client for CordBeat."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any, cast

import websockets

_CORE_SLASH_COMMANDS = (
    "/approve",
    "/reject",
    "/proposals",
    "/link",
    "/link-confirm",
    "/unlink",
    "/name",
    "/quiet",
    "/prefer",
    "/draw",
)


def _build_cli_completion_tree(pending_proposal_ids: Iterable[str]) -> dict[str, Any]:
    proposal_tree = {proposal_id: None for proposal_id in pending_proposal_ids}
    return {
        "/approve": dict(proposal_tree),
        "/reject": dict(proposal_tree),
        **{
            command: None
            for command in _CORE_SLASH_COMMANDS
            if command
            not in {
                "/approve",
                "/reject",
            }
        },
    }


def _mark_cli_proposal_action_sent(
    line: str, pending_proposals: dict[str, dict[str, Any]]
) -> None:
    parts = line.split(maxsplit=1)
    if len(parts) == 2 and parts[0].lower() in {"/approve", "/reject"}:
        pending_proposals.pop(parts[1].strip(), None)


async def main(ws_url: str = "ws://localhost:8765", auth_token: str = "") -> None:
    print(f"Connecting to CordBeat Core at {ws_url}...")
    async with websockets.connect(ws_url) as ws:
        # Handshake: identify as CLI adapter
        handshake: dict[str, str] = {"adapter_id": "cli"}
        if auth_token:
            handshake["auth_token"] = auth_token
        await ws.send(json.dumps(handshake))
        ack = json.loads(await ws.recv())
        print(f"Connected: {ack.get('content', 'OK')}")
        print("Type a message and press Enter. Ctrl+C to quit.\n")

        _waiting = False
        _reply_event = asyncio.Event()
        pending_proposals: dict[str, dict[str, Any]] = {}

        prompt_session: Any = None
        nested_completer: Any = None
        if sys.stdin.isatty():
            try:
                from prompt_toolkit import PromptSession
                from prompt_toolkit.completion import NestedCompleter

                prompt_session = PromptSession()
                nested_completer = NestedCompleter
            except Exception:
                prompt_session = None
                nested_completer = None

        # Listen for incoming messages in background
        async def listener() -> None:
            nonlocal _waiting
            try:
                async for raw in ws:
                    data = json.loads(raw)
                    msg_type = data.get("type", "")
                    content = data.get("content", "")
                    images: list[str] = data.get("images") or []
                    _waiting = False
                    _reply_event.set()
                    if msg_type == "error":
                        print(f"\n[error] {content}")
                    elif msg_type == "skill_confirm":
                        meta = data.get("metadata") or {}
                        skill_name = meta.get("skill_name", "unknown")
                        skill_params = meta.get("skill_params") or {}
                        proposal_id = meta.get("proposal_id", "")
                        if proposal_id:
                            pending_proposals[str(proposal_id)] = {
                                "skill_name": skill_name,
                                "skill_params": skill_params,
                            }
                        print("\n┌─────────────────────────────────────────────────┐")
                        print(f"│  🔧  Skill Execution Required: {skill_name:<17}│")
                        print("├─────────────────────────────────────────────────┤")
                        if skill_params:
                            for k, v in skill_params.items():
                                line_ = f"│  • {k}: {v}"
                                print(f"{line_:<51}│")
                        print("├─────────────────────────────────────────────────┤")
                        print(f"│  /approve {proposal_id[:32]}... (once)         │")
                        print(f"│  /reject  {proposal_id[:32]}... (deny)         │")
                        print("└─────────────────────────────────────────────────┘")
                    elif msg_type in ("message", "ack"):
                        print(f"\nBot: {content}")
                    else:
                        print(f"\n[{msg_type}] {content}")
                    if images:
                        import base64 as _b64  # noqa: PLC0415

                        for idx, b64 in enumerate(images):
                            try:
                                size = len(_b64.b64decode(b64))
                                print(f"  [🖼️ Image {idx + 1}: PNG, {size:,} bytes]")
                            except Exception:
                                print(f"  [🖼️ Image {idx + 1}: (decode error)]")
                    print("> ", end="", flush=True)
            except websockets.ConnectionClosed:
                print("\nDisconnected from Core.")
            finally:
                _reply_event.set()  # always unblock send loop on exit

        listen_task = asyncio.create_task(listener())

        # Send loop
        loop = asyncio.get_event_loop()

        async def read_line() -> str:
            if prompt_session is not None and nested_completer is not None:
                prompt_session.completer = nested_completer.from_nested_dict(
                    _build_cli_completion_tree(pending_proposals.keys())
                )
                return cast(str, await prompt_session.prompt_async("> "))
            return await loop.run_in_executor(
                None,
                lambda: input("> "),
            )

        try:
            while True:
                line = (await read_line()).strip()
                if not line:
                    continue
                _mark_cli_proposal_action_sent(line, pending_proposals)
                msg = {
                    "type": "message",
                    "adapter_id": "cli",
                    "platform_user_id": "cli_user",
                    "content": line,
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                }
                _reply_event.clear()
                await ws.send(json.dumps(msg))
                _waiting = True
                print("  (thinking...) ", end="", flush=True)
                # Wait for reply before next input; skip if listener already exited
                if not listen_task.done():
                    await _reply_event.wait()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
        finally:
            listen_task.cancel()


if __name__ == "__main__":
    from cordbeat.config import cordbeat_home, load_config

    _config = load_config(cordbeat_home() / "config.yaml")
    _ws_url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else f"ws://{_config.gateway.host}:{_config.gateway.port}"
    )
    asyncio.run(main(_ws_url, auth_token=_config.gateway.auth_token))


def cli_main(config_path: str | None = None) -> None:
    """Entry point: connect CLI to a running CordBeat server (no server started).

    *config_path* may be provided by callers that have already resolved the
    config (e.g. cli_chat()).  When omitted, it falls back to sys.argv[1] or
    ``~/.cordbeat/config.yaml`` as before.
    """
    from cordbeat.config import cordbeat_home, load_config

    if config_path is None:
        config_path = (
            sys.argv[1]
            if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
            else str(cordbeat_home() / "config.yaml")
        )
    _config = load_config(config_path)
    ws_url = f"ws://{_config.gateway.host}:{_config.gateway.port}"
    try:
        asyncio.run(main(ws_url, auth_token=_config.gateway.auth_token))
    except KeyboardInterrupt:
        pass
