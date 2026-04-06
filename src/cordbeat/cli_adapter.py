"""CLI adapter — interactive terminal client for CordBeat."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime

import websockets


async def main(ws_url: str = "ws://localhost:8765") -> None:
    print(f"Connecting to CordBeat Core at {ws_url}...")
    async with websockets.connect(ws_url) as ws:
        # Handshake: identify as CLI adapter
        await ws.send(json.dumps({"adapter_id": "cli"}))
        ack = json.loads(await ws.recv())
        print(f"Connected: {ack.get('content', 'OK')}")
        print("Type a message and press Enter. Ctrl+C to quit.\n")

        # Listen for incoming messages in background
        async def listener() -> None:
            try:
                async for raw in ws:
                    data = json.loads(raw)
                    msg_type = data.get("type", "")
                    content = data.get("content", "")
                    if msg_type == "error":
                        print(f"\n[error] {content}")
                    else:
                        print(f"\n[{msg_type}] {content}")
                    print("> ", end="", flush=True)
            except websockets.ConnectionClosed:
                print("\nDisconnected from Core.")

        listen_task = asyncio.create_task(listener())

        # Send loop
        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(
                    None,
                    lambda: input("> "),
                )
                if not line.strip():
                    continue
                msg = {
                    "type": "message",
                    "adapter_id": "cli",
                    "platform_user_id": "cli_user",
                    "content": line.strip(),
                    "timestamp": datetime.now().isoformat(),
                }
                await ws.send(json.dumps(msg))
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
        finally:
            listen_task.cancel()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:8765"
    asyncio.run(main(url))
