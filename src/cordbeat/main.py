"""CordBeat main entry point — boots all subsystems."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from cordbeat.ai_backend import create_backend
from cordbeat.config import load_config
from cordbeat.engine import CoreEngine
from cordbeat.gateway import GatewayServer, MessageQueue
from cordbeat.heartbeat import HeartbeatLoop
from cordbeat.memory import MemoryStore
from cordbeat.skills import SkillRegistry
from cordbeat.soul import Soul

logger = logging.getLogger("cordbeat")


async def main(config_path: str = "config.yaml") -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config(config_path)
    logger.info("Configuration loaded from %s", config_path)

    # ── Initialize subsystems ─────────────────────────────────────
    soul = Soul(
        config.soul.soul_dir,
        emotion_decay_rate=config.soul.emotion_decay_rate,
        emotion_baseline_intensity=config.soul.emotion_baseline_intensity,
        emotion_secondary_clear_threshold=config.soul.emotion_secondary_clear_threshold,
    )
    logger.info("SOUL loaded: %s", soul.name)

    memory = MemoryStore(config.memory)
    await memory.initialize()
    logger.info("MEMORY store initialized")

    ai = create_backend(config.ai_backend)
    logger.info(
        "AI backend: %s (%s)",
        config.ai_backend.provider,
        config.ai_backend.model,
    )

    skills = SkillRegistry(config.skills_dir)
    skills.load_all()
    logger.info("SKILL registry: %d skills loaded", len(skills.available_skills))

    queue = MessageQueue()
    gateway = GatewayServer(config.gateway, queue)

    engine = CoreEngine(
        ai=ai,
        soul=soul,
        memory=memory,
        skills=skills,
        gateway=gateway,
        memory_config=config.memory,
    )
    queue.set_handler(engine.handle_message)

    heartbeat = HeartbeatLoop(
        config=config.heartbeat,
        ai=ai,
        soul=soul,
        memory=memory,
        skills=skills,
        gateway=gateway,
        queue=queue,
        memory_config=config.memory,
    )

    # ── Start services ────────────────────────────────────────────
    await gateway.start()
    await heartbeat.start()

    queue_task = asyncio.create_task(queue.process_loop())

    logger.info(
        "CordBeat is alive — %s is ready (ws://%s:%d)",
        soul.name,
        config.gateway.host,
        config.gateway.port,
    )

    # ── Graceful shutdown ─────────────────────────────────────────
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler for SIGTERM
            pass

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        pass

    logger.info("Shutting down...")
    await heartbeat.stop()
    await gateway.stop()
    queue_task.cancel()
    try:
        await queue_task
    except asyncio.CancelledError:
        pass
    await ai.aclose()
    await memory.close()
    logger.info("CordBeat stopped")


def cli() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    asyncio.run(main(config_path))


if __name__ == "__main__":
    cli()
