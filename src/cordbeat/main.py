"""CordBeat main entry point — boots all subsystems."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

from cordbeat.ai_backend import create_backend
from cordbeat.config import cordbeat_home, load_config
from cordbeat.engine import CoreEngine
from cordbeat.gateway import GatewayServer, MessageQueue
from cordbeat.heartbeat import HeartbeatLoop
from cordbeat.memory import MemoryStore
from cordbeat.metrics import REGISTRY as METRICS_REGISTRY
from cordbeat.metrics_server import PrometheusServer
from cordbeat.skill_sandbox import SandboxConfig
from cordbeat.skills import SkillRateLimiter, SkillRegistry
from cordbeat.soul import Soul

logger = logging.getLogger("cordbeat")


def _resolve_config_path() -> str:
    """Find the config file, or run the setup wizard if none exists."""
    # Explicit argument overrides everything
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]

    # Try ~/.cordbeat/config.yaml
    home_config = cordbeat_home() / "config.yaml"
    if home_config.is_file():
        return str(home_config)

    # Try CWD config.yaml (backward compat)
    if Path("config.yaml").is_file():
        return "config.yaml"

    # Nothing found — run wizard
    from cordbeat.setup_wizard import run_wizard

    return str(run_wizard())


async def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)

    logging.basicConfig(
        level=getattr(logging, config.log.level.upper(), logging.INFO),
        format=config.log.format,
    )

    # Optional file logging with rotation
    if config.log.file:
        from logging.handlers import RotatingFileHandler

        log_path = Path(config.log.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=config.log.max_bytes,
            backupCount=config.log.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(config.log.format))
        logging.getLogger().addHandler(file_handler)

    # Suppress noisy websockets handshake errors (e.g. plain HTTP to WS)
    logging.getLogger("websockets.server").setLevel(logging.WARNING)

    logger.info("Configuration loaded from %s", config_path)

    METRICS_REGISTRY.set_enabled(config.metrics.enabled)
    metrics_server: PrometheusServer | None = None
    if config.metrics.enabled and config.metrics.prometheus_port > 0:
        metrics_server = PrometheusServer(
            host=config.metrics.prometheus_host,
            port=config.metrics.prometheus_port,
        )
        await metrics_server.start()
        logger.info(
            "Prometheus metrics endpoint listening on http://%s:%d/metrics",
            config.metrics.prometheus_host,
            config.metrics.prometheus_port,
        )

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

    skills = SkillRegistry(
        config.skills_dir,
        sandbox_config=SandboxConfig(
            timeout_seconds=int(config.skills.sandbox.timeout_seconds),
            memory_mb=config.skills.sandbox.memory_limit_mb,
            max_stdout_bytes=config.skills.sandbox.max_output_bytes,
        ),
        rate_limiter=SkillRateLimiter(
            default_per_minute=config.skills.default_rate_limit_per_minute,
        ),
    )
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
        vision_enabled=config.ai_backend.vision_enabled,
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
    if metrics_server is not None:
        await metrics_server.stop()
    await ai.aclose()
    await memory.close()
    logger.info("CordBeat stopped")


def cli() -> None:
    # Handle subcommands
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from cordbeat.doctor import run_doctor

        raise SystemExit(run_doctor())

    config_path = _resolve_config_path()
    asyncio.run(main(config_path))


if __name__ == "__main__":
    cli()
