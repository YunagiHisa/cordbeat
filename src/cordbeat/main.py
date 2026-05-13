"""CordBeat main entry point — boots all subsystems."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from cordbeat.agent.heartbeat import HeartbeatLoop
from cordbeat.agent.soul import Soul
from cordbeat.ai.backend import create_backend
from cordbeat.config import cordbeat_home, load_config
from cordbeat.core.engine import CoreEngine
from cordbeat.core.gateway import GatewayServer, MessageQueue
from cordbeat.memory import MemoryStore
from cordbeat.skills import SandboxConfig, SkillRateLimiter, SkillRegistry
from cordbeat.tools.metrics import REGISTRY as METRICS_REGISTRY
from cordbeat.tools.metrics_server import PrometheusServer

logger = logging.getLogger("cordbeat")

# Windows: use ProactorEventLoop (default since Python 3.8) to support
# asyncio.create_subprocess_exec (required by skill uv-env builds).
# SIGINT (Ctrl+C) is handled by the signal handlers below instead.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


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
    from cordbeat.tools.wizard import run_wizard

    return str(run_wizard())


def _sync_builtin_skills(skills_dir: Path) -> None:
    """Copy missing built-in skills to the configured skills directory.

    Skills that already exist in the target are left untouched so
    that user modifications are preserved.  New built-in skills added
    in future releases are automatically deployed on the next startup.
    """
    # Locate the bundled skills/ directory (project-root sibling of src/cordbeat/)
    _here = Path(__file__).resolve()  # …/src/cordbeat/main.py
    bundled = _here.parent.parent.parent / "skills"
    if not bundled.is_dir():
        return

    skills_dir.mkdir(parents=True, exist_ok=True)
    for src_skill in bundled.iterdir():
        if not src_skill.is_dir():
            continue
        dst_skill = skills_dir / src_skill.name
        if dst_skill.exists():
            continue  # user copy already present — don't overwrite
        try:
            shutil.copytree(src_skill, dst_skill)
            logger.info("Installed built-in skill '%s' → %s", src_skill.name, dst_skill)
        except OSError:
            logger.exception("Failed to install built-in skill '%s'", src_skill.name)


async def main(
    config_path: str = "config.yaml",
    _ready: asyncio.Event | None = None,
) -> None:
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

    _sync_builtin_skills(Path(config.skills_dir))
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
        timezone_name=config.heartbeat.timezone,
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

    if _ready is not None:
        _ready.set()  # Signal to main_with_cli that the gateway is up

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
        except (NotImplementedError, OSError):
            # Windows: add_signal_handler is not supported for any signal.
            # Fall back to the synchronous signal module; call_soon_threadsafe
            # ensures the callback runs safely inside the running event loop.
            try:
                signal.signal(
                    sig,
                    lambda *_: loop.call_soon_threadsafe(_signal_handler),
                )
            except (ValueError, OSError):
                pass

    async def _interrupt_monitor() -> None:
        """Belt-and-suspenders: catches KeyboardInterrupt raised by asyncio on
        Windows when the process is sent SIGINT while the event loop is busy."""
        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.2)
        except (KeyboardInterrupt, asyncio.CancelledError):
            stop_event.set()

    monitor_task = asyncio.create_task(_interrupt_monitor())

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass

    logger.info("Shutting down...")

    # Use individual timeouts so one stuck subsystem can't hang the entire
    # shutdown sequence.
    async def _cancel_with_timeout(
        coro_or_task: Any, label: str, timeout: float = 10.0
    ) -> None:
        try:
            await asyncio.wait_for(coro_or_task, timeout=timeout)
        except TimeoutError:
            logger.warning("Shutdown timeout for %s (%.0fs)", label, timeout)
        except asyncio.CancelledError:
            pass  # expected when task was explicitly cancelled
        except Exception:
            logger.exception("Error during shutdown of %s", label)

    await _cancel_with_timeout(heartbeat.stop(), "heartbeat")
    await _cancel_with_timeout(gateway.stop(), "gateway")
    queue_task.cancel()
    await _cancel_with_timeout(queue_task, "queue_task")
    if metrics_server is not None:
        await _cancel_with_timeout(metrics_server.stop(), "metrics_server")
    await ai.aclose()
    await memory.close()

    # Cancel any tasks that outlived explicit shutdown to unblock asyncio.run()
    # cleanup (e.g. loop.shutdown_default_executor).  ChromaDB, aiosqlite, and
    # httpx may leave background tasks that would otherwise hang the process.
    current = asyncio.current_task()
    pending = {t for t in asyncio.all_tasks() if t is not current}
    if pending:
        logger.debug("Cancelling %d lingering task(s) before exit", len(pending))
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    logger.info("CordBeat stopped")


async def main_with_cli(config_path: str) -> None:
    """Start the CordBeat server and connect the interactive CLI adapter.

    The server runs as a background task; the CLI adapter runs in the
    foreground.  When the user quits the CLI (Ctrl+C / EOF), the server
    is shut down gracefully.
    """
    from cordbeat.adapters.cli import main as cli_main

    # Load config early so we know the WS URL & auth token.
    # (load_config deduplicates secret warnings, so the second call inside
    # main() will not re-emit them.)
    _cfg = load_config(config_path)
    ws_url = f"ws://{_cfg.gateway.host}:{_cfg.gateway.port}"
    auth_token = _cfg.gateway.auth_token

    # Use an Event so main() signals us when the gateway is ready, avoiding
    # the old socket-probe approach that caused a spurious WS handshake error.
    _ready = asyncio.Event()
    server_task = asyncio.create_task(main(config_path, _ready=_ready))

    try:
        await asyncio.wait_for(_ready.wait(), timeout=10.0)
    except TimeoutError:
        if server_task.done() and not server_task.cancelled():
            exc = server_task.exception()
            raise RuntimeError(f"CordBeat server failed to start: {exc}") from exc
        server_task.cancel()
        raise RuntimeError(
            "CordBeat server did not start within 10 s. "
            "Check logs for startup errors (e.g. Ollama not running, "
            "missing config, or port already in use)."
        )

    try:
        await cli_main(ws_url, auth_token)
    finally:
        server_task.cancel()
        try:
            await server_task
        except (asyncio.CancelledError, Exception):
            pass


_HELP = """\
CordBeat — local-first autonomous AI agent

Usage:
  cordbeat                     Start the CordBeat server
  cordbeat chat                Start server + interactive CLI chat
  cordbeat setup               Run the interactive setup wizard
  cordbeat doctor              Check system health
  cordbeat config              Show current configuration summary
  cordbeat config edit         Open config.yaml in $EDITOR
  cordbeat service install     Register CordBeat as a system service (auto-start)
  cordbeat service start       Start the registered service
  cordbeat service stop        Stop the running service
  cordbeat service status      Show service status
  cordbeat service uninstall   Remove the service registration
  cordbeat update              Update CordBeat to the latest version
"""


def _cmd_config(args: list[str]) -> None:
    config_path = _resolve_config_path()
    if args and args[0] == "edit":
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")
        if not editor:
            editor = "notepad" if sys.platform == "win32" else "nano"
        os.execvp(editor, [editor, config_path])
        return  # unreachable after exec
    print(f"Config: {config_path}")
    try:
        import yaml

        with open(config_path) as f:
            data = yaml.safe_load(f)
        ai_cfg = data.get("ai_backend", {})
        gw_cfg = data.get("gateway", {})
        print(f"  Backend : {ai_cfg.get('backend', '?')}")
        print(f"  Model   : {ai_cfg.get('model', '?')}")
        print(
            f"  Gateway : {gw_cfg.get('host', '127.0.0.1')}:{gw_cfg.get('port', 8765)}"
        )
    except Exception:
        pass


def _cmd_update() -> None:
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if Path(uv).exists() or shutil.which("uv"):
        print("Updating CordBeat via uv…")
        raise SystemExit(
            subprocess.run([uv, "tool", "upgrade", "cordbeat"]).returncode
        )
    print("Updating CordBeat via pip…")
    raise SystemExit(
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "cordbeat"]
        ).returncode
    )


def cli() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(_HELP)
        return

    sub = args[0]

    if sub == "doctor":
        from cordbeat.tools.doctor import run_doctor

        raise SystemExit(run_doctor())

    if sub in ("chat", "--chat"):
        sys.argv = [sys.argv[0]] + args[1:]
        cli_chat()
        return

    if sub in ("setup", "init"):
        from cordbeat.tools.wizard import cordbeat_init_cli

        cordbeat_init_cli()
        return

    if sub == "config":
        _cmd_config(args[1:])
        return

    if sub == "service":
        action = args[1] if len(args) > 1 else "status"
        from cordbeat.tools.service import run_service_command

        raise SystemExit(run_service_command(action))

    if sub == "update":
        _cmd_update()
        return

    if sub.startswith("-"):
        print(f"Unknown option: {sub}\n{_HELP}", file=sys.stderr)
        raise SystemExit(1)

    # No recognised subcommand → start server
    config_path = _resolve_config_path()
    # Safety watchdog: force-exit if the process hangs more than 30 s after
    # Ctrl+C.  Background threads from ChromaDB / httpx can prevent Python
    # from exiting even after asyncio cleanup completes.
    _watchdog = threading.Timer(30.0, lambda: os._exit(1))
    _watchdog.daemon = True
    try:
        asyncio.run(main(config_path))
    except KeyboardInterrupt:
        _watchdog.start()
    finally:
        _watchdog.cancel()


def cli_chat() -> None:
    """Start CordBeat in combined server + interactive CLI chat mode."""
    config_path = _resolve_config_path()
    _watchdog = threading.Timer(30.0, lambda: os._exit(1))
    _watchdog.daemon = True
    try:
        asyncio.run(main_with_cli(config_path))
    except KeyboardInterrupt:
        _watchdog.start()
    finally:
        _watchdog.cancel()


if __name__ == "__main__":
    cli()
