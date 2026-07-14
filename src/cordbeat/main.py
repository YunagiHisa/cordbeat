"""CordBeat main entry point — boots all subsystems."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from cordbeat.agent.heartbeat import HeartbeatLoop
from cordbeat.agent.soul import Soul
from cordbeat.ai.backend import create_backend
from cordbeat.config import cordbeat_home, gateway_ws_url, load_config
from cordbeat.core.engine import CoreEngine
from cordbeat.core.gateway import GatewayServer, MessageQueue
from cordbeat.memory import MemoryStore
from cordbeat.skills import SandboxConfig, SkillRateLimiter, SkillRegistry
from cordbeat.skills.policy import apply_default_skill_settings
from cordbeat.tools.metrics import REGISTRY as METRICS_REGISTRY
from cordbeat.tools.metrics_server import PrometheusServer

logger = logging.getLogger("cordbeat")

# Windows: use ProactorEventLoop (default since Python 3.8) to support
# asyncio.create_subprocess_exec (required by skill uv-env builds).
# SIGINT (Ctrl+C) is handled by the signal handlers below instead.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def _sync_builtin_skill_contexts(src_skill: Path, dst_skill: Path) -> None:
    """Add missing built-in context defaults without overwriting user settings."""
    src_yaml = src_skill / "skill.yaml"
    dst_yaml = dst_skill / "skill.yaml"
    if not src_yaml.is_file() or not dst_yaml.is_file():
        return
    try:
        src_data = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}
        dst_data = yaml.safe_load(dst_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(src_data, dict) or not isinstance(dst_data, dict):
            return
        src_contexts = src_data.get("contexts") or {}
        dst_contexts = dst_data.get("contexts") or {}
        if not isinstance(src_contexts, dict) or not isinstance(dst_contexts, dict):
            return
        if "shared_voice" in dst_contexts:
            return
        shared_voice = src_contexts.get("shared_voice")
        if not isinstance(shared_voice, bool):
            return

        text = dst_yaml.read_text(encoding="utf-8")
        lines = text.splitlines()
        setting = f"  shared_voice: {str(shared_voice).lower()}"
        contexts_index = next(
            (index for index, line in enumerate(lines) if line == "contexts:"),
            None,
        )
        if contexts_index is not None:
            lines.insert(contexts_index + 1, setting)
        elif "contexts" in dst_data:
            logger.warning(
                "Cannot safely add shared_voice default to non-block contexts "
                "for skill '%s'",
                src_skill.name,
            )
            return
        else:
            safety_index = next(
                (index for index, line in enumerate(lines) if line == "safety:"),
                len(lines),
            )
            lines[safety_index:safety_index] = ["contexts:", setting, ""]
        dst_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(
            "Added built-in shared_voice context default to skill '%s'",
            src_skill.name,
        )
    except (OSError, yaml.YAMLError, AttributeError):
        logger.exception(
            "Failed to sync context metadata for skill '%s'",
            src_skill.name,
        )


def _sync_managed_builtin_skill(src_skill: Path, dst_skill: Path) -> bool:
    """Update a CordBeat-managed built-in while preserving user settings."""
    src_yaml = src_skill / "skill.yaml"
    dst_yaml = dst_skill / "skill.yaml"
    if not src_yaml.is_file() or not dst_yaml.is_file():
        return False

    try:
        src_data = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}
        dst_data = yaml.safe_load(dst_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(src_data, dict) or not isinstance(dst_data, dict):
            return False
        if dst_data.get("author") != "cordbeat":
            return False
        src_data = apply_default_skill_settings(src_data)

        if "enabled" in dst_data:
            src_data["enabled"] = dst_data["enabled"]
        for key in (
            "ownership",
            "mutable_by_ai",
            "requires_approval_to_modify",
        ):
            if key in dst_data:
                src_data[key] = dst_data[key]
        dst_contexts = dst_data.get("contexts")
        if isinstance(dst_contexts, dict) and "shared_voice" in dst_contexts:
            src_contexts = src_data.setdefault("contexts", {})
            if isinstance(src_contexts, dict):
                src_contexts["shared_voice"] = dst_contexts["shared_voice"]

        dst_yaml.write_text(
            yaml.safe_dump(src_data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        for src_file in src_skill.iterdir():
            if src_file.name == "skill.yaml" or not src_file.is_file():
                continue
            shutil.copy2(src_file, dst_skill / src_file.name)
        logger.warning(
            "Updated managed built-in skill '%s'; CordBeat-managed code files "
            "were overwritten while user settings were preserved",
            src_skill.name,
        )
        return True
    except (OSError, yaml.YAMLError, AttributeError):
        logger.exception("Failed to update managed built-in skill '%s'", src_skill.name)
        return False


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
    """Install and update managed skills in the configured skills directory.

    Existing CordBeat-managed skills are updated while explicit user settings
    are preserved. User-created and AI-created skills are left untouched.
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
            if not _sync_managed_builtin_skill(src_skill, dst_skill):
                _sync_builtin_skill_contexts(src_skill, dst_skill)
            continue
        try:
            shutil.copytree(src_skill, dst_skill)
            dst_yaml = dst_skill / "skill.yaml"
            if dst_yaml.is_file():
                raw = yaml.safe_load(dst_yaml.read_text(encoding="utf-8")) or {}
                if isinstance(raw, dict):
                    raw = apply_default_skill_settings(raw)
                    dst_yaml.write_text(
                        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
                        encoding="utf-8",
                    )
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
            work_dir=Path(config.data_dir) / "sandbox",
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
        react_config=config.react,
        soul_config=config.soul,
        vision_enabled=config.ai_backend.vision_enabled,
        timezone_name=config.heartbeat.timezone,
        adapters_options={name: dict(a.options) for name, a in config.adapters.items()},
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
        adapters_options={name: dict(a.options) for name, a in config.adapters.items()},
    )

    # ── Start services ────────────────────────────────────────────
    stop_event = asyncio.Event()

    await gateway.start()
    await heartbeat.start()

    queue_task = asyncio.create_task(queue.process_loop())

    def _watch_queue_task(task: asyncio.Task[None]) -> None:
        # The message loop must never exit on its own; if it does, the bot
        # silently stops responding ("presence" dies). Shut down loudly
        # instead of running as a zombie.
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.critical("Message queue loop crashed: %s", exc, exc_info=exc)
        else:
            logger.critical("Message queue loop exited unexpectedly")
        stop_event.set()

    queue_task.add_done_callback(_watch_queue_task)

    if _ready is not None:
        _ready.set()  # Signal to main_with_cli that the gateway is up

    logger.info(
        "CordBeat is alive — %s is ready (ws://%s:%d)",
        soul.name,
        config.gateway.host,
        config.gateway.port,
    )

    # ── Graceful shutdown ─────────────────────────────────────────
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
            # A task we cancelled on purpose re-raises CancelledError when
            # awaited — that is a clean shutdown, not a timeout.
            logger.debug("Shutdown of %s completed via cancellation", label)
        except Exception:
            logger.exception("Error during shutdown of %s", label)

    await _cancel_with_timeout(heartbeat.stop(), "heartbeat")
    await _cancel_with_timeout(gateway.stop(), "gateway")
    queue_task.cancel()
    await _cancel_with_timeout(queue_task, "queue_task")
    if metrics_server is not None:
        await _cancel_with_timeout(metrics_server.stop(), "metrics_server")
    await _cancel_with_timeout(ai.aclose(), "ai_backend")
    await _cancel_with_timeout(memory.close(), "memory")
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
    ws_url = gateway_ws_url(_cfg.gateway.host, _cfg.gateway.port)
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
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("CordBeat server task failed during CLI shutdown")


def cli() -> None:
    # Handle subcommands
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        from cordbeat.tools.doctor import run_doctor

        raise SystemExit(run_doctor())

    # ``cordbeat server [path]`` — explicit server invocation.  Strip the
    # subcommand so that ``_resolve_config_path`` sees only the (optional)
    # config path argument.  Without this, ``server`` itself gets used as
    # the config path, ``load_config('server')`` silently returns defaults
    # (provider=ollama) and the agent connects to the wrong backend.
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        sys.argv.pop(1)

    config_path = _resolve_config_path()
    try:
        asyncio.run(main(config_path))
    except KeyboardInterrupt:
        pass


def cli_chat() -> None:
    """Start an interactive CLI chat, auto-starting the server if needed.

    If a CordBeat server is already listening on the configured port, this
    attaches the CLI to it — quitting the CLI does NOT stop the server.
    If no server is found, one is launched automatically as a detached
    background process (also survives CLI exit).
    Use ``cordbeat service stop`` or send SIGTERM to shut the server down.
    """
    from cordbeat.adapters.cli import cli_main

    config_path = _resolve_config_path()
    cfg = load_config(config_path)
    ws_url = gateway_ws_url(cfg.gateway.host, cfg.gateway.port)

    if not _probe_ws(ws_url):
        print("[CordBeat] Starting server in background...")
        _start_server_background(config_path, ws_url)

    try:
        cli_main(config_path)
    except KeyboardInterrupt:
        pass


def _probe_ws(url: str, timeout: float = 0.5) -> bool:
    """Return True if something is already listening at *url*."""
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8765
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _start_server_background(config_path: str, ws_url: str) -> None:
    """Spawn the CordBeat server as a fully detached background process.

    Blocks until the server's port is accepting connections (up to 15 s).
    """
    import time

    exe = shutil.which("cordbeat")
    if exe is not None:
        cmd: list[str] = [exe, "server", config_path]
    else:
        cmd = [sys.executable, __file__, "server", config_path]

    popen_kwargs: dict[str, Any] = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        popen_kwargs["start_new_session"] = True

    subprocess.Popen(cmd, **popen_kwargs)

    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if _probe_ws(ws_url, timeout=0.3):
            return
    raise RuntimeError(
        "CordBeat server did not become ready within 15 seconds.\n"
        "Run 'cordbeat server' in a separate terminal to see startup errors."
    )


if __name__ == "__main__":
    cli()
