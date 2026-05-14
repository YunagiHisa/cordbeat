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
        cwd_config = Path("config.yaml")
        if cwd_config.is_file():
            # Both exist — warn so users aren't confused about which is active
            import warnings

            warnings.warn(
                f"Two config files found: {home_config} (active) and {cwd_config} "
                f"(ignored). CordBeat always uses ~/.cordbeat/config.yaml. "
                f"You can safely delete {cwd_config} to avoid confusion.",
                stacklevel=1,
            )
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
    _shutdown_started = False

    def _signal_handler() -> None:
        nonlocal _shutdown_started
        if _shutdown_started:
            # Second Ctrl+C during graceful shutdown → force-quit immediately.
            # This matches the standard Unix convention where a second SIGINT
            # means "I really want out now".
            logger.warning("Forced exit (second interrupt during shutdown)")
            os._exit(130)  # 130 = 128 + SIGINT, conventional exit code
        logger.info("Shutdown signal received")
        _shutdown_started = True
        stop_event.set()

    loop = asyncio.get_running_loop()

    # Suppress RecursionError from asyncio internal callbacks (Python 3.12 bug:
    # Task.cancel() recursion in deep task trees printed as "Exception in callback").
    _orig_exc_handler = loop.get_exception_handler()

    def _loop_exc_handler(
        lp: asyncio.AbstractEventLoop, context: dict[str, Any]
    ) -> None:
        exc = context.get("exception")
        if isinstance(exc, RecursionError):
            logger.warning(
                "RecursionError in asyncio callback suppressed "
                "(Python 3.12 task-cancel depth bug)"
            )
            return
        if _orig_exc_handler is not None:
            _orig_exc_handler(lp, context)
        else:
            lp.default_exception_handler(context)

    loop.set_exception_handler(_loop_exc_handler)

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
        except RecursionError:
            # Python 3.12 bug: deep asyncio task graphs can overflow the
            # call stack during cancel() propagation.  Log and continue.
            logger.warning("RecursionError during shutdown of %s (ignored)", label)
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
        try:
            await asyncio.gather(*pending, return_exceptions=True)
        except RecursionError:
            pass  # deep task graphs on Python 3.12 — safe to ignore here

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
  cordbeat                     Start server + interactive CLI chat (default)
  cordbeat server              Start the CordBeat server only (headless)
  cordbeat setup               Run the interactive setup wizard
  cordbeat doctor              Check system health
  cordbeat config              Show current configuration summary
  cordbeat config edit         Open config.yaml in $EDITOR
  cordbeat model               Switch AI provider / model
  cordbeat service install     Register CordBeat as a system service (auto-start)
  cordbeat service start       Start the registered service
  cordbeat service stop        Stop the running service
  cordbeat service status      Show service status
  cordbeat service uninstall   Remove the service registration
  cordbeat update              Update CordBeat to the latest version
  cordbeat uninstall           Remove CordBeat service + data directory

Separate commands:
  cordbeat-cli                 Connect CLI chat to a running CordBeat server
  cordbeat-discord             Start the Discord adapter (connects to running server)
  cordbeat-telegram            Start the Telegram adapter
"""


def _cmd_model() -> None:
    """Interactive provider/model switcher — updates config.yaml."""
    config_path = _resolve_config_path()

    try:
        import yaml  # noqa: PLC0415

        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Failed to load config: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    ai = data.get("ai_backend", {})
    current_provider = ai.get("provider", "ollama")
    current_model = ai.get("model", "")
    print(f"Current: provider={current_provider}  model={current_model}")
    print()

    providers = [
        ("ollama", "Ollama (local, default)"),
        ("openai_compat", "OpenAI-compat (LM Studio, llama.cpp, vLLM, etc.)"),
        ("anthropic", "Anthropic Claude (cloud)"),
        ("openrouter", "OpenRouter (200+ models via cloud)"),
    ]
    print("Select provider:")
    for i, (key, label) in enumerate(providers, 1):
        marker = " *" if key == current_provider else ""
        print(f"  {i}. {label}{marker}")
    choice = input("\nEnter number (Enter = keep current): ").strip()

    if choice:
        try:
            idx = int(choice) - 1
            if not 0 <= idx < len(providers):
                raise ValueError
        except ValueError:
            print("Invalid choice.", file=sys.stderr)
            raise SystemExit(1)
        new_provider = providers[idx][0]
    else:
        new_provider = current_provider

    # Suggest model based on provider
    new_model = ""
    if new_provider == "ollama":
        # Try to list models from the running Ollama API
        base_url = ai.get("base_url", "http://localhost:11434")
        print(f"\nFetching models from {base_url}…")
        try:
            import urllib.request  # noqa: PLC0415

            with urllib.request.urlopen(
                f"{base_url}/api/tags", timeout=5
            ) as resp:
                import json as _json  # noqa: PLC0415

                tags = _json.loads(resp.read())
            models_list: list[str] = [
                m["name"] for m in tags.get("models", [])
            ]
            if models_list:
                print("\nAvailable Ollama models:")
                for j, m in enumerate(models_list, 1):
                    marker = " *" if m == current_model else ""
                    print(f"  {j}. {m}{marker}")
                model_choice = input(
                    "\nEnter number or model name (Enter = keep current): "
                ).strip()
                if model_choice:
                    try:
                        mi = int(model_choice) - 1
                        if 0 <= mi < len(models_list):
                            new_model = models_list[mi]
                        else:
                            new_model = model_choice
                    except ValueError:
                        new_model = model_choice
            else:
                new_model = input(
                    f"Model name (Enter = keep '{current_model}'): "
                ).strip()
        except Exception:
            new_model = input(
                f"Model name (Enter = keep '{current_model}'): "
            ).strip()
    elif new_provider == "anthropic":
        defaults = [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-20241022",
        ]
        print("\nCommon Anthropic models:")
        for j, m in enumerate(defaults, 1):
            print(f"  {j}. {m}")
        model_input = input(
            f"\nModel (Enter = keep '{current_model or defaults[0]}'): "
        ).strip()
        if model_input:
            try:
                mi = int(model_input) - 1
                new_model = defaults[mi] if 0 <= mi < len(defaults) else model_input
            except ValueError:
                new_model = model_input
        else:
            new_model = current_model or defaults[0]
    elif new_provider == "openrouter":
        new_model = input(
            f"Model slug (e.g. mistralai/mistral-7b-instruct, "
            f"Enter = keep '{current_model}'): "
        ).strip()
    else:
        new_model = input(
            f"Model name (Enter = keep '{current_model}'): "
        ).strip()

    if not new_model:
        new_model = current_model

    # Apply and persist
    if "ai_backend" not in data:
        data["ai_backend"] = {}
    data["ai_backend"]["provider"] = new_provider
    data["ai_backend"]["model"] = new_model

    # Prompt for API key when switching to a cloud provider
    if new_provider in ("anthropic", "openrouter", "openai_compat"):
        options = data["ai_backend"].get("options", {})
        existing_key = options.get("api_key", "")
        prompt_label = (
            "Anthropic API key"
            if new_provider == "anthropic"
            else "OpenRouter API key"
            if new_provider == "openrouter"
            else "API key"
        )
        key_input = input(
            f"{prompt_label} (Enter = keep existing): "
        ).strip()
        if key_input:
            options["api_key"] = key_input
            data["ai_backend"]["options"] = options
        elif not existing_key and new_provider != "openai_compat":
            print(
                "⚠️  No API key set. Add it later via `cordbeat config edit` "
                "(ai_backend.options.api_key)."
            )

    try:
        with open(config_path, "w") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        print(
            f"\n✅ Config updated: provider={new_provider}  model={new_model}"
        )
        print(f"   Config file: {config_path}")
    except Exception as exc:
        print(f"Failed to write config: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


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


def _cmd_uninstall() -> None:
    """Remove the CordBeat service registration and optionally wipe data."""
    from cordbeat.tools.service import run_service_command

    print("CordBeat Uninstall")
    print("=" * 40)

    # Step 1: remove the service (ignore errors — may not be installed)
    print("\n[1/2] Removing service registration…")
    run_service_command("uninstall")

    # Step 2: optionally delete ~/.cordbeat data directory
    home_dir = Path.home() / ".cordbeat"
    if home_dir.exists():
        print(f"\n[2/2] Data directory: {home_dir}")
        print("  This contains memories, config, and logs.")
        answer = input("  Delete data directory? [y/N] ").strip().lower()
        if answer == "y":
            import shutil as _shutil

            _shutil.rmtree(home_dir, ignore_errors=True)
            print(f"  🗑  Deleted {home_dir}")
        else:
            print("  ↳  Kept (you can delete it manually later).")
    else:
        print(f"\n[2/2] Data directory {home_dir} not found — skipping.")

    print("\n✅ CordBeat uninstalled.")
    print("   To fully remove the package: uv tool uninstall cordbeat")


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

    if sub == "server":
        # Headless server-only mode (no interactive CLI).
        # Strip "server" from argv so _resolve_config_path sees the right arg.
        sys.argv = [sys.argv[0]] + args[1:]
        config_path = _resolve_config_path()
        _watchdog = threading.Timer(30.0, lambda: os._exit(1))
        _watchdog.daemon = True
        try:
            asyncio.run(main(config_path))
        except KeyboardInterrupt:
            _watchdog.start()
        finally:
            _watchdog.cancel()
        return

    if sub in ("setup", "init"):
        from cordbeat.tools.wizard import cordbeat_init_cli

        cordbeat_init_cli()
        return

    if sub == "config":
        _cmd_config(args[1:])
        return

    if sub == "model":
        _cmd_model()
        return

    if sub == "service":
        action = args[1] if len(args) > 1 else "status"
        from cordbeat.tools.service import run_service_command

        raise SystemExit(run_service_command(action))

    if sub == "update":
        _cmd_update()
        return

    if sub == "uninstall":
        _cmd_uninstall()
        return

    if sub.startswith("-"):
        print(f"Unknown option: {sub}\n{_HELP}", file=sys.stderr)
        raise SystemExit(1)

    # No recognised subcommand → default: server + interactive CLI chat
    cli_chat()


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
    ws_url = f"ws://{cfg.gateway.host}:{cfg.gateway.port}"

    if not _probe_ws(ws_url):
        print("[CordBeat] Starting server in background...")
        _start_server_background(config_path, ws_url)

    cli_main()


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
        # Fallback: invoke via the current interpreter
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
