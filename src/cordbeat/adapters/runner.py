"""Adapter runner — standalone entry points for platform adapters."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from cordbeat.config import AdapterConfig, cordbeat_home, load_config

logger = logging.getLogger("cordbeat")


def _resolve_adapter_config_path() -> str:
    """Resolve config path: explicit arg → ~/.cordbeat/config.yaml → CWD/config.yaml."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    home_config = cordbeat_home() / "config.yaml"
    if home_config.is_file():
        return str(home_config)
    cwd_config = Path("config.yaml")
    if cwd_config.is_file():
        return str(cwd_config)
    return str(cordbeat_home() / "config.yaml")


async def _run_adapter(adapter_name: str, config_path: str) -> None:
    config = load_config(config_path)

    logging.basicConfig(
        level=getattr(logging, config.log.level.upper(), logging.INFO),
        format=config.log.format,
    )

    # Optional file logging with rotation (per-adapter log file)
    if config.log.file:
        from logging.handlers import RotatingFileHandler

        core_log_path = Path(config.log.file)
        adapter_log_path = core_log_path.with_name(
            f"{core_log_path.stem}-{adapter_name}{core_log_path.suffix}"
        )
        adapter_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            adapter_log_path,
            maxBytes=config.log.max_bytes,
            backupCount=config.log.backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(config.log.format))
        logging.getLogger().addHandler(file_handler)
        logger.info("Adapter '%s' logging to %s", adapter_name, adapter_log_path)

    logger.debug("Loaded config from: %s", config_path)

    adapter_cfg = config.adapters.get(adapter_name)

    if adapter_cfg is None:
        adapter_cfg = AdapterConfig()
        logger.warning(
            "No config for adapter '%s', using defaults",
            adapter_name,
        )

    # Propagate the gateway auth_token so the adapter can authenticate
    if config.gateway.auth_token and not adapter_cfg.auth_token:
        adapter_cfg.auth_token = config.gateway.auth_token

    if not adapter_cfg.enabled:
        logger.info(
            "Adapter '%s' is disabled in config (%s)",
            adapter_name,
            config_path,
        )
        return

    adapter: Any
    if adapter_name == "discord":
        from .discord import DiscordAdapter

        adapter = DiscordAdapter(
            adapter_cfg, stt_config=config.stt, tts_config=config.tts
        )
    elif adapter_name == "telegram":
        from .telegram import TelegramAdapter

        adapter = TelegramAdapter(
            adapter_cfg, stt_config=config.stt, tts_config=config.tts
        )
    elif adapter_name == "slack":
        from .slack import SlackAdapter

        adapter = SlackAdapter(adapter_cfg)
    elif adapter_name == "line":
        from .line import LineAdapter

        adapter = LineAdapter(adapter_cfg)
    elif adapter_name == "whatsapp":
        from .whatsapp import WhatsAppAdapter

        adapter = WhatsAppAdapter(adapter_cfg)
    elif adapter_name == "signal":
        from .signal import SignalAdapter

        adapter = SignalAdapter(adapter_cfg)
    else:
        logger.error("Unknown adapter: %s", adapter_name)
        return

    logger.info("Starting %s adapter...", adapter_name)

    judge_backend: Any = None
    if config.ai_decision is not None:
        try:
            from cordbeat.adapters._utils import set_judge_backend
            from cordbeat.ai.backend import create_backend

            judge_backend = create_backend(config.ai_decision)
            opts = config.ai_decision.options or {}
            judge_max_tokens_raw = opts.get("judge_max_tokens")
            judge_temperature_raw = opts.get("judge_temperature")
            try:
                judge_max_tokens = (
                    int(judge_max_tokens_raw)
                    if judge_max_tokens_raw is not None
                    else None
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid ai_decision.options.judge_max_tokens=%r; using default",
                    judge_max_tokens_raw,
                )
                judge_max_tokens = None
            try:
                judge_temperature = (
                    float(judge_temperature_raw)
                    if judge_temperature_raw is not None
                    else None
                )
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid ai_decision.options.judge_temperature=%r; using default",
                    judge_temperature_raw,
                )
                judge_temperature = None
            set_judge_backend(
                judge_backend,
                max_tokens=judge_max_tokens,
                temperature=judge_temperature,
            )
            logger.info(
                "ai_decision_llm judge backend ready (%s)",
                config.ai_decision.provider,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to initialise ai_decision_llm backend; "
                "respond_mode=ai_decision_llm will fall back to keyword matching",
                exc_info=True,
            )
            judge_backend = None

    try:
        await adapter.start()
    except KeyboardInterrupt:
        pass
    finally:
        await adapter.stop()
        if judge_backend is not None:
            try:
                from cordbeat.adapters._utils import set_judge_backend

                await judge_backend.aclose()
                set_judge_backend(None)
            except Exception:  # noqa: BLE001
                logger.debug("judge backend close failed", exc_info=True)
        logger.info("%s adapter stopped", adapter_name)


def discord_cli() -> None:
    config_path = _resolve_adapter_config_path()
    try:
        asyncio.run(_run_adapter("discord", config_path))
    except KeyboardInterrupt:
        pass


def telegram_cli() -> None:
    config_path = _resolve_adapter_config_path()
    try:
        asyncio.run(_run_adapter("telegram", config_path))
    except KeyboardInterrupt:
        pass


def slack_cli() -> None:
    config_path = _resolve_adapter_config_path()
    try:
        asyncio.run(_run_adapter("slack", config_path))
    except KeyboardInterrupt:
        pass


def line_cli() -> None:
    config_path = _resolve_adapter_config_path()
    try:
        asyncio.run(_run_adapter("line", config_path))
    except KeyboardInterrupt:
        pass


def whatsapp_cli() -> None:
    config_path = _resolve_adapter_config_path()
    try:
        asyncio.run(_run_adapter("whatsapp", config_path))
    except KeyboardInterrupt:
        pass


def signal_cli() -> None:
    config_path = _resolve_adapter_config_path()
    try:
        asyncio.run(_run_adapter("signal", config_path))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m cordbeat.adapter_runner <adapter_name> [config.yaml]")
        sys.exit(1)
    name = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    try:
        asyncio.run(_run_adapter(name, path))
    except KeyboardInterrupt:
        pass
