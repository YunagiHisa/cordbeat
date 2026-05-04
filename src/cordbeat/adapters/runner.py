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
    try:
        await adapter.start()
    except KeyboardInterrupt:
        pass
    finally:
        await adapter.stop()
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
