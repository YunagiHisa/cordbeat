"""Adapter runner — standalone entry points for platform adapters."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from cordbeat.config import AdapterConfig, load_config

logger = logging.getLogger("cordbeat")


async def _run_adapter(adapter_name: str, config_path: str) -> None:
    config = load_config(config_path)

    logging.basicConfig(
        level=getattr(logging, config.log.level.upper(), logging.INFO),
        format=config.log.format,
    )

    adapter_cfg = config.adapters.get(adapter_name)

    if adapter_cfg is None:
        adapter_cfg = AdapterConfig()
        logger.warning(
            "No config for adapter '%s', using defaults",
            adapter_name,
        )

    if not adapter_cfg.enabled:
        logger.info("Adapter '%s' is disabled in config", adapter_name)
        return

    adapter: Any
    if adapter_name == "discord":
        from cordbeat.discord_adapter import DiscordAdapter

        adapter = DiscordAdapter(adapter_cfg)
    elif adapter_name == "telegram":
        from cordbeat.telegram_adapter import TelegramAdapter

        adapter = TelegramAdapter(adapter_cfg)
    elif adapter_name == "slack":
        from cordbeat.slack_adapter import SlackAdapter

        adapter = SlackAdapter(adapter_cfg)
    elif adapter_name == "line":
        from cordbeat.line_adapter import LineAdapter

        adapter = LineAdapter(adapter_cfg)
    elif adapter_name == "whatsapp":
        from cordbeat.whatsapp_adapter import WhatsAppAdapter

        adapter = WhatsAppAdapter(adapter_cfg)
    elif adapter_name == "signal":
        from cordbeat.signal_adapter import SignalAdapter

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
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    asyncio.run(_run_adapter("discord", config_path))


def telegram_cli() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    asyncio.run(_run_adapter("telegram", config_path))


def slack_cli() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    asyncio.run(_run_adapter("slack", config_path))


def line_cli() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    asyncio.run(_run_adapter("line", config_path))


def whatsapp_cli() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    asyncio.run(_run_adapter("whatsapp", config_path))


def signal_cli() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    asyncio.run(_run_adapter("signal", config_path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m cordbeat.adapter_runner <adapter_name> [config.yaml]")
        sys.exit(1)
    name = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    asyncio.run(_run_adapter(name, path))
