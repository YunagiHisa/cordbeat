"""Adapter runner — standalone entry points for platform adapters."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from cordbeat.config import AdapterConfig, load_config

logger = logging.getLogger("cordbeat")


async def _run_adapter(adapter_name: str, config_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = load_config(config_path)
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m cordbeat.adapter_runner <adapter_name> [config.yaml]")
        sys.exit(1)
    name = sys.argv[1]
    path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    asyncio.run(_run_adapter(name, path))
