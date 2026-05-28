"""Unified entrypoint for booking and reminder workers."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def run_mode(mode: str) -> None:
    if mode == "booking":
        module = importlib.import_module("booking_agent")
        await module.main()
        return

    if mode == "reminder":
        module = importlib.import_module("reminder_agent")
        await module.main()
        return

    if mode == "both":
        module = importlib.import_module("combined_agent")
        await module.main()
        return

    raise RuntimeError(f"Unknown AGENT_MODE: {mode}")


def main() -> None:
    mode = os.getenv("AGENT_MODE", "booking")
    try:
        asyncio.run(run_mode(mode))
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as exc:
        logger.exception("Agent failed")
        print(f"Agent failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
