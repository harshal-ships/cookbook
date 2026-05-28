"""HealthFirst inbound booking agent using Google ADK + Telcoflow."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from telcoflow_sdk import ActiveCall, TelcoflowClient, TelcoflowClientConfig
import telcoflow_sdk.events as events

from adk_voice import run_adk_voice_call
from config import require_env
from credentials import ensure_google_calendar_credentials
from post_call import make_gemini_client, process_booking_after_call

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def make_telcoflow_config() -> TelcoflowClientConfig:
    return TelcoflowClientConfig.sandbox(
        api_key=require_env("WSS_API_KEY"),
        connector_uuid=require_env("WSS_CONNECTOR_UUID"),
        sample_rate=24000,
    )


async def handle_incoming_call(call: ActiveCall, gemini_client) -> None:
    transcript: list = []
    try:
        transcript = await run_adk_voice_call(call)
    except Exception as exc:
        logger.exception("Call %s voice session failed", call.call_id)
        print(f"Call {call.call_id} voice failed: {exc}", file=sys.stderr)

    if not transcript:
        print(json.dumps({"call_id": call.call_id, "booking_result": {"status": "no_transcript"}}, indent=2))
        return

    result = await process_booking_after_call(gemini_client, call, transcript)
    print(json.dumps({"call_id": call.call_id, "booking_result": result}, indent=2))


async def main() -> None:
    ensure_google_calendar_credentials()
    gemini_client = make_gemini_client()
    config = make_telcoflow_config()

    async with TelcoflowClient(config) as client:
        @client.on(events.INCOMING_CALL)
        async def on_call(call: ActiveCall) -> None:
            try:
                await handle_incoming_call(call, gemini_client)
            except Exception as exc:
                logger.exception("Call %s failed", call.call_id)
                print(f"Call {call.call_id} failed: {exc}", file=sys.stderr)

        logger.info("HealthFirst ADK booking agent connected. Waiting for calls...")
        await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
