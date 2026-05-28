"""Run booking and reminder workers on one Telcoflow connection."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from telcoflow_sdk import ActiveCall, TelcoflowClient
import telcoflow_sdk.events as events

from booking_agent import handle_incoming_call, make_telcoflow_config
from bookings import BookingStore
from credentials import ensure_google_calendar_credentials
from post_call import make_gemini_client
from reminder_agent import ReminderCoordinator

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    ensure_google_calendar_credentials()
    gemini_client = make_gemini_client()
    coordinator = ReminderCoordinator(
        store=BookingStore(),
        gemini_client=gemini_client,
    )
    config = make_telcoflow_config()

    async with TelcoflowClient(config) as client:
        @client.on(events.INCOMING_CALL)
        async def on_call(call: ActiveCall) -> None:
            try:
                if coordinator.match_booking_for_call(call) is not None:
                    await coordinator.handle_telcoflow_call(call)
                else:
                    await handle_incoming_call(call, gemini_client)
            except Exception as exc:
                logger.exception("Call %s failed", call.call_id)
                print(json.dumps({"call_id": call.call_id, "error": str(exc)}, indent=2), file=sys.stderr)
                await call.disconnect()

        logger.info("HealthFirst ADK combined agent connected.")
        await asyncio.gather(client.run_forever(), coordinator.hourly_loop())


if __name__ == "__main__":
    asyncio.run(main())
