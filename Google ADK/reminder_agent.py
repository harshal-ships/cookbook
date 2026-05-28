"""HealthFirst reminder worker using ADK voice + Gemini post-call updates."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta

from telcoflow_sdk import ActiveCall, TelcoflowClient
import telcoflow_sdk.events as events

from adk_voice import run_adk_voice_call
from bookings import Booking, BookingStore
from calendar_tools import CALENDAR_TOOLS
from config import require_env
from credentials import ensure_google_calendar_credentials
from notify import send_telegram_message
from post_call import make_gemini_client
from post_call_reminder import process_reminder_after_call
from prompts import MAYA_REMINDER_INSTRUCTION

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = int(os.getenv("REMINDER_CHECK_INTERVAL_SECONDS", str(60 * 60)))


def make_telcoflow_config():
    from telcoflow_sdk import TelcoflowClientConfig

    return TelcoflowClientConfig.sandbox(
        api_key=require_env("WSS_API_KEY"),
        connector_uuid=require_env("WSS_CONNECTOR_UUID"),
        sample_rate=24000,
    )


def make_reminder_prompt(booking: Booking) -> str:
    return f"""{MAYA_REMINDER_INSTRUCTION}

Patient name: {booking.patient_name}
Appointment type: {booking.appointment_type}
Appointment date: {booking.appointment_date}
Appointment time: {booking.appointment_time}
Phone number: {booking.phone_number}
"""


class ReminderCoordinator:
    def __init__(self, store: BookingStore, gemini_client):
        self.store = store
        self.gemini_client = gemini_client
        self.sent_booking_ids: set[str] = set()
        self.pending_calls_by_phone: dict[str, Booking] = {}

    async def hourly_loop(self) -> None:
        while True:
            try:
                await self.check_due_bookings()
            except Exception as exc:
                logger.exception("Reminder check failed: %s", exc)
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)

    async def check_due_bookings(self) -> None:
        now = datetime.now()
        due_before = now + timedelta(hours=24)

        for booking in self.store.load():
            if booking.status != "confirmed":
                continue
            if booking.id in self.sent_booking_ids:
                continue
            appointment_at = booking.appointment_datetime
            if now <= appointment_at <= due_before:
                await self.send_reminder(booking)

    async def send_reminder(self, booking: Booking) -> None:
        message = (
            "HealthFirst appointment reminder\n"
            f"Patient: {booking.patient_name}\n"
            f"Phone: {booking.phone_number}\n"
            f"Type: {booking.appointment_type}\n"
            f"Date: {booking.appointment_date}\n"
            f"Time: {booking.appointment_time}"
        )
        result = await send_telegram_message(message)
        updated = self.store.update_booking(booking.id, {"status": "reminder_sent"})
        self.sent_booking_ids.add(booking.id)
        self.pending_calls_by_phone[booking.phone_number] = booking
        print(json.dumps({"reminder_sent": updated, "notify_result": result, "message": message}, indent=2))

    async def handle_telcoflow_call(self, call: ActiveCall) -> None:
        booking = self.match_booking_for_call(call)
        if booking is None:
            logger.warning("No pending reminder matched call %s", call.call_id)
            await call.disconnect()
            return

        transcript = await run_adk_voice_call(
            call,
            instruction=make_reminder_prompt(booking),
            tools=CALENDAR_TOOLS,
        )
        result = await process_reminder_after_call(self.gemini_client, call, booking, transcript)
        print(json.dumps({"call_id": call.call_id, "reminder_result": result}, indent=2))
        self.pending_calls_by_phone.pop(booking.phone_number, None)

    def match_booking_for_call(self, call: ActiveCall) -> Booking | None:
        for phone in (call.caller_number, call.callee_number):
            if phone in self.pending_calls_by_phone:
                return self.pending_calls_by_phone[phone]
        return None


async def main() -> None:
    ensure_google_calendar_credentials()
    coordinator = ReminderCoordinator(
        store=BookingStore(),
        gemini_client=make_gemini_client(),
    )
    config = make_telcoflow_config()

    async with TelcoflowClient(config) as client:
        @client.on(events.INCOMING_CALL)
        async def on_call(call: ActiveCall) -> None:
            try:
                await coordinator.handle_telcoflow_call(call)
            except Exception as exc:
                logger.exception("Reminder call %s failed", call.call_id)
                print(f"Reminder call {call.call_id} failed: {exc}", file=sys.stderr)
                await call.disconnect()

        logger.info("HealthFirst ADK reminder worker connected.")
        await asyncio.gather(client.run_forever(), coordinator.hourly_loop())


if __name__ == "__main__":
    asyncio.run(main())
