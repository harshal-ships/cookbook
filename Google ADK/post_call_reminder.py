"""Post-call reminder outcome processing without OpenClaw."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types
from telcoflow_sdk import ActiveCall

from bookings import Booking, BookingStore
from calendar_tools import get_calendar_client
from config import EXTRACTION_MODEL
from transcript import TranscriptLine, transcript_text

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    parsed = json.loads(match.group(0))
    return parsed if isinstance(parsed, dict) else None


async def process_reminder_after_call(
    gemini_client: genai.Client,
    call: ActiveCall,
    booking: Booking,
    transcript: list[TranscriptLine],
) -> dict[str, Any]:
    rendered_transcript = transcript_text(transcript)
    if not rendered_transcript.strip():
        raise RuntimeError("ADK did not return a reminder transcript to process.")

    prompt = f"""
You are Maya's post-reminder worker for HealthFirst Clinic.

Read the transcript and decide what the patient chose:
- confirmed -> status "reminder_sent"
- cancelled -> status "cancelled"
- rescheduled -> status "rescheduled" and include new appointment_date and appointment_time
- unclear -> status "needs_human_review"

Return only JSON:
{{
  "status": "reminder_sent" | "cancelled" | "rescheduled" | "needs_human_review",
  "appointment_date": "YYYY-MM-DD or null",
  "appointment_time": "HH:MM or null",
  "notes": "short operational note"
}}

Current booking:
{json.dumps(booking.__dict__, indent=2)}

Transcript:
{rendered_transcript}
""".strip()

    response = await gemini_client.aio.models.generate_content(
        model=EXTRACTION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
    )
    outcome = extract_json_object(response.text or "")
    if outcome is None:
        raise RuntimeError("Gemini reminder extraction did not return valid JSON.")

    status = outcome.get("status")
    store = BookingStore()
    calendar = get_calendar_client()

    if status == "reminder_sent":
        updated = store.update_booking(booking.id, {"status": "reminder_sent"})
        return {"status": "reminder_sent", "booking": updated, "notes": outcome.get("notes", "")}

    if status == "cancelled" and booking.calendar_event_id:
        calendar.delete_event(booking.calendar_event_id)
        updated = store.update_booking(booking.id, {"status": "cancelled"})
        return {"status": "cancelled", "booking": updated, "notes": outcome.get("notes", "")}

    if status == "rescheduled":
        new_date = str(outcome.get("appointment_date", "")).strip()
        new_time = str(outcome.get("appointment_time", "")).strip()
        if not new_date or not new_time:
            return {
                "status": "needs_human_review",
                "booking": None,
                "notes": "Reschedule requested but new date/time missing.",
            }
        booking_payload = {
            "appointment_date": new_date,
            "appointment_time": new_time,
        }
        if booking.calendar_event_id:
            calendar.update_event_time(booking.calendar_event_id, booking_payload)
        updated = store.update_booking(
            booking.id,
            {
                "appointment_date": new_date,
                "appointment_time": new_time[:5],
                "status": "confirmed",
            },
        )
        return {"status": "rescheduled", "booking": updated, "notes": outcome.get("notes", "")}

    return {
        "status": "needs_human_review",
        "booking": None,
        "notes": outcome.get("notes", "Reminder outcome unclear."),
    }
