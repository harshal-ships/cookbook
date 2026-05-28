"""Post-call booking extraction and Google Calendar writes."""

from __future__ import annotations

import json
import logging
import re
import asyncio
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from google import genai
from google.genai import types
from google.genai.errors import ServerError
from telcoflow_sdk import ActiveCall

from bookings import append_booking_record, build_booking_record
from google_calendar import GoogleCalendarClient
from config import CLINIC_TIMEZONE, EXTRACTION_MODEL, require_env
from credentials import ensure_google_calendar_credentials
from notify import (
    send_telegram_confirmation,
    send_telegram_review_alert,
    send_telegram_unavailable_alert,
)
from transcript import TranscriptLine, transcript_text

logger = logging.getLogger(__name__)


def make_gemini_client() -> genai.Client:
    return genai.Client(api_key=require_env("GOOGLE_API_KEY"))


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


async def extract_booking_from_transcript(
    gemini_client: genai.Client,
    call: ActiveCall,
    transcript: list[TranscriptLine],
) -> dict[str, Any]:
    rendered_transcript = transcript_text(transcript)
    if not rendered_transcript.strip():
        raise RuntimeError("ADK did not return a transcript to process.")

    today = datetime.now(ZoneInfo(CLINIC_TIMEZONE)).date().isoformat()
    prompt = f"""
You are Maya's post-call extraction worker for HealthFirst Clinic.

Extract structured appointment details from the transcript only.
Today is {today}; resolve relative dates into YYYY-MM-DD.

Important extraction rules:
- Maya's final read-back before ending the call is the authoritative source when the patient agrees.
- If Maya summarizes the appointment and asks "Is that correct?" (or similar), and the patient replies with any affirmative phrase such as "yes", "yeah", "correct", "all right", "alright", "okay", "that's right", "perfect", "sounds good", or equivalent in any language, use Maya's summarized details and return status "extracted".
- Prefer Maya's final confirmed values over earlier noisy patient speech-to-text, especially for phone numbers and appointment times.
- Only return "needs_human_review" when a required field is missing or the patient explicitly contradicts or corrects Maya's final read-back.
- Normalize phone_number to digits only, optionally with a leading +. If Maya confirmed a specific phone number and the patient agreed, use Maya's number.
- Use caller_number metadata only when Maya said the patient is calling from that number or did not state any specific phone number: {call.caller_number or "unknown"}
- Post-call booking creates the Google Calendar event; Maya checks availability during the call but does not book during the call.

Return only one JSON object with:
{{
  "status": "extracted" | "needs_human_review",
  "booking": null | {{
    "patient_name": "string",
    "phone_number": "string",
    "appointment_date": "YYYY-MM-DD",
    "appointment_time": "HH:MM",
    "appointment_type": "general checkup|specialist|follow-up"
  }},
  "notes": "short operational note"
}}

Telcoflow call metadata:
- call_id: {call.call_id}
- caller_number: {call.caller_number}
- callee_number: {call.callee_number}

Transcript:
{rendered_transcript}
""".strip()

    response = None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = await gemini_client.aio.models.generate_content(
                model=EXTRACTION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
            break
        except ServerError as exc:
            last_error = exc
            if exc.code != 503 or attempt == 2:
                raise
            wait_seconds = 2 ** attempt
            logger.warning(
                "Gemini extraction unavailable (503), retrying in %ss (attempt %s/3)",
                wait_seconds,
                attempt + 1,
            )
            await asyncio.sleep(wait_seconds)

    if response is None:
        raise last_error or RuntimeError("Gemini extraction failed.")

    parsed = extract_json_object(response.text or "")
    if parsed is None:
        raise RuntimeError("Gemini extraction did not return valid JSON.")
    return parsed


async def retry_confirmed_booking_extraction(
    gemini_client: genai.Client,
    call: ActiveCall,
    transcript: list[TranscriptLine],
) -> dict[str, Any]:
    rendered_transcript = transcript_text(transcript)
    today = datetime.now(ZoneInfo(CLINIC_TIMEZONE)).date().isoformat()
    prompt = f"""
The caller already confirmed Maya's final read-back with yes, all right, okay, or similar.

Extract the booking ONLY from Maya's final confirmation summary in the transcript.
Use Maya's values for name, phone, date, time, and appointment type.
Do not return needs_human_review because of earlier garbled patient speech if the patient agreed to Maya's summary.
Today is {today}; resolve relative dates into YYYY-MM-DD.

Return only one JSON object with:
{{
  "status": "extracted" | "needs_human_review",
  "booking": null | {{
    "patient_name": "string",
    "phone_number": "string",
    "appointment_date": "YYYY-MM-DD",
    "appointment_time": "HH:MM",
    "appointment_type": "general checkup|specialist|follow-up"
  }},
  "notes": "short operational note"
}}

Telcoflow caller_number (use only if Maya said calling from this number): {call.caller_number or "unknown"}

Transcript:
{rendered_transcript}
""".strip()

    response = await gemini_client.aio.models.generate_content(
        model=EXTRACTION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
        ),
    )
    parsed = extract_json_object(response.text or "")
    if parsed is None:
        raise RuntimeError("Gemini confirmation retry did not return valid JSON.")
    return parsed


def normalize_phone_number(value: str, caller_number: str | None = None, *, use_caller_fallback: bool = False) -> str:
    digits = re.sub(r"\D", "", value)
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    if len(digits) == 10:
        return digits
    if use_caller_fallback and caller_number:
        caller_digits = re.sub(r"\D", "", caller_number)
        if caller_digits.startswith("91") and len(caller_digits) == 12:
            caller_digits = caller_digits[2:]
        if len(caller_digits) == 10:
            return caller_digits
    return digits


def find_phone_in_maya_transcript(transcript: list[TranscriptLine]) -> str | None:
    for line in reversed(transcript):
        if line.speaker != "MAYA":
            continue
        for match in re.finditer(r"\+?\d[\d\s]{8,14}\d", line.text):
            digits = re.sub(r"\D", "", match.group(0))
            if digits.startswith("91") and len(digits) == 12:
                digits = digits[2:]
            if len(digits) == 10:
                return digits
    return None


def resolve_phone_number(
    extracted_phone: str,
    call: ActiveCall,
    transcript: list[TranscriptLine],
) -> str:
    normalized = normalize_phone_number(extracted_phone)
    if len(normalized) == 10:
        caller_digits = re.sub(r"\D", "", call.caller_number or "")
        if caller_digits.startswith("91") and len(caller_digits) == 12:
            caller_digits = caller_digits[2:]
        if normalized != caller_digits:
            return normalized

    maya_phone = find_phone_in_maya_transcript(transcript)
    if maya_phone:
        return maya_phone

    if extracted_phone.lower() in {"caller_number", "calling_number", "unknown"}:
        return normalize_phone_number(
            call.caller_number or "",
            use_caller_fallback=True,
        )

    return normalized


def normalize_extracted_booking(
    extraction: dict[str, Any],
    call: ActiveCall,
    transcript: list[TranscriptLine] | None = None,
) -> dict[str, str] | None:
    if extraction.get("status") not in {"extracted", "confirmed"}:
        return None
    booking = extraction.get("booking")
    if not isinstance(booking, dict):
        return None

    required_fields = [
        "patient_name",
        "phone_number",
        "appointment_date",
        "appointment_time",
        "appointment_type",
    ]
    normalized: dict[str, str] = {}
    for field in required_fields:
        value = str(booking.get(field, "")).strip()
        if not value or value.lower() in {"unknown", "null", "none"}:
            return None
        normalized[field] = value

    if normalized["phone_number"].lower() == "caller_number":
        normalized["phone_number"] = call.caller_number or ""
    normalized["phone_number"] = resolve_phone_number(
        normalized["phone_number"],
        call,
        transcript or [],
    )
    if len(normalized["phone_number"]) != 10:
        return None
    return normalized


async def process_booking_after_call(
    gemini_client: genai.Client,
    call: ActiveCall,
    transcript: list[TranscriptLine],
) -> dict[str, Any]:
    calendar = GoogleCalendarClient(
        credentials_path=ensure_google_calendar_credentials(),
        calendar_id=require_env("GOOGLE_CALENDAR_ID"),
    )

    logger.info("Call %s: extracting booking details with Gemini", call.call_id)
    extraction = await extract_booking_from_transcript(gemini_client, call, transcript)
    booking = normalize_extracted_booking(extraction, call, transcript)
    if booking is None and extraction.get("status") == "needs_human_review":
        logger.info(
            "Call %s: retrying extraction using Maya's confirmed read-back",
            call.call_id,
        )
        extraction = await retry_confirmed_booking_extraction(gemini_client, call, transcript)
        booking = normalize_extracted_booking(extraction, call, transcript)
    if booking is None:
        notes = extraction.get("notes", "Could not extract a complete booking.")
        logger.info(
            "Call %s: booking needs human review. status=%s notes=%s",
            call.call_id,
            extraction.get("status"),
            notes,
        )
        review_result = await send_telegram_review_alert(
            call.call_id,
            notes,
            call.caller_number,
        )
        return {
            "status": "needs_human_review",
            "booking": None,
            "next_available_slots": [],
            "message_sent": review_result.get("message_sent") is True,
            "notes": notes,
        }

    appointment = calendar.appointment_range(booking)
    logger.info(
        "Call %s: checking calendar availability for %s to %s",
        call.call_id,
        appointment.start.isoformat(),
        appointment.end.isoformat(),
    )
    if not calendar.is_available(appointment):
        logger.info("Call %s: requested calendar slot is unavailable", call.call_id)
        next_slots = calendar.next_available_slots(appointment)
        unavailable_result = await send_telegram_unavailable_alert(
            call.call_id,
            booking.get("patient_name"),
            booking["appointment_date"],
            booking["appointment_time"],
            next_slots,
            call.caller_number,
        )
        return {
            "status": "unavailable",
            "booking": None,
            "next_available_slots": next_slots,
            "message_sent": unavailable_result.get("message_sent") is True,
            "notes": "Requested slot is not available in Google Calendar.",
        }

    event = calendar.create_event(booking, appointment, call.call_id)
    logger.info("Call %s: created Google Calendar event %s", call.call_id, event.get("id"))
    booking_record = build_booking_record(booking, event)
    append_booking_record(booking_record)
    message_result = await send_telegram_confirmation(booking_record)

    return {
        "status": "confirmed",
        "booking": booking_record,
        "next_available_slots": [],
        "message_sent": message_result.get("message_sent") is True,
        "notes": message_result.get("notes", "Calendar event created and booking stored."),
    }
