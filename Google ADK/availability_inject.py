"""Parse transcript slots and inject calendar availability into the live voice session."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from calendar_tools import _run_calendar_check, _run_find_next_slots
from config import CLINIC_TIMEZONE
from transcript import TranscriptLine, transcript_text


def _clinic_today() -> date:
    return datetime.now(ZoneInfo(CLINIC_TIMEZONE)).date()


def _parse_time_24h(text: str) -> str | None:
    lowered = text.lower()
    match = re.search(r"(\d{1,2})\s*:\s*(\d{2})\s*(am|pm)?", lowered)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        meridiem = match.group(3)
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    match = re.search(r"(\d{1,2})\s*(am|pm)\b", lowered)
    if match:
        hour = int(match.group(1))
        meridiem = match.group(2)
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:00"

    return None


def _parse_date_iso(text: str, today: date) -> str | None:
    lowered = text.lower()
    if "today" in lowered:
        return today.isoformat()
    if "tomorrow" in lowered:
        return (today + timedelta(days=1)).isoformat()

    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
    if match:
        return match.group(1)

    match = re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b",
        lowered,
    )
    if match:
        month_names = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        month = month_names[match.group(1)]
        day = int(match.group(2))
        return date(today.year, month, day).isoformat()

    match = re.search(r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b", lowered)
    if match:
        month_names = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }
        day = int(match.group(1))
        month = month_names[match.group(2)]
        return date(today.year, month, day).isoformat()

    return None


def parse_appointment_slot(transcript: list[TranscriptLine]) -> tuple[str, str] | None:
    """Best-effort date/time extraction from recent transcript text."""
    if not transcript:
        return None

    today = _clinic_today()
    recent = transcript[-12:]
    chunks = [line.text for line in recent]
    combined = " ".join(chunks)

    appointment_date = _parse_date_iso(combined, today)
    appointment_time = _parse_time_24h(combined)

    if not appointment_date or not appointment_time:
        full_text = transcript_text(transcript)
        appointment_date = appointment_date or _parse_date_iso(full_text, today)
        appointment_time = appointment_time or _parse_time_24h(full_text)

    if appointment_date and appointment_time:
        return appointment_date, appointment_time
    return None


def build_calendar_system_message(transcript: list[TranscriptLine]) -> str | None:
    slot = parse_appointment_slot(transcript)
    if slot is None:
        return None

    appointment_date, appointment_time = slot
    check = _run_calendar_check(appointment_date, appointment_time)
    if check.get("available"):
        return (
            f"[Calendar system] The slot on {appointment_date} at {appointment_time} is AVAILABLE. "
            "Tell the patient this time is open, read back all details once, and ask them to confirm "
            "before ending the call."
        )

    alternatives = _run_find_next_slots(appointment_date, appointment_time, 3)
    return (
        f"[Calendar system] The slot on {appointment_date} at {appointment_time} is NOT AVAILABLE. "
        f"{alternatives.get('message', '')} "
        "Do not confirm the requested time. Offer these alternatives and ask the patient to choose one."
    )
