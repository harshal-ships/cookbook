"""Parse transcript slots and inject calendar availability into the live voice session."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from calendar_tools import _run_calendar_check, _run_find_next_slots
from config import CLINIC_TIMEZONE
from transcript import TranscriptLine


def _clinic_today() -> date:
    return datetime.now(ZoneInfo(CLINIC_TIMEZONE)).date()


def _patient_text(transcript: list[TranscriptLine]) -> str:
    return " ".join(line.text for line in transcript if line.speaker == "PATIENT").strip()


def _time_from_match(hour: int, minute: int, meridiem: str | None) -> str | None:
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"
    return None


def _parse_last_time_24h(text: str) -> str | None:
    """Return the last explicit time mentioned in text (patient's latest request wins)."""
    lowered = text.lower()
    candidates: list[tuple[int, str]] = []

    for match in re.finditer(r"(\d{1,2})\s*:\s*(\d{2})\s*(am|pm)?", lowered):
        parsed = _time_from_match(int(match.group(1)), int(match.group(2)), match.group(3))
        if parsed:
            candidates.append((match.start(), parsed))

    for match in re.finditer(r"(\d{1,2})\s*(am|pm)\b", lowered):
        parsed = _time_from_match(int(match.group(1)), 0, match.group(2))
        if parsed:
            candidates.append((match.start(), parsed))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _parse_last_date_iso(text: str, today: date) -> str | None:
    """Return the last explicit date mentioned in text."""
    lowered = text.lower()
    candidates: list[tuple[int, str]] = []

    if "today" in lowered:
        for match in re.finditer(r"\btoday\b", lowered):
            candidates.append((match.start(), today.isoformat()))
    if "tomorrow" in lowered:
        for match in re.finditer(r"\btomorrow\b", lowered):
            candidates.append((match.start(), (today + timedelta(days=1)).isoformat()))

    for match in re.finditer(r"\b(\d{4}-\d{2}-\d{2})\b", text):
        candidates.append((match.start(), match.group(1)))

    month_pattern = (
        r"\b(january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\b"
    )
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
    for match in re.finditer(month_pattern, lowered):
        month = month_names[match.group(1)]
        day = int(match.group(2))
        candidates.append((match.start(), date(today.year, month, day).isoformat()))

    for match in re.finditer(
        r"\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        lowered,
    ):
        day = int(match.group(1))
        month = month_names[match.group(2)]
        candidates.append((match.start(), date(today.year, month, day).isoformat()))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def patient_time_is_complete(transcript: list[TranscriptLine]) -> bool:
    """Wait until the patient has said a full time like 4 pm or 16:00."""
    if not transcript:
        return False
    recent_patient = " ".join(
        line.text for line in transcript if line.speaker == "PATIENT"
    )[-120:]
    lowered = recent_patient.lower()
    return bool(
        re.search(r"\d{1,2}\s*(am|pm)\b", lowered)
        or re.search(r"\d{1,2}\s*:\s*\d{2}", lowered)
    )


def parse_appointment_slot(transcript: list[TranscriptLine]) -> tuple[str, str] | None:
    """Parse date/time from patient speech only — ignore Maya's offered alternatives."""
    patient_only = _patient_text(transcript)
    if not patient_only:
        return None

    today = _clinic_today()
    appointment_date = _parse_last_date_iso(patient_only, today)
    appointment_time = _parse_last_time_24h(patient_only)

    if appointment_date and appointment_time:
        return appointment_date, appointment_time
    return None


def build_calendar_system_message(
    transcript: list[TranscriptLine],
    *,
    is_update: bool = False,
) -> str | None:
    slot = parse_appointment_slot(transcript)
    if slot is None:
        return None

    appointment_date, appointment_time = slot
    check = _run_calendar_check(appointment_date, appointment_time)
    prefix = "[Calendar system] UPDATE:" if is_update else "[Calendar system]"
    if check.get("available"):
        return (
            f"{prefix} The slot on {appointment_date} at {appointment_time} is AVAILABLE. "
            "This is the final calendar result for this date and time. "
            "Tell the patient this time is open, read back all details once, and ask them to confirm "
            "before ending the call. "
            "If you previously said a different availability for this same request, correct yourself briefly using this message."
        )

    alternatives = _run_find_next_slots(appointment_date, appointment_time, 3)
    return (
        f"{prefix} The slot on {appointment_date} at {appointment_time} is NOT AVAILABLE. "
        f"{alternatives.get('message', '')} "
        "This is the final calendar result for this date and time. "
        "Do not confirm the requested time. Offer these alternatives and ask the patient to choose one. "
        "If you previously said this time was available, apologize and use this updated result instead."
    )
