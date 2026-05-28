"""ADK function tools for live Google Calendar availability checks."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from google_calendar import GoogleCalendarClient
from config import require_env
from credentials import ensure_google_calendar_credentials

logger = logging.getLogger(__name__)

_calendar_client: GoogleCalendarClient | None = None
_calendar_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="calendar-tools")


def get_calendar_client() -> GoogleCalendarClient:
    global _calendar_client
    if _calendar_client is None:
        _calendar_client = GoogleCalendarClient(
            credentials_path=ensure_google_calendar_credentials(),
            calendar_id=require_env("GOOGLE_CALENDAR_ID"),
        )
    return _calendar_client


def _run_calendar_check(appointment_date: str, appointment_time: str) -> dict[str, Any]:
    calendar = get_calendar_client()
    appointment = calendar.appointment_range(
        {"appointment_date": appointment_date, "appointment_time": appointment_time}
    )
    available = calendar.is_available(appointment)
    logger.info(
        "Tool check_appointment_availability %s %s -> %s",
        appointment_date,
        appointment_time,
        available,
    )
    if available:
        return {
            "status": "available",
            "available": True,
            "appointment_date": appointment_date,
            "appointment_time": appointment_time,
            "message": f"The slot on {appointment_date} at {appointment_time} is available.",
        }
    return {
        "status": "unavailable",
        "available": False,
        "appointment_date": appointment_date,
        "appointment_time": appointment_time,
        "message": f"The slot on {appointment_date} at {appointment_time} is not available.",
    }


def _run_find_next_slots(
    appointment_date: str,
    appointment_time: str,
    count: int,
) -> dict[str, Any]:
    calendar = get_calendar_client()
    requested = calendar.appointment_range(
        {"appointment_date": appointment_date, "appointment_time": appointment_time}
    )
    slots = calendar.next_available_slots(requested, count=max(1, min(count, 5)))
    logger.info(
        "Tool find_next_available_slots %s %s -> %d slots",
        appointment_date,
        appointment_time,
        len(slots),
    )
    if not slots:
        return {
            "status": "none_found",
            "slots": [],
            "message": "No nearby alternative slots were found.",
        }
    summary = ", ".join(
        f"{slot['appointment_date']} at {slot['appointment_time']}" for slot in slots
    )
    return {
        "status": "ok",
        "slots": slots,
        "message": f"Next available options: {summary}.",
    }


def _run_in_background(func, *args, **kwargs) -> Any:
    try:
        asyncio.get_running_loop()
        in_async = True
    except RuntimeError:
        in_async = False

    if not in_async:
        return func(*args, **kwargs)

    future = _calendar_executor.submit(func, *args, **kwargs)
    return future.result(timeout=30)


def check_appointment_availability(appointment_date: str, appointment_time: str) -> dict[str, Any]:
    """Check whether the clinic calendar is free for one appointment slot.

    Args:
        appointment_date: Preferred date in YYYY-MM-DD format.
        appointment_time: Preferred start time in HH:MM 24-hour format.

    Returns:
        Dictionary with availability status and a short message for the caller.
    """
    try:
        return _run_in_background(_run_calendar_check, appointment_date, appointment_time)
    except Exception as exc:
        logger.exception("check_appointment_availability failed")
        return {
            "status": "error",
            "available": False,
            "message": f"Could not check calendar availability: {exc}",
        }


def find_next_available_slots(appointment_date: str, appointment_time: str, count: int = 3) -> dict[str, Any]:
    """Find nearby alternative appointment slots when the requested time is busy.

    Args:
        appointment_date: Requested date in YYYY-MM-DD format.
        appointment_time: Requested start time in HH:MM 24-hour format.
        count: Number of alternative slots to return. Defaults to 3.

    Returns:
        Dictionary with alternative slot list and a short summary message.
    """
    try:
        return _run_in_background(_run_find_next_slots, appointment_date, appointment_time, count)
    except Exception as exc:
        logger.exception("find_next_available_slots failed")
        return {
            "status": "error",
            "slots": [],
            "message": f"Could not search for alternative slots: {exc}",
        }


CALENDAR_TOOLS = [check_appointment_availability, find_next_available_slots]
