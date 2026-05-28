"""Direct Google Calendar API client for HealthFirst bookings."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build

from config import APPOINTMENT_DURATION_MINUTES, CLINIC_TIMEZONE, GOOGLE_CALENDAR_SCOPES


@dataclass(frozen=True)
class AppointmentRange:
    start: datetime
    end: datetime


class GoogleCalendarClient:
    def __init__(
        self,
        credentials_path: str,
        calendar_id: str,
        timezone_name: str = CLINIC_TIMEZONE,
    ):
        self.calendar_id = calendar_id
        self.timezone_name = timezone_name
        self.timezone = ZoneInfo(timezone_name)
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=GOOGLE_CALENDAR_SCOPES,
        )
        self.service = build("calendar", "v3", credentials=credentials, cache_discovery=False)

    def appointment_range(self, booking: dict[str, Any]) -> AppointmentRange:
        appointment_date = str(booking["appointment_date"]).strip()
        appointment_time = str(booking["appointment_time"]).strip()
        if re.fullmatch(r"\d{2}:\d{2}", appointment_time):
            appointment_time = f"{appointment_time}:00"

        start = datetime.fromisoformat(f"{appointment_date}T{appointment_time}")
        if start.tzinfo is None:
            start = start.replace(tzinfo=self.timezone)
        end = start + timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
        return AppointmentRange(start=start, end=end)

    def is_available(self, appointment: AppointmentRange) -> bool:
        body = {
            "timeMin": appointment.start.isoformat(),
            "timeMax": appointment.end.isoformat(),
            "timeZone": self.timezone_name,
            "items": [{"id": self.calendar_id}],
        }
        response = self.service.freebusy().query(body=body).execute()
        busy_blocks = response.get("calendars", {}).get(self.calendar_id, {}).get("busy", [])
        return not busy_blocks

    def create_event(
        self,
        booking: dict[str, Any],
        appointment: AppointmentRange,
        call_id: str,
    ) -> dict[str, Any]:
        patient_name = str(booking["patient_name"]).strip()
        appointment_type = str(booking["appointment_type"]).strip()
        phone_number = str(booking["phone_number"]).strip()
        event = {
            "summary": f"HealthFirst {appointment_type} - {patient_name}",
            "description": (
                "Booked by Maya ADK phone agent.\n"
                f"Patient: {patient_name}\n"
                f"Phone: {phone_number}\n"
                f"Appointment type: {appointment_type}\n"
                f"Telcoflow call id: {call_id}"
            ),
            "start": {
                "dateTime": appointment.start.isoformat(),
                "timeZone": self.timezone_name,
            },
            "end": {
                "dateTime": appointment.end.isoformat(),
                "timeZone": self.timezone_name,
            },
        }
        return self.service.events().insert(calendarId=self.calendar_id, body=event).execute()

    def next_available_slots(
        self,
        requested: AppointmentRange,
        count: int = 5,
    ) -> list[dict[str, str]]:
        slots: list[dict[str, str]] = []
        candidate_start = requested.start + timedelta(minutes=APPOINTMENT_DURATION_MINUTES)
        clinic_open_hour = int(os.getenv("CLINIC_OPEN_HOUR", "9"))
        clinic_close_hour = int(os.getenv("CLINIC_CLOSE_HOUR", "22"))

        while len(slots) < count:
            if candidate_start.hour < clinic_open_hour:
                candidate_start = candidate_start.replace(
                    hour=clinic_open_hour, minute=0, second=0, microsecond=0
                )
            if candidate_start.hour >= clinic_close_hour:
                next_day = candidate_start + timedelta(days=1)
                candidate_start = next_day.replace(
                    hour=clinic_open_hour, minute=0, second=0, microsecond=0
                )

            candidate = AppointmentRange(
                start=candidate_start,
                end=candidate_start + timedelta(minutes=APPOINTMENT_DURATION_MINUTES),
            )
            if self.is_available(candidate):
                slots.append(
                    {
                        "appointment_date": candidate.start.date().isoformat(),
                        "appointment_time": candidate.start.strftime("%H:%M"),
                    }
                )
            candidate_start += timedelta(minutes=APPOINTMENT_DURATION_MINUTES)

        return slots

    def delete_event(self, event_id: str) -> None:
        self.service.events().delete(calendarId=self.calendar_id, eventId=event_id).execute()

    def update_event_time(
        self,
        event_id: str,
        booking: dict[str, Any],
    ) -> dict[str, Any]:
        appointment = self.appointment_range(booking)
        body = {
            "start": {
                "dateTime": appointment.start.isoformat(),
                "timeZone": self.timezone_name,
            },
            "end": {
                "dateTime": appointment.end.isoformat(),
                "timeZone": self.timezone_name,
            },
        }
        return (
            self.service.events()
            .patch(calendarId=self.calendar_id, eventId=event_id, body=body)
            .execute()
        )
