"""Local bookings.json persistence."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from config import BOOKINGS_PATH


@dataclass(frozen=True)
class Booking:
    id: str
    patient_name: str
    phone_number: str
    appointment_date: str
    appointment_time: str
    appointment_type: str
    status: str
    calendar_event_id: str

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Booking":
        return cls(
            id=str(data["id"]),
            patient_name=str(data["patient_name"]),
            phone_number=str(data["phone_number"]),
            appointment_date=str(data["appointment_date"]),
            appointment_time=str(data["appointment_time"]),
            appointment_type=str(data["appointment_type"]),
            status=str(data["status"]),
            calendar_event_id=str(data.get("calendar_event_id", "")),
        )

    @property
    def appointment_datetime(self) -> datetime:
        return datetime.fromisoformat(f"{self.appointment_date}T{self.appointment_time}")


class BookingStore:
    def __init__(self, path: Path = BOOKINGS_PATH):
        self.path = path

    def load(self) -> list[Booking]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        bookings = data.get("bookings", [])
        if not isinstance(bookings, list):
            raise RuntimeError(f"{self.path} must contain a top-level bookings list.")
        return [Booking.from_json(item) for item in bookings if isinstance(item, dict)]

    def save_all(self, bookings: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"bookings": bookings}
        tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def update_booking(self, booking_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"bookings": []}
        bookings = data.get("bookings", [])
        for index, item in enumerate(bookings):
            if str(item.get("id")) == booking_id:
                bookings[index] = {**item, **updates}
                self.save_all(bookings)
                return bookings[index]
        return None


def build_booking_record(booking: dict[str, str], event: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(uuid.uuid4()),
        "patient_name": booking["patient_name"],
        "phone_number": booking["phone_number"],
        "appointment_date": booking["appointment_date"],
        "appointment_time": booking["appointment_time"][:5],
        "appointment_type": booking["appointment_type"],
        "status": "confirmed",
        "calendar_event_id": str(event["id"]),
    }


def append_booking_record(booking: dict[str, str]) -> None:
    store = BookingStore()
    existing: list[dict[str, Any]] = []
    if store.path.exists():
        existing = json.loads(store.path.read_text(encoding="utf-8")).get("bookings", [])
    existing.append(booking)
    store.save_all(existing)
