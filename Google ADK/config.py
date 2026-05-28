"""Environment and runtime configuration for the ADK HealthFirst agent."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

BOOKINGS_PATH = Path(os.getenv("BOOKINGS_PATH", "bookings.json")).resolve()
AUDIO_MIME_TYPE = "audio/pcm;rate=24000"
ADK_MODEL = os.getenv("ADK_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
EXTRACTION_MODEL = os.getenv("EXTRACTION_MODEL", "gemini-2.5-flash")
APP_NAME = os.getenv("ADK_APP_NAME", "healthfirst_adk")
CLINIC_TIMEZONE = os.getenv("CLINIC_TIMEZONE", "Asia/Singapore")
APPOINTMENT_DURATION_MINUTES = int(os.getenv("APPOINTMENT_DURATION_MINUTES", "30"))
LOG_TRANSCRIPTS = os.getenv("LOG_TRANSCRIPTS", "true").lower() in {"1", "true", "yes", "on"}
GOOGLE_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
