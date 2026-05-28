"""Google Calendar service-account credential loading."""

from __future__ import annotations

import json
import os
from pathlib import Path


def ensure_google_calendar_credentials() -> str:
    credentials_json = os.getenv("GOOGLE_CALENDAR_CREDENTIALS_JSON")
    if credentials_json:
        target_path = Path(
            os.getenv("GOOGLE_CALENDAR_CREDENTIALS_PATH", "./google-calendar-credentials.json")
        ).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            credentials_data = json.loads(credentials_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "GOOGLE_CALENDAR_CREDENTIALS_JSON must be one complete JSON object."
            ) from exc

        target_path.write_text(json.dumps(credentials_data), encoding="utf-8")
        target_path.chmod(0o600)
        os.environ["GOOGLE_CALENDAR_CREDENTIALS"] = str(target_path)
        return str(target_path)

    credentials_path = os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
    if credentials_path:
        return credentials_path

    raise RuntimeError(
        "Missing Google Calendar credentials. Set GOOGLE_CALENDAR_CREDENTIALS "
        "to a file path or GOOGLE_CALENDAR_CREDENTIALS_JSON to the full JSON object."
    )
