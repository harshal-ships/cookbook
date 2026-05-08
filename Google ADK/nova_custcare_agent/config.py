"""Environment-based configuration for the Nova customer care agent."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


@dataclass(frozen=True)
class NovaConfig:
    gemini_api_key: str
    gemini_model: str
    wss_api_key: str
    wss_connector_uuid: str
    sample_rate: int
    db_path: str
    business_name: str

    @classmethod
    def from_env(cls) -> NovaConfig:
        return cls(
            gemini_api_key=_require("GEMINI_API_KEY"),
            gemini_model=os.getenv(
                "GEMINI_MODEL",
                "gemini-2.5-flash-native-audio-preview-12-2025",
            ),
            wss_api_key=_require("WSS_API_KEY"),
            wss_connector_uuid=_require("WSS_CONNECTOR_UUID"),
            sample_rate=int(os.getenv("SAMPLE_RATE", "24000")),
            db_path=os.getenv("DB_PATH", "nova.db"),
            business_name=os.getenv("BUSINESS_NAME", "B3Networks"),
        )
