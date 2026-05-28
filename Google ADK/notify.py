"""Optional Telegram notifications."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _telegram_config() -> tuple[str, str] | None:
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CONFIRM_CHAT_ID", os.getenv("TELEGRAM_ALLOW_FROM", "")).strip()
    if not bot_token or not chat_id:
        return None
    return bot_token, chat_id


async def send_telegram_message(message: str) -> dict[str, Any]:
    config = _telegram_config()
    if config is None:
        return {
            "message_sent": False,
            "channel": "none",
            "notes": "Telegram not configured; skipped message.",
        }

    bot_token, chat_id = config
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(url, json={"chat_id": chat_id, "text": message})
        response.raise_for_status()

    logger.info("Sent Telegram message to chat_id=%s", chat_id)
    return {
        "message_sent": True,
        "channel": "telegram",
        "notes": "Message sent through Telegram Bot API.",
    }


async def send_telegram_confirmation(booking: dict[str, str]) -> dict[str, Any]:
    message = (
        "HealthFirst appointment confirmed\n"
        f"Patient: {booking['patient_name']}\n"
        f"Phone: {booking['phone_number']}\n"
        f"Type: {booking['appointment_type']}\n"
        f"Date: {booking['appointment_date']}\n"
        f"Time: {booking['appointment_time']}"
    )
    return await send_telegram_message(message)


async def send_telegram_review_alert(
    call_id: str,
    notes: str,
    caller_number: str | None = None,
) -> dict[str, Any]:
    message = f"Call {call_id} needs review: {notes}"
    if caller_number:
        message += f"\nCaller: {caller_number}"
    return await send_telegram_message(message)


async def send_telegram_unavailable_alert(
    call_id: str,
    patient_name: str | None,
    appointment_date: str,
    appointment_time: str,
    next_available_slots: list[dict[str, str]],
    caller_number: str | None = None,
) -> dict[str, Any]:
    message = (
        f"Call {call_id}: requested slot unavailable\n"
        f"Patient: {patient_name or 'unknown'}\n"
        f"Requested: {appointment_date} at {appointment_time}"
    )
    if caller_number:
        message += f"\nCaller: {caller_number}"
    if next_available_slots:
        options = ", ".join(
            f"{slot['appointment_date']} {slot['appointment_time']}" for slot in next_available_slots
        )
        message += f"\nNext options: {options}"
    return await send_telegram_message(message)
