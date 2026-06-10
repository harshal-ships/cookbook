"""After-hours voicemail agent — Telcoflow + Gemini Live + Telegram.

Based on: https://docs.agentao.com/use-cases/after-hours-voicemail

Business hours: connect caller to the original callee (pre-answer), then leave.
After hours: Gemini Live answers, collects a voicemail, saves transcript, notifies team via Telegram.

Run:
    python voicemail_agent.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, parse, request
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google import genai
from google.genai import types
from telcoflow_sdk import ActiveCall, TelcoflowClient, TelcoflowClientConfig
import telcoflow_sdk.events as events

load_dotenv()

AUDIO_MIME_TYPE = "audio/pcm;rate=24000"
VOICEMAILS_PATH = Path(os.getenv("VOICEMAILS_PATH", "voicemails.json")).resolve()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
BUSINESS_TIMEZONE = os.getenv("BUSINESS_TIMEZONE", "Asia/Kolkata")
BUSINESS_OPEN_HOUR = int(os.getenv("BUSINESS_OPEN_HOUR", "9"))
BUSINESS_CLOSE_HOUR = int(os.getenv("BUSINESS_CLOSE_HOUR", "17"))
LOG_TRANSCRIPTS = os.getenv("LOG_TRANSCRIPTS", "true").lower() in {"1", "true", "yes", "on"}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

VOICEMAIL_SYSTEM_PROMPT = """You are the after-hours voicemail assistant for this office.
The office is currently closed. Your only job is to take a voicemail.

1. Greet the caller warmly. Say the office is closed and ask them to leave a message after the tone.
2. Listen silently while they speak. Do not interrupt unless they ask a direct question.
3. When they finish speaking and there is a pause, briefly confirm you recorded their message.
4. End with: "Thank you. Your message has been recorded. We will get back to you on the next business day. Goodbye."
5. Keep responses short. Do not book appointments or answer business questions — only take the message."""


@dataclass(frozen=True)
class TranscriptLine:
    speaker: str
    text: str


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def make_telcoflow_config() -> TelcoflowClientConfig:
    return TelcoflowClientConfig.sandbox(
        api_key=require_env("WSS_API_KEY"),
        connector_uuid=require_env("WSS_CONNECTOR_UUID"),
        sample_rate=24000,
    )


def make_gemini_client() -> genai.Client:
    return genai.Client(api_key=require_env("GOOGLE_API_KEY"))


def is_business_hours(now: datetime | None = None) -> bool:
    """True on weekdays between BUSINESS_OPEN_HOUR and BUSINESS_CLOSE_HOUR (local tz)."""
    current = now or datetime.now(ZoneInfo(BUSINESS_TIMEZONE))
    if current.weekday() >= 5:
        return False
    return BUSINESS_OPEN_HOUR <= current.hour < BUSINESS_CLOSE_HOUR


def record_transcript_line(
    transcript: list[TranscriptLine],
    speaker: str,
    text: str,
) -> None:
    clean = text.strip()
    if not clean:
        return
    if transcript and transcript[-1].speaker == speaker:
        merged = f"{transcript[-1].text} {clean}".strip()
        transcript[-1] = TranscriptLine(speaker, merged)
    else:
        transcript.append(TranscriptLine(speaker, clean))
    if LOG_TRANSCRIPTS:
        logger.info("Transcript [%s]: %s", speaker, clean)


def caller_transcript_text(transcript: list[TranscriptLine]) -> str:
    lines = [line.text for line in transcript if line.speaker == "CALLER"]
    return " ".join(lines).strip()


class VoicemailStore:
    def __init__(self, path: Path = VOICEMAILS_PATH):
        self.path = path

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"voicemails": []}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def append(
        self,
        *,
        call_id: str,
        caller_number: str | None,
        callee_number: str | None,
        transcript: str,
        telegram_notified: bool,
    ) -> dict[str, Any]:
        data = self.load()
        entry = {
            "id": str(uuid.uuid4()),
            "call_id": call_id,
            "caller_number": caller_number,
            "callee_number": callee_number,
            "recorded_at": datetime.now(ZoneInfo(BUSINESS_TIMEZONE)).isoformat(),
            "transcript": transcript,
            "telegram_notified": telegram_notified,
        }
        data.setdefault("voicemails", []).append(entry)
        self.save(data)
        return entry


def send_telegram_message(text: str) -> None:
    """Notify the team via Telegram Bot API."""
    token = require_env("TELEGRAM_BOT_TOKEN")
    chat_id = require_env("TELEGRAM_CHAT_ID")
    payload = parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    req = request.Request(url, data=payload, method="POST")
    try:
        with request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Telegram API error ({exc.code}): {detail}") from exc
    if not body.get("ok"):
        raise RuntimeError(f"Telegram API returned error: {body}")


def format_telegram_voicemail_alert(
    *,
    caller_number: str | None,
    call_id: str,
    transcript: str,
    recorded_at: str,
) -> str:
    caller = caller_number or "unknown"
    message = transcript or "(no speech detected)"
    return (
        "New after-hours voicemail\n\n"
        f"From: {caller}\n"
        f"Time: {recorded_at}\n"
        f"Call ID: {call_id}\n\n"
        f"Message:\n{message}"
    )


async def route_to_callee(call: ActiveCall) -> None:
    """Business hours: pre-answer connect to original callee, then leave."""
    logger.info("Business hours — connecting call %s to callee", call.call_id)
    await call.connect()
    await call.close()
    logger.info("Call %s handed off to callee", call.call_id)


async def run_voicemail_with_gemini(
    call: ActiveCall,
    gemini_client: genai.Client,
) -> list[TranscriptLine]:
    """After hours: Gemini Live takes the voicemail and returns transcript lines."""
    transcript: list[TranscriptLine] = []
    call_ended = asyncio.Event()

    @call.on(events.CALL_TERMINATED)
    def on_terminated() -> None:
        call_ended.set()

    await call.answer()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=VOICEMAIL_SYSTEM_PROMPT,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        speech_config=types.SpeechConfig(
            language_code="en-US",
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Kore")
            ),
        ),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                silence_duration_ms=1200,
                prefix_padding_ms=300,
                end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
            ),
            activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
        ),
    )

    async with gemini_client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
        await session.send_client_content(
            turns=types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=(
                            "An after-hours call is connected. Greet the caller and ask them "
                            "to leave a voicemail message."
                        )
                    )
                ],
            ),
            turn_complete=True,
        )

        async def stream_to_gemini() -> None:
            try:
                async for chunk in call.audio_stream():
                    if call_ended.is_set():
                        break
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type=AUDIO_MIME_TYPE)
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed streaming caller audio to Gemini")

        async def receive_from_gemini() -> None:
            try:
                async for response in session.receive():
                    if call_ended.is_set():
                        break
                    content = response.server_content
                    if not content:
                        continue

                    if content.input_transcription and content.input_transcription.text:
                        record_transcript_line(
                            transcript, "CALLER", content.input_transcription.text
                        )

                    if content.output_transcription and content.output_transcription.text:
                        record_transcript_line(
                            transcript, "AGENT", content.output_transcription.text
                        )

                    if content.interrupted:
                        await call.interrupt()
                        continue

                    if content.model_turn:
                        for part in content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                await call.send_audio(part.inline_data.data)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed receiving Gemini audio responses")

        stream_task = asyncio.create_task(stream_to_gemini())
        receive_task = asyncio.create_task(receive_from_gemini())
        try:
            await asyncio.wait(
                [stream_task, receive_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in (stream_task, receive_task):
                task.cancel()
            await asyncio.gather(stream_task, receive_task, return_exceptions=True)

    return transcript


async def handle_after_hours_voicemail(
    call: ActiveCall,
    gemini_client: genai.Client,
    store: VoicemailStore,
) -> dict[str, Any]:
    """Record voicemail via Gemini Live, persist, and notify Telegram."""
    logger.info("After hours — taking voicemail for call %s", call.call_id)
    transcript_lines = await run_voicemail_with_gemini(call, gemini_client)
    caller_message = caller_transcript_text(transcript_lines)

    telegram_notified = False
    recorded_at = datetime.now(ZoneInfo(BUSINESS_TIMEZONE)).isoformat()
    try:
        send_telegram_message(
            format_telegram_voicemail_alert(
                caller_number=call.caller_number,
                call_id=call.call_id,
                transcript=caller_message,
                recorded_at=recorded_at,
            )
        )
        telegram_notified = True
        logger.info("Telegram notification sent for call %s", call.call_id)
    except Exception:
        logger.exception("Failed to send Telegram notification for call %s", call.call_id)

    entry = store.append(
        call_id=call.call_id,
        caller_number=call.caller_number,
        callee_number=call.callee_number,
        transcript=caller_message,
        telegram_notified=telegram_notified,
    )

    try:
        await call.disconnect()
    except Exception:
        logger.debug("Call %s already disconnected", call.call_id)

    return entry


async def handle_incoming_call(
    call: ActiveCall,
    gemini_client: genai.Client,
    store: VoicemailStore,
) -> None:
    now = datetime.now(ZoneInfo(BUSINESS_TIMEZONE))
    logger.info(
        "Incoming call %s from %s (local time %s, business_hours=%s)",
        call.call_id,
        call.caller_number,
        now.strftime("%Y-%m-%d %H:%M %Z"),
        is_business_hours(now),
    )

    if is_business_hours(now):
        await route_to_callee(call)
        return

    result = await handle_after_hours_voicemail(call, gemini_client, store)
    print(json.dumps({"call_id": call.call_id, "voicemail": result}, indent=2))


async def main() -> None:
    gemini_client = make_gemini_client()
    store = VoicemailStore()
    config = make_telcoflow_config()

    async with TelcoflowClient(config) as client:
        @client.on(events.INCOMING_CALL)
        async def on_call(call: ActiveCall) -> None:
            try:
                await handle_incoming_call(call, gemini_client, store)
            except Exception as exc:
                logger.exception("Call %s failed", call.call_id)
                print(f"Call {call.call_id} failed: {exc}", file=sys.stderr)
                try:
                    await call.disconnect()
                except Exception:
                    pass

        logger.info(
            "After-hours voicemail agent listening (tz=%s, hours=%s–%s)",
            BUSINESS_TIMEZONE,
            BUSINESS_OPEN_HOUR,
            BUSINESS_CLOSE_HOUR,
        )
        await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
