"""Google ADK bidirectional voice bridge for Telcoflow phone calls."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from google.adk.agents import Agent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.genai.errors import APIError
from telcoflow_sdk import ActiveCall
import telcoflow_sdk.events as events

from availability_inject import (
    build_calendar_system_message,
    parse_appointment_slot,
    patient_time_is_complete,
)
from config import ADK_MODEL, APP_NAME, AUDIO_MIME_TYPE
from prompts import INITIAL_TURN_TEXT, MAYA_BOOKING_INSTRUCTION
from transcript import TranscriptLine, record_transcript_line

logger = logging.getLogger(__name__)

session_service = InMemorySessionService()


def create_voice_agent(
    instruction: str,
    tools: Sequence | None = None,
) -> tuple[Agent, Runner]:
    agent = Agent(
        name="maya_voice_agent",
        model=ADK_MODEL,
        instruction=instruction,
        tools=list(tools or []),
    )
    runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)
    return agent, runner


class ADKVoiceSession:
    """Bridge one Telcoflow call to one ADK live session."""

    def __init__(
        self,
        call: ActiveCall,
        runner: Runner,
        instruction: str,
        user_id: str = "caller",
    ):
        self.call = call
        self.runner = runner
        self.instruction = instruction
        self.user_id = user_id
        self.session_id = call.call_id
        self.live_request_queue = LiveRequestQueue()
        self.transcript: list[TranscriptLine] = []
        self._terminated = False
        self._call_ended = asyncio.Event()
        self._checked_slots: set[str] = set()
        self._availability_task: asyncio.Task | None = None
        self._last_injected_slot: tuple[str, str] | None = None

    async def run(self) -> list[TranscriptLine]:
        @self.call.on(events.CALL_TERMINATED)
        def on_terminated() -> None:
            self._terminated = True
            self._call_ended.set()

        await self.call.answer()

        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=self.user_id,
            session_id=self.session_id,
        )
        if not session:
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=self.user_id,
                session_id=self.session_id,
            )

        tasks = [
            asyncio.create_task(self._stream_to_adk()),
            asyncio.create_task(self._receive_from_adk()),
            asyncio.create_task(self._wait_for_call_end()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            try:
                task.result()
            except Exception as exc:
                if self.transcript:
                    logger.warning(
                        "Live voice ended early for call %s (%s); using partial transcript",
                        self.call.call_id,
                        exc,
                    )
                else:
                    raise

        return self.transcript

    async def _stream_to_adk(self) -> None:
        self.live_request_queue.send_content(
            types.Content(parts=[types.Part(text=INITIAL_TURN_TEXT)])
        )

        async for audio_chunk in self.call.audio_stream():
            if self._terminated:
                break
            self.live_request_queue.send_realtime(
                types.Blob(data=audio_chunk, mime_type=AUDIO_MIME_TYPE)
            )

    async def _receive_from_adk(self) -> None:
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

        try:
            async for event in self.runner.run_live(
                user_id=self.user_id,
                session_id=self.session_id,
                live_request_queue=self.live_request_queue,
                run_config=run_config,
            ):
                if self._terminated:
                    break

                if getattr(event, "interrupted", False):
                    if hasattr(self.call, "interrupt"):
                        await self.call.interrupt()
                    else:
                        await self.call.clear_send_audio_buffer()
                    continue

                input_transcription = getattr(event, "input_transcription", None)
                if input_transcription and getattr(input_transcription, "text", None):
                    record_transcript_line(self.transcript, "PATIENT", input_transcription.text)
                    self._schedule_availability_check()

                output_transcription = getattr(event, "output_transcription", None)
                if output_transcription and getattr(output_transcription, "text", None):
                    record_transcript_line(self.transcript, "MAYA", output_transcription.text)

                content = getattr(event, "content", None)
                if content and content.parts:
                    for part in content.parts:
                        inline_data = getattr(part, "inline_data", None)
                        if inline_data and inline_data.data:
                            await self.call.send_audio(inline_data.data)
        except APIError as exc:
            logger.error(
                "Gemini live session failed for call %s: %s",
                self.call.call_id,
                exc,
            )
            self._terminated = True
            self._call_ended.set()

    def _schedule_availability_check(self) -> None:
        if self._availability_task and not self._availability_task.done():
            self._availability_task.cancel()
        self._availability_task = asyncio.create_task(self._inject_availability_if_ready())

    async def _inject_availability_if_ready(self) -> None:
        # Wait for the patient to finish saying the time (e.g. "4 pm", not just "4").
        await asyncio.sleep(3.0)
        if self._terminated:
            return
        if not patient_time_is_complete(self.transcript):
            return

        slot = await asyncio.to_thread(parse_appointment_slot, self.transcript)
        if slot is None:
            return

        # Require the same slot twice so partial speech does not trigger a wrong check.
        await asyncio.sleep(1.5)
        if self._terminated:
            return
        stable_slot = await asyncio.to_thread(parse_appointment_slot, self.transcript)
        if stable_slot != slot:
            logger.info(
                "Skipping calendar inject for call %s; slot still changing (%s -> %s)",
                self.call.call_id,
                slot,
                stable_slot,
            )
            return

        slot_key = f"{slot[0]}_{slot[1]}"
        if slot_key in self._checked_slots:
            return

        is_update = self._last_injected_slot is not None and self._last_injected_slot != slot
        message = await asyncio.to_thread(
            build_calendar_system_message,
            self.transcript,
            is_update=is_update,
        )
        if not message:
            return

        self._checked_slots.add(slot_key)
        self._last_injected_slot = slot
        logger.info("Injecting calendar availability for call %s: %s %s", self.call.call_id, slot[0], slot[1])
        self.live_request_queue.send_content(types.Content(parts=[types.Part(text=message)]))

    async def _wait_for_call_end(self) -> None:
        await self._call_ended.wait()


async def run_adk_voice_call(
    call: ActiveCall,
    instruction: str = MAYA_BOOKING_INSTRUCTION,
    tools: Sequence | None = None,
) -> list[TranscriptLine]:
    # Calendar checks are injected as [Calendar system] text messages to avoid
    # Gemini Live API 1011 crashes from native-audio function tools.
    _, runner = create_voice_agent(instruction=instruction, tools=tools or [])
    session = ADKVoiceSession(call=call, runner=runner, instruction=instruction)
    return await session.run()
