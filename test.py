"""
Plain Google GenAI SDK integration (no ADK).
Receives incoming calls and bridges audio with Gemini Live.
"""

import asyncio
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from telcoflow_sdk import TelcoflowClient, TelcoflowClientConfig, ActiveCall
import telcoflow_sdk.events as events
from telcoflow_sdk.exceptions import BufferFullError

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
AUDIO_MIME = "audio/pcm;rate=24000"


class GeminiCallSession:
    def __init__(self, call: ActiveCall):
        self._call = call
        self._session = None
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._terminated = False

    async def _on_terminated(self):
        logger.info(f"Call {self._call.call_id} terminated")
        self._terminated = True
        if self._send_task:
            self._send_task.cancel()
        if self._recv_task:
            self._recv_task.cancel()

    async def run(self):
        self._call.register_event_handler(events.CALL_TERMINATED, self._on_terminated)

        async with gemini_client.aio.live.connect(
            model=MODEL,
            config=types.LiveConnectConfig(
                response_modalities=["AUDIO"],
            ),
        ) as session:
            self._session = session

            self._send_task = asyncio.create_task(self._stream_to_gemini())
            self._recv_task = asyncio.create_task(self._receive_from_gemini())

            try:
                await asyncio.gather(self._send_task, self._recv_task)
            except asyncio.CancelledError:
                logger.debug("Gemini session tasks cancelled")

        logger.info(f"Call {self._call.call_id} session ended")

    async def _stream_to_gemini(self):
        try:
            async for audio_chunk in self._call.audio_stream():
                if self._terminated:
                    break
                await self._session.send_realtime_input(
                    audio=types.Blob(data=audio_chunk, mime_type=AUDIO_MIME)
                )
        except Exception:
            if not self._terminated:
                logger.exception("Error streaming to Gemini")
            raise
        finally:
            logger.debug("Stream to Gemini completed")

    async def _receive_from_gemini(self):
        try:
            while not self._terminated:
                async for response in self._session.receive():
                    if self._terminated:
                        break
                    if content := response.server_content:
                        if content.interrupted:
                            await self._call.clear_send_audio_buffer()

                        if content.model_turn:
                            for part in content.model_turn.parts:
                                if part.inline_data:
                                    try:
                                        await self._call.send_audio(part.inline_data.data)
                                    except BufferFullError:
                                        await self._call.clear_send_audio_buffer()

                        if content.turn_complete:
                            break
        except Exception:
            if not self._terminated:
                logger.exception("Error receiving from Gemini")
            raise
        finally:
            logger.debug("Receive from Gemini completed")


async def main():
    config = TelcoflowClientConfig.sandbox(
        sample_rate=24000,
        api_key=os.getenv("WSS_API_KEY"),
        connector_uuid=os.getenv("WSS_CONNECTOR_UUID"),
    )

    async with TelcoflowClient(config) as client:
        logger.info("Connected to Telcoflow. Waiting for calls...")

        @client.on(events.INCOMING_CALL)
        async def on_call(call: ActiveCall):
            logger.info(f"Incoming call from {call.caller_number} (ID: {call.call_id})")
            try:
                await call.answer()
                session = GeminiCallSession(call)
                await session.run()
            except Exception:
                logger.exception(f"Error handling call {call.call_id}")
                try:
                    await call.hangup()
                except Exception:
                    pass

        await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())