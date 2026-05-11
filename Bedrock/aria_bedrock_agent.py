"""
Aria — B3networks Sales Enquiry Voice Agent
=================================================================

A sales enquiry voice agent using:
  - Telcoflow SDK  →  phone call audio I/O
  - Amazon Nova Sonic 2 (Bedrock)  →  speech-to-speech AI (native)

Architecture
------------
    Caller audio  →  Telcoflow  →  Nova Sonic 2 (Bedrock)  →  Audio back to caller
                                    (speech-to-speech, native)

Nova Sonic 2 handles speech-to-speech natively via the
InvokeModelWithBidirectionalStream API.  No STT or TTS bridge is needed.
Telcoflow handles the phone line.  This script bridges them.

Audio format bridge:
    Telcoflow sends/receives 24 kHz PCM.
    Nova Sonic 2 expects 16 kHz PCM input and outputs 24 kHz PCM.
    A linear-interpolation resampler converts 24→16 kHz on the input path.

Run with:  python aria_bedrock_agent.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import uuid

from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.models import (
    InvokeModelWithBidirectionalStreamInputChunk,
    BidirectionalInputPayloadPart,
)
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

from telcoflow_sdk import TelcoflowClient, TelcoflowClientConfig, ActiveCall
import telcoflow_sdk.events as events

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aria")

# ---------------------------------------------------------------------------
# Configuration — all values from environment variables
# ---------------------------------------------------------------------------

NOVA_SONIC_MODEL = "amazon.nova-2-sonic-v1:0"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

TELCOFLOW_SAMPLE_RATE = 24000     # Telcoflow sends and receives 24 kHz PCM
NOVA_INPUT_SAMPLE_RATE = 16000    # Nova Sonic 2 expects 16 kHz input
NOVA_OUTPUT_SAMPLE_RATE = 24000   # Nova Sonic 2 produces 24 kHz output

# ---------------------------------------------------------------------------
# System prompt — tells Nova Sonic 2 who Aria is and how she should behave
# ---------------------------------------------------------------------------

ARIA_SYSTEM_PROMPT = (
    "Your name is Aria. You are a sales enquiry agent from B3networks. "
    "You are polite, warm, professional, and enthusiastic about B3networks "
    "products at all times. "
    "You greet every caller with: Hi, I am Aria from B3networks. "
    "How can I help you today? "
    "You listen carefully to what the caller is interested in. "
    "You answer questions about B3networks products, features, and pricing "
    "clearly and confidently. "
    "You encourage the caller to sign up or request a demo at the end of "
    "the conversation."
)

# ---------------------------------------------------------------------------
# Audio resampler  (24 kHz → 16 kHz via linear interpolation)
#
# Telcoflow delivers caller audio at 24 kHz.  Nova Sonic 2 expects 16 kHz.
# This function converts between the two using linear interpolation,
# producing clean downsampled PCM without any external library.
# ---------------------------------------------------------------------------


def downsample_24k_to_16k(pcm_24k: bytes) -> bytes:
    """Resample 24 kHz 16-bit mono PCM to 16 kHz via linear interpolation."""
    if len(pcm_24k) < 2:
        return pcm_24k
    samples = struct.unpack(f"<{len(pcm_24k) // 2}h", pcm_24k)
    n_out = int(len(samples) * NOVA_INPUT_SAMPLE_RATE / TELCOFLOW_SAMPLE_RATE)
    ratio = TELCOFLOW_SAMPLE_RATE / NOVA_INPUT_SAMPLE_RATE
    out = []
    for i in range(n_out):
        src = i * ratio
        idx = int(src)
        frac = src - idx
        if idx + 1 < len(samples):
            val = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
        else:
            val = samples[min(idx, len(samples) - 1)]
        out.append(max(-32768, min(32767, val)))
    return struct.pack(f"<{len(out)}h", *out)


# ===========================================================================
#
#  Nova Sonic 2 session manager
#
#  Wraps the InvokeModelWithBidirectionalStream API into a clean class
#  that the call handler drives.  All communication with Nova Sonic 2
#  uses JSON events over the bidirectional stream.
#
#  Session lifecycle (events must be sent in this order):
#    1. sessionStart    — configure inference and turn detection
#    2. promptStart     — set audio output format and voice
#    3. textInput       — send the system prompt (Aria's persona)
#    4. contentStart    — open the audio input channel
#    5. audioInput      — stream caller audio chunks (ongoing)
#    6. contentEnd      — close audio input on disconnect
#    7. promptEnd       — end the prompt
#    8. sessionEnd      — close the session
#
# ===========================================================================


class NovaSonicSession:
    """Manages a single Nova Sonic 2 bidirectional stream session."""

    def __init__(
        self,
        region: str = AWS_REGION,
        model_id: str = NOVA_SONIC_MODEL,
    ):
        self.region = region
        self.model_id = model_id
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.stream = None
        self.is_active = False

    def _build_client(self) -> BedrockRuntimeClient:
        """Create a Bedrock Runtime client using environment credentials.

        Requires AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to be set.
        """
        config = Config(
            endpoint_uri=(
                f"https://bedrock-runtime.{self.region}.amazonaws.com"
            ),
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        return BedrockRuntimeClient(config=config)

    async def send_event(self, payload: dict | str):
        """Send a JSON event to the bidirectional stream."""
        raw = json.dumps(payload) if isinstance(payload, dict) else payload
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(
                bytes_=raw.encode("utf-8")
            )
        )
        await self.stream.input_stream.send(event)

    # ---- Session initialisation -------------------------------------------

    async def open(self, system_prompt: str):
        """Open the stream and send the full initialisation sequence.

        Sends, in order:
          1. sessionStart   — inference config + turn detection
          2. promptStart    — audio output config (24 kHz lpcm, 16-bit, mono)
          3. system prompt  — TEXT content block with role=SYSTEM
          4. audio input    — opens the AUDIO content block for caller audio
        """
        client = self._build_client()
        self.stream = (
            await client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(
                    model_id=self.model_id
                )
            )
        )
        self.is_active = True

        # 1) sessionStart — inference and turn detection configuration
        await self.send_event({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7,
                    },
                    "turnDetectionConfiguration": {
                        "endpointingSensitivity": "HIGH",
                    },
                }
            }
        })

        # 2) promptStart — audio output format and voice selection
        await self.send_event({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {
                        "mediaType": "text/plain",
                    },
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": NOVA_OUTPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "tiffany",
                        "encoding": "base64",
                        "audioType": "SPEECH",
                    },
                }
            }
        })

        # 3) System prompt — sent as a TEXT content block with role SYSTEM
        await self.send_event({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "SYSTEM",
                    "textInputConfiguration": {
                        "mediaType": "text/plain",
                    },
                }
            }
        })
        await self.send_event({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "content": system_prompt,
                }
            }
        })
        await self.send_event({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                }
            }
        })

        # 4) Open the audio input content block (16 kHz lpcm, 16-bit, mono)
        await self.send_event({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": NOVA_INPUT_SAMPLE_RATE,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64",
                    },
                }
            }
        })

    # ---- Audio streaming --------------------------------------------------

    async def send_audio(self, pcm_16k: bytes):
        """Send a chunk of 16 kHz PCM audio to Nova Sonic 2.

        The audio is base64-encoded before sending as required by the
        Nova Sonic 2 audioInput event format.
        """
        b64 = base64.b64encode(pcm_16k).decode("utf-8")
        await self.send_event({
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": b64,
                }
            }
        })

    # ---- Receiving responses ----------------------------------------------

    async def receive(self):
        """Yield parsed JSON events from the Nova Sonic 2 output stream.

        Runs until the session ends or an unrecoverable error occurs.
        Each yielded item is a dict containing an "event" key with the
        response event (audioOutput, textOutput, contentStart, etc.).
        """
        while self.is_active:
            try:
                output = await self.stream.await_output()
                result = await output[1].receive()
                if result.value and result.value.bytes_:
                    data = json.loads(
                        result.value.bytes_.decode("utf-8")
                    )
                    yield data
            except StopAsyncIteration:
                break
            except Exception as exc:
                if "ValidationException" in str(exc):
                    log.error("Validation error: %s", exc)
                else:
                    log.error("Stream receive error: %s", exc)
                break
        self.is_active = False

    # ---- Teardown ---------------------------------------------------------

    async def close(self):
        """Close the audio input, prompt, and session in the correct order.

        Sends contentEnd → promptEnd → sessionEnd, then closes the stream.
        Safe to call multiple times.
        """
        if not self.is_active:
            return
        self.is_active = False
        try:
            await self.send_event({
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                    }
                }
            })
            await self.send_event({
                "event": {
                    "promptEnd": {"promptName": self.prompt_name},
                }
            })
            await self.send_event({"event": {"sessionEnd": {}}})
            await self.stream.input_stream.close()
        except Exception:
            pass


# ===========================================================================
#
#  Call handler — bridges Telcoflow and Nova Sonic 2
#
#  Audio flow:
#    Telcoflow  →  24 kHz PCM  →  downsample to 16 kHz  →  Nova Sonic 2
#    Nova Sonic 2  →  24 kHz PCM  →  Telcoflow  →  caller
#
#  Two concurrent coroutines run for each call:
#    stream_to_nova    — caller audio → resample → Nova Sonic 2
#    receive_from_nova — Nova Sonic 2 events → caller audio + logging
#
#  Interruption handling:
#    Nova Sonic 2 sends a textOutput with '{ "interrupted" : true }'
#    when it detects the caller speaking over Aria.  On receiving this,
#    we call call.clear_send_audio_buffer() to stop Aria's audio
#    immediately so the caller can be heard.
#
# ===========================================================================


async def handle_aria_call(call: ActiveCall) -> None:
    """Handle one incoming phone call from start to finish.

    1. Answer the call and open the audio channel.
    2. Open a Nova Sonic 2 session with Aria's system prompt.
    3. Stream caller audio to Nova Sonic 2 (with 24→16 kHz resampling).
    4. Receive Nova Sonic 2 audio and send it back to the caller.
    5. Handle interruptions when the caller speaks over Aria.
    """
    call_id = call.call_id
    clog = log.getChild(call_id[:8])

    # -- Step 1: Answer the call -------------------------------------------
    await call.answer()
    clog.info("Call answered — starting Aria session")

    # -- Step 2: Open a Nova Sonic 2 session with Aria's persona -----------
    sonic = NovaSonicSession()
    await sonic.open(ARIA_SYSTEM_PROMPT)

    # -- Step 3: Stream caller audio → Nova Sonic 2 (24→16 kHz) -----------
    async def stream_to_nova():
        """Read PCM chunks from the phone line, downsample 24→16 kHz,
        and forward to Nova Sonic 2 as base64-encoded audioInput events.

        call.audio_stream() is an async iterator that yields raw PCM
        bytes at TELCOFLOW_SAMPLE_RATE (24 kHz).  Each chunk is
        resampled to 16 kHz and sent to Nova Sonic 2.
        """
        try:
            async for chunk in call.audio_stream():
                pcm_16k = downsample_24k_to_16k(chunk)
                await sonic.send_audio(pcm_16k)
        finally:
            await sonic.close()

    # -- Step 4: Receive Nova Sonic 2 events → caller audio ----------------
    async def receive_from_nova():
        """Handle all events from the Nova Sonic 2 output stream.

        Key event types handled:

          audioOutput   — decoded PCM audio sent to the caller via
                          call.send_audio().  This is Aria speaking.

          textOutput    — text transcripts from Nova Sonic 2.  These
                          include caller transcription (role=USER),
                          Aria's response text (role=ASSISTANT), and
                          the interrupt signal '{ "interrupted" : true }'.

          contentStart  — tracks the current content role so we know
                          whether text belongs to the caller or Aria.

          completionEnd — marks the end of a full Aria response turn.
                          Logged for observability.
        """
        current_role = None

        async for data in sonic.receive():
            ev = data.get("event", {})

            # Track which role is producing content
            if "contentStart" in ev:
                current_role = ev["contentStart"].get("role")

            # Audio output from Aria → forward to caller
            if "audioOutput" in ev:
                audio_bytes = base64.b64decode(
                    ev["audioOutput"]["content"]
                )
                await call.send_audio(audio_bytes)

            # Text output — transcripts and interrupt detection
            if "textOutput" in ev:
                text = ev["textOutput"].get("content", "")

                # Nova Sonic 2 sends this exact string when the caller
                # speaks over Aria (barge-in).  Clear the audio buffer
                # so the caller can be heard immediately.
                if '{ "interrupted" : true }' in text:
                    await call.clear_send_audio_buffer()
                    clog.info("Interrupted — audio buffer cleared")
                elif current_role == "ASSISTANT":
                    clog.debug("Aria: %s", text[:120])
                elif current_role == "USER":
                    clog.info("Caller: %s", text[:120])

            # Aria completed a full response turn
            if "completionEnd" in ev:
                clog.info("Aria completed a response turn")

    # -- Run both directions concurrently for the call's lifetime ----------
    try:
        await asyncio.gather(stream_to_nova(), receive_from_nova())
    finally:
        await sonic.close()

    clog.info("Aria session complete")


# ===========================================================================
#  Entry point
# ===========================================================================


async def main() -> None:
    """Start the Aria sales enquiry voice agent.

    1. Connect to Telcoflow in sandbox mode.
    2. Register the incoming-call handler.
    3. Wait for calls forever.
    """
    tf_config = TelcoflowClientConfig.sandbox(
        api_key=os.environ["WSS_API_KEY"],
        connector_uuid=os.environ["WSS_CONNECTOR_UUID"],
        sample_rate=TELCOFLOW_SAMPLE_RATE,
    )

    async with TelcoflowClient(tf_config) as client:

        @client.on(events.INCOMING_CALL)
        async def on_incoming_call(call: ActiveCall):
            """Each incoming call is handled in its own coroutine."""
            cid = call.call_id[:8]
            log.info("[%s] Incoming call", cid)
            try:
                await handle_aria_call(call)
            except Exception:
                log.exception("[%s] Call handler failed", cid)

        log.info(
            "Aria (B3networks · Nova Sonic 2) is live — waiting for calls …"
        )
        await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
