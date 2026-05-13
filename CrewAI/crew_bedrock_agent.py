"""
Crew — B3networks Call Triage (CrewAI + Nova Sonic 2 + Telcoflow)
================================================================

Same triage pipeline as `crew_agent.py` (Receptionist → Analyst → Resolver),
but the speech layer is **Amazon Nova Sonic 2** (Bedrock bidirectional stream)
instead of Gemini Live.

Audio bridge (matches `LangGraph/nova_bedrock_agent.py`):
    Telcoflow 24 kHz PCM  →  resample to 16 kHz  →  Nova input
    Nova audio output 24 kHz  →  Telcoflow

CrewAI still uses **Gemini 2.0 Flash** via `crewai.LLM` (`GOOGLE_API_KEY`).
Nova Sonic uses **IAM** credentials (`AWS_*`) — not Bedrock API keys for streaming.

Run with:  python crew_bedrock_agent.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import struct
import uuid

from dotenv import load_dotenv

load_dotenv()

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

from crew_gemini_agent import build_triage_crew

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-18s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crew.bedrock")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NOVA_SONIC_MODEL = "amazon.nova-2-sonic-v1:0"
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
TELCOFLOW_SAMPLE_RATE = 24000
NOVA_INPUT_SAMPLE_RATE = 16000
NOVA_OUTPUT_SAMPLE_RATE = 24000

CREW_SONIC_SYSTEM_PROMPT = """\
Your name is Crew. You are a customer care agent from B3networks.
You are polite, warm, and professional at all times.
Listen carefully to the caller and transcribe their message accurately.

When you receive a text message from the user, it contains the exact response you must
speak to the caller. Say those words naturally in your warm, professional
voice. Do not add commentary or change the message.
"""

GREETING = "Hi, I am Crew from B3networks. How can I help you today?"


def downsample_24k_to_16k(pcm_24k: bytes) -> bytes:
    if len(pcm_24k) < 2:
        return pcm_24k
    samples = struct.unpack(f"<{len(pcm_24k) // 2}h", pcm_24k)
    n_out = int(len(samples) * NOVA_INPUT_SAMPLE_RATE / TELCOFLOW_SAMPLE_RATE)
    ratio = TELCOFLOW_SAMPLE_RATE / NOVA_INPUT_SAMPLE_RATE
    out: list[int] = []
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


class CrewNovaSonicSession:
    """Nova Sonic 2 bidirectional stream — audio I/O only (no Bedrock tools)."""

    def __init__(self, region: str = AWS_REGION, model_id: str = NOVA_SONIC_MODEL):
        self.region = region
        self.model_id = model_id
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.stream = None
        self.is_active = False
        self._closed = False

    def _build_client(self) -> BedrockRuntimeClient:
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        return BedrockRuntimeClient(config=config)

    async def send_event(self, payload: dict | str):
        raw = json.dumps(payload) if isinstance(payload, dict) else payload
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=raw.encode("utf-8"))
        )
        await self.stream.input_stream.send(event)

    async def open(self, system_prompt: str):
        client = self._build_client()
        self.stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True

        await self.send_event({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7,
                    },
                    "turnDetectionConfiguration": {"endpointingSensitivity": "HIGH"},
                }
            }
        })

        await self.send_event({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
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

        await self.send_event({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.content_name,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "SYSTEM",
                    "textInputConfiguration": {"mediaType": "text/plain"},
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

    async def send_audio(self, pcm_16k: bytes):
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

    async def send_user_text(self, text: str):
        """Send user-role text to trigger Nova to speak it (per system instructions)."""
        content_name = str(uuid.uuid4())
        await self.send_event({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "type": "TEXT",
                    "interactive": True,
                    "role": "USER",
                    "textInputConfiguration": {"mediaType": "text/plain"},
                }
            }
        })
        await self.send_event({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": text,
                }
            }
        })
        await self.send_event({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                }
            }
        })

    async def receive(self):
        while self.is_active:
            try:
                output = await self.stream.await_output()
                result = await output[1].receive()
                if result.value and result.value.bytes_:
                    yield json.loads(result.value.bytes_.decode("utf-8"))
            except StopAsyncIteration:
                break
            except Exception as exc:
                log.error("Stream receive error: %s", exc)
                break
        self.is_active = False

    async def close(self):
        if self._closed or self.stream is None:
            return
        self._closed = True
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
            await self.send_event({"event": {"promptEnd": {"promptName": self.prompt_name}}})
            await self.send_event({"event": {"sessionEnd": {}}})
            await self.stream.input_stream.close()
        except Exception:
            pass


call_histories: dict[str, list[str]] = {}


async def handle_crew_bedrock_call(call: ActiveCall) -> None:
    call_id = call.call_id
    call_histories[call_id] = []
    clog = log.getChild(call_id[:8])

    await call.answer()
    clog.info("Call answered — Crew triage + Nova Sonic 2")

    sonic = CrewNovaSonicSession()
    await sonic.open(CREW_SONIC_SYSTEM_PROMPT)

    speaking_phase = True
    call_histories[call_id].append(f"Crew: {GREETING}")
    await sonic.send_user_text(GREETING)

    async def stream_to_nova():
        try:
            async for chunk in call.audio_stream():
                pcm_16k = downsample_24k_to_16k(chunk)
                await sonic.send_audio(pcm_16k)
        finally:
            await sonic.close()

    async def receive_from_nova():
        nonlocal speaking_phase
        text_buffer: list[str] = []
        current_role: str | None = None

        async for data in sonic.receive():
            ev = data.get("event", {})

            if "contentStart" in ev:
                current_role = ev["contentStart"].get("role")

            if "audioOutput" in ev:
                if speaking_phase:
                    raw = ev["audioOutput"]["content"]
                    audio_bytes = base64.b64decode(raw)
                    await call.send_audio(audio_bytes)

            if "textOutput" in ev:
                text = ev["textOutput"].get("content", "")
                if '{ "interrupted" : true }' in text:
                    await call.clear_send_audio_buffer()
                    text_buffer.clear()
                    clog.info("Interrupted — audio buffer cleared")
                elif current_role == "USER" and not speaking_phase:
                    text_buffer.append(text)

            if "completionEnd" in ev:
                transcript = "".join(text_buffer).strip()
                text_buffer.clear()

                if speaking_phase:
                    speaking_phase = False
                    clog.info("Crew finished speaking — listening")
                else:
                    caller_text = transcript or "(inaudible)"
                    call_histories[call_id].append(f"Caller: {caller_text}")
                    clog.info("Caller: %s", caller_text)

                    history_text = "\n".join(call_histories[call_id])
                    crew = build_triage_crew(
                        caller_transcript=caller_text,
                        conversation_history=history_text,
                        call_id=call_id,
                    )
                    clog.info("Running triage: Receptionist → Analyst → Resolver")
                    try:
                        result = await asyncio.to_thread(crew.kickoff)
                        resolver_response = result.raw.strip()
                    except Exception as exc:
                        clog.error("CrewAI triage failed: %s", exc)
                        resolver_response = (
                            "I apologise, I'm having a little trouble processing that. "
                            "Could you please repeat your question?"
                        )

                    call_histories[call_id].append(f"Crew: {resolver_response}")
                    clog.info("Resolver: %s", resolver_response)

                    speaking_phase = True
                    await sonic.send_user_text(resolver_response)

    try:
        await asyncio.gather(stream_to_nova(), receive_from_nova())
    finally:
        await sonic.close()

    call_histories.pop(call_id, None)
    clog.info("Crew + Nova session complete")


async def main() -> None:
    tf_config = TelcoflowClientConfig.sandbox(
        api_key=os.environ["WSS_API_KEY"],
        connector_uuid=os.environ["WSS_CONNECTOR_UUID"],
        sample_rate=TELCOFLOW_SAMPLE_RATE,
    )

    async with TelcoflowClient(tf_config) as client:

        @client.on(events.INCOMING_CALL)
        async def on_incoming_call(call: ActiveCall):
            cid = call.call_id[:8]
            log.info("[%s] Incoming call", cid)
            try:
                await handle_crew_bedrock_call(call)
            except Exception:
                log.exception("[%s] Call handler failed", cid)

        log.info(
            "Crew (B3networks · Nova Sonic 2 + CrewAI) is live — waiting for calls …"
        )
        await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
