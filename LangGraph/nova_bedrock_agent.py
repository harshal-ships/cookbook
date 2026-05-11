"""
Nova — B3networks Customer Care Voice Agent (Nova Sonic 2 + LangGraph)
======================================================================

Architecture:
    Caller audio  →  Telcoflow  →  Amazon Nova Sonic 2  →  Audio back to caller
                                            ↕
                                     LangGraph StateGraph
                                (conversation state per call)

Nova Sonic 2 handles speech-to-speech.  LangGraph manages per-call state,
conversation history, and tool execution.  Telcoflow handles the phone line.

Audio format bridge:
    Telcoflow sends/receives 24 kHz PCM.
    Nova Sonic 2 expects 16 kHz PCM input and outputs 24 kHz PCM.
    A linear-interpolation resampler converts 24→16 kHz on the input path.

Run with:  python nova_bedrock_agent.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import operator
import os
import random
import struct
import uuid
from typing import Annotated, Literal

from typing_extensions import TypedDict

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

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

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
log = logging.getLogger("nova")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NOVA_SONIC_MODEL = "amazon.nova-2-sonic-v1:0"
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
TELCOFLOW_SAMPLE_RATE = 24000
NOVA_INPUT_SAMPLE_RATE = 16000
NOVA_OUTPUT_SAMPLE_RATE = 24000

NOVA_SYSTEM_PROMPT = (
    "Your name is Nova. You are a customer care agent from B3networks. "
    "You are polite, warm, and professional at all times. "
    "You greet every caller with: Hi, I am Nova from B3networks. How can I help you today? "
    "You listen carefully and answer questions about B3networks products and services. "
    "You try your best to resolve the customer issue in the same call. "
    "When you need information about a product, use the getProductInfo tool. "
    "When a caller asks about service status, use the checkServiceStatus tool. "
    "When you cannot fully resolve an issue, use the createSupportTicket tool."
)

# ---------------------------------------------------------------------------
# Audio resampler  (24 kHz → 16 kHz via linear interpolation)
# ---------------------------------------------------------------------------


def downsample_24k_to_16k(pcm_24k: bytes) -> bytes:
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


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

PRODUCT_CATALOG = {
    "unified communications": {
        "name": "Unified Communications",
        "description": "Cloud-based phone system with voice, video, messaging, and collaboration.",
        "pricing": "Starting at $15 per user per month",
    },
    "contact center": {
        "name": "Contact Center",
        "description": "Omnichannel contact center with IVR, intelligent routing, and analytics.",
        "pricing": "Starting at $50 per agent per month",
    },
    "cpaas": {
        "name": "CPaaS",
        "description": "Programmable communication APIs for voice, SMS, and video.",
        "pricing": "Pay-as-you-go based on usage",
    },
    "sip trunking": {
        "name": "SIP Trunking",
        "description": "Enterprise-grade SIP trunking for connecting your PBX to the PSTN.",
        "pricing": "Starting at $0.01 per minute",
    },
}


def get_product_info(product_name: str) -> dict:
    query = product_name.lower().strip()
    for key, info in PRODUCT_CATALOG.items():
        if query in key or query in info["name"].lower():
            return info
    return {
        "error": f"No product found matching '{product_name}'.",
        "available_products": [v["name"] for v in PRODUCT_CATALOG.values()],
    }


def check_service_status() -> dict:
    return {
        "overall_status": "operational",
        "message": "All B3networks services are running normally.",
    }


def create_support_ticket(issue_description: str) -> dict:
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    return {
        "ticket_id": ticket_id,
        "status": "created",
        "message": f"Support ticket {ticket_id} created. Follow-up within 24 hours.",
    }


TOOL_REGISTRY: dict[str, callable] = {
    "getProductInfo": get_product_info,
    "checkServiceStatus": check_service_status,
    "createSupportTicket": create_support_ticket,
}

# ---------------------------------------------------------------------------
# Nova Sonic tool schemas (included in the promptStart event)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "toolSpec": {
            "name": "getProductInfo",
            "description": "Look up B3networks product information by name or keyword.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "product_name": {
                            "type": "string",
                            "description": "Product name or keyword",
                        }
                    },
                    "required": ["product_name"],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "checkServiceStatus",
            "description": "Check current operational status of B3networks services.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }
            },
        }
    },
    {
        "toolSpec": {
            "name": "createSupportTicket",
            "description": "Create a support ticket for unresolved issues.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "issue_description": {
                            "type": "string",
                            "description": "Description of the customer issue",
                        }
                    },
                    "required": ["issue_description"],
                }
            },
        }
    },
]

# ===========================================================================
#  LangGraph — per-call state (identical role to the Gemini version)
# ===========================================================================


class CallState(TypedDict):
    call_id: str
    conversation_history: Annotated[list[dict], operator.add]
    turn_count: int
    tool_results: Annotated[list[dict], operator.add]
    status: str  # "active" | "ending"


def process_event(state: CallState) -> dict:
    return {"turn_count": state.get("turn_count", 0) + 1}


def execute_tool(state: CallState) -> dict:
    history = state.get("conversation_history", [])
    if not history or history[-1].get("type") != "tool_call":
        return {}

    last = history[-1]
    func_name = last.get("function_name", "")
    func_args = last.get("arguments", {})
    func = TOOL_REGISTRY.get(func_name)

    if func is None:
        result = {"error": f"Unknown tool: {func_name}"}
    else:
        try:
            result = func(**func_args)
        except Exception as exc:
            result = {"error": str(exc)}

    return {"tool_results": [{"function_name": func_name, "arguments": func_args, "result": result}]}


def assess_call(state: CallState) -> dict:
    history = state.get("conversation_history", [])
    if not history:
        return {"status": "active"}
    last = history[-1]
    if last.get("type") == "nova_turn" and last.get("text"):
        text = last["text"].lower()
        if any(s in text for s in ["goodbye", "bye", "have a great day", "take care"]):
            return {"status": "ending"}
    return {"status": "active"}


def route_event(state: CallState) -> Literal["execute_tool", "assess_call"]:
    history = state.get("conversation_history", [])
    if history and history[-1].get("type") == "tool_call":
        return "execute_tool"
    return "assess_call"


def build_call_graph():
    builder = StateGraph(CallState)
    builder.add_node("process_event", process_event)
    builder.add_node("execute_tool", execute_tool)
    builder.add_node("assess_call", assess_call)
    builder.add_edge(START, "process_event")
    builder.add_conditional_edges("process_event", route_event)
    builder.add_edge("execute_tool", "assess_call")
    builder.add_edge("assess_call", END)
    return builder.compile(checkpointer=MemorySaver())


# ===========================================================================
#  Nova Sonic 2 session manager
#
#  Wraps the low-level bidirectional stream protocol into a class that
#  the call handler can drive.  Sends/receives JSON events over the
#  Bedrock streaming API.
# ===========================================================================


class NovaSonicSession:
    """Manages a single Nova Sonic 2 bidirectional stream session."""

    def __init__(self, region: str = AWS_REGION, model_id: str = NOVA_SONIC_MODEL):
        self.region = region
        self.model_id = model_id
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.stream = None
        self.is_active = False

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
        """Open the stream and send the full initialisation sequence."""
        client = self._build_client()
        self.stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        self.is_active = True

        # 1) sessionStart
        await self.send_event({
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7},
                    "turnDetectionConfiguration": {"endpointingSensitivity": "HIGH"},
                }
            }
        })

        # 2) promptStart with audio output config + tools
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
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "toolConfiguration": {"tools": TOOL_SCHEMAS},
                }
            }
        })

        # 3) System prompt (TEXT content block)
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
            "event": {"contentEnd": {"promptName": self.prompt_name, "contentName": self.content_name}}
        })

        # 4) Open the audio input content block
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

    async def send_tool_result(self, tool_use_id: str, result: dict):
        content_name = str(uuid.uuid4())
        await self.send_event({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "interactive": False,
                    "type": "TOOL",
                    "role": "TOOL",
                    "toolResultInputConfiguration": {
                        "toolUseId": tool_use_id,
                        "type": "TEXT",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    },
                }
            }
        })
        await self.send_event({
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": json.dumps(result),
                }
            }
        })
        await self.send_event({
            "event": {"contentEnd": {"promptName": self.prompt_name, "contentName": content_name}}
        })

    async def receive(self):
        """Yield parsed JSON events from the stream."""
        while self.is_active:
            try:
                output = await self.stream.await_output()
                result = await output[1].receive()
                if result.value and result.value.bytes_:
                    data = json.loads(result.value.bytes_.decode("utf-8"))
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

    async def close(self):
        if not self.is_active:
            return
        self.is_active = False
        try:
            await self.send_event({
                "event": {"contentEnd": {"promptName": self.prompt_name, "contentName": self.audio_content_name}}
            })
            await self.send_event({"event": {"promptEnd": {"promptName": self.prompt_name}}})
            await self.send_event({"event": {"sessionEnd": {}}})
            await self.stream.input_stream.close()
        except Exception:
            pass


# ===========================================================================
#  Call handler — bridges Telcoflow, Nova Sonic 2, and LangGraph
# ===========================================================================


async def handle_nova_call(call: ActiveCall, graph) -> None:
    call_id = call.call_id
    thread_cfg = {"configurable": {"thread_id": call_id}}
    clog = log.getChild(call_id[:8])

    await call.answer()
    clog.info("Call answered")

    await graph.ainvoke(
        {
            "call_id": call_id,
            "conversation_history": [{"type": "call_started", "call_id": call_id}],
            "turn_count": 0,
            "tool_results": [],
            "status": "active",
        },
        config=thread_cfg,
    )

    sonic = NovaSonicSession()
    await sonic.open(NOVA_SYSTEM_PROMPT)

    # --- Caller audio → Nova Sonic 2 (with 24→16 kHz resampling) ----------
    async def stream_to_nova():
        try:
            async for chunk in call.audio_stream():
                pcm_16k = downsample_24k_to_16k(chunk)
                await sonic.send_audio(pcm_16k)
        finally:
            await sonic.close()

    # --- Nova Sonic 2 events → caller audio + LangGraph --------------------
    async def receive_from_nova():
        nova_text_buffer: list[str] = []
        current_role = None
        pending_tool_name = ""
        pending_tool_id = ""
        pending_tool_content: dict = {}

        async for data in sonic.receive():
            ev = data.get("event", {})

            # Track content role
            if "contentStart" in ev:
                current_role = ev["contentStart"].get("role")

            # Audio output → forward to caller
            if "audioOutput" in ev:
                audio_bytes = base64.b64decode(ev["audioOutput"]["content"])
                await call.send_audio(audio_bytes)

            # Text output → capture transcript / detect interruption
            if "textOutput" in ev:
                text = ev["textOutput"].get("content", "")

                if '{ "interrupted" : true }' in text:
                    await call.clear_send_audio_buffer()
                    nova_text_buffer.clear()
                    clog.info("Interrupted — buffer cleared")
                elif current_role == "ASSISTANT":
                    nova_text_buffer.append(text)
                elif current_role == "USER":
                    clog.info("Caller: %s", text[:80])

            # Tool use request from Nova Sonic
            if "toolUse" in ev:
                pending_tool_name = ev["toolUse"]["toolName"]
                pending_tool_id = ev["toolUse"]["toolUseId"]
                pending_tool_content = ev["toolUse"]

            # contentEnd with type=TOOL means execute the pending tool
            if "contentEnd" in ev and ev["contentEnd"].get("type") == "TOOL":
                clog.info("Tool requested: %s", pending_tool_name)

                raw_content = pending_tool_content.get("content", "{}")
                if isinstance(raw_content, str):
                    try:
                        tool_args = json.loads(raw_content)
                    except json.JSONDecodeError:
                        tool_args = {}
                else:
                    tool_args = raw_content

                result = await graph.ainvoke(
                    {"conversation_history": [{"type": "tool_call", "function_name": pending_tool_name, "arguments": tool_args}]},
                    config=thread_cfg,
                )
                tool_results = result.get("tool_results", [])
                tool_output = tool_results[-1]["result"] if tool_results else {"error": "no result"}

                await sonic.send_tool_result(pending_tool_id, tool_output)

            # completionEnd → Nova finished a full response turn
            if "completionEnd" in ev:
                transcript = "".join(nova_text_buffer).strip()
                nova_text_buffer.clear()

                result = await graph.ainvoke(
                    {"conversation_history": [{"type": "nova_turn", "text": transcript or "(audio only)"}]},
                    config=thread_cfg,
                )
                clog.info("Turn %d — status=%s", result.get("turn_count", 0), result.get("status", "active"))

                if result.get("status") == "ending":
                    clog.info("LangGraph decided to end the call")
                    await asyncio.sleep(1.5)
                    await call.disconnect()
                    return

    try:
        await asyncio.gather(stream_to_nova(), receive_from_nova())
    finally:
        await sonic.close()

    clog.info("Call session ended")


# ===========================================================================
#  Entry point
# ===========================================================================


async def main() -> None:
    graph = build_call_graph()
    log.info("LangGraph call-management graph compiled")

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
                await handle_nova_call(call, graph)
            except Exception:
                log.exception("[%s] Call handler failed", cid)

        log.info("Nova (B3networks · Nova Sonic 2) is live — waiting for calls …")
        await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
