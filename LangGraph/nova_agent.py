"""
Nova — B3networks Customer Care Voice Agent (LangGraph Edition)
================================================================

A complete voice agent using:
  - Telcoflow SDK  →  phone call audio I/O
  - Google Gemini Live (GenAI SDK)  →  speech-to-speech AI
  - LangGraph StateGraph  →  per-call conversation state & tool decisions

Architecture
------------
    Caller audio  →  Telcoflow  →  Gemini Live  →  Audio back to caller
                                        ↕
                                 LangGraph StateGraph
                            (conversation state per call)

Gemini Live handles audio natively — it listens to the caller and speaks
back in real time.  LangGraph sits above that layer and manages per-call
state: conversation history, turn counts, and tool execution.  When
Gemini decides it needs information (a tool call), the request is routed
through LangGraph, which records it, executes the tool, and hands the
result back to Gemini so it can continue the conversation.

The call_id from Telcoflow is used as the LangGraph thread_id, so every
concurrent call gets its own completely isolated state.

Run with:  python nova_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import operator
import os
import random
from typing import Annotated, Literal

from typing_extensions import TypedDict

from google import genai
from google.genai import types

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
# Configuration — every value comes from an environment variable
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# System prompt — tells Gemini Live who Nova is and how she should behave
# ---------------------------------------------------------------------------

NOVA_SYSTEM_PROMPT = """\
Your name is Nova. You are a customer care agent from B3networks.
You are polite, warm, and professional at all times.
You greet every caller with: Hi, I am Nova from B3networks. How can I help you today?
You listen carefully and answer questions about B3networks products and services.
You try your best to resolve the customer issue in the same call.

When you need information about a product, use the get_product_info tool.
When a caller asks about service status, use the check_service_status tool.
When you cannot fully resolve an issue on the call, use the create_support_ticket
tool so the team can follow up within 24 hours.
"""

# ---------------------------------------------------------------------------
# Tool implementations
#
# Plain Python functions that do the actual work when Gemini requests a
# tool call.  These are executed inside a LangGraph node, which records
# every call and its result in the per-call state.
# ---------------------------------------------------------------------------

PRODUCT_CATALOG = {
    "unified communications": {
        "name": "Unified Communications",
        "description": (
            "Cloud-based phone system with voice, video, messaging, "
            "and team collaboration."
        ),
        "pricing": "Starting at $15 per user per month",
    },
    "contact center": {
        "name": "Contact Center",
        "description": (
            "Omnichannel contact center with IVR, intelligent routing, "
            "queuing, and real-time analytics."
        ),
        "pricing": "Starting at $50 per agent per month",
    },
    "cpaas": {
        "name": "CPaaS",
        "description": (
            "Programmable communication APIs for voice, SMS, and video "
            "integration into your applications."
        ),
        "pricing": "Pay-as-you-go based on usage",
    },
    "sip trunking": {
        "name": "SIP Trunking",
        "description": (
            "Enterprise-grade SIP trunking for connecting your PBX "
            "to the PSTN with high reliability."
        ),
        "pricing": "Starting at $0.01 per minute",
    },
}


def get_product_info(product_name: str) -> dict:
    """Look up B3networks product information by name or keyword."""
    query = product_name.lower().strip()
    for key, info in PRODUCT_CATALOG.items():
        if query in key or query in info["name"].lower():
            return info
    available = [v["name"] for v in PRODUCT_CATALOG.values()]
    return {
        "error": f"No product found matching '{product_name}'.",
        "available_products": available,
    }


def check_service_status() -> dict:
    """Check the current operational status of B3networks services."""
    return {
        "overall_status": "operational",
        "services": {
            "Unified Communications": "operational",
            "Contact Center": "operational",
            "CPaaS": "operational",
            "SIP Trunking": "operational",
        },
        "message": "All B3networks services are running normally.",
    }


def create_support_ticket(issue_description: str) -> dict:
    """Create a support ticket for issues that cannot be resolved on the call."""
    ticket_id = f"TKT-{random.randint(10000, 99999)}"
    return {
        "ticket_id": ticket_id,
        "status": "created",
        "message": (
            f"Support ticket {ticket_id} created. "
            "Our team will follow up within 24 hours."
        ),
    }


# Maps each function name to its implementation so LangGraph can look them up
TOOL_REGISTRY: dict[str, callable] = {
    "get_product_info": get_product_info,
    "check_service_status": check_service_status,
    "create_support_ticket": create_support_ticket,
}

# ---------------------------------------------------------------------------
# Gemini tool declarations
#
# These tell Gemini what tools Nova can use during the conversation.
# The schemas mirror the Python functions above.  When Gemini decides a
# tool is needed, it pauses audio generation and sends a structured
# tool_call event that we intercept and route through LangGraph.
# ---------------------------------------------------------------------------

GEMINI_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="get_product_info",
                description=(
                    "Look up B3networks product information. Use when a "
                    "caller asks about a specific product or service."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "product_name": {
                            "type": "STRING",
                            "description": "Product name or keyword to search for",
                        }
                    },
                    "required": ["product_name"],
                },
            ),
            types.FunctionDeclaration(
                name="check_service_status",
                description=(
                    "Check the current operational status of B3networks "
                    "services. Use when a caller reports an outage or "
                    "asks if services are working."
                ),
                parameters={"type": "OBJECT", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="create_support_ticket",
                description=(
                    "Create a support ticket for issues that cannot be "
                    "resolved during the call."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "issue_description": {
                            "type": "STRING",
                            "description": "Clear description of the customer issue",
                        }
                    },
                    "required": ["issue_description"],
                },
            ),
        ]
    )
]

# ===========================================================================
#
#  LangGraph — Per-call conversation state management
#
#  LangGraph does NOT handle audio.  Its job is to maintain an isolated
#  state for each phone call: what turns have happened, which tools were
#  called, what results came back, and whether the call should end.
#
#  The graph is invoked every time something interesting happens on the
#  audio layer (a turn completes, a tool is requested).  The MemorySaver
#  checkpointer persists state between invocations, keyed by thread_id
#  which is set to the Telcoflow call_id.
#
#  Graph structure:
#
#      START
#        │
#        ▼
#    process_event          ← records the event, increments turn count
#        │
#        ▼
#    route_event            ← conditional edge
#        │
#    ┌───┴────────┐
#    ▼            ▼
#  execute_tool   assess_call
#    │            │
#    ▼            ▼
#  assess_call   END
#    │
#    ▼
#   END
#
# ===========================================================================


class CallState(TypedDict):
    """Per-call state schema.

    Fields annotated with operator.add use an append reducer: new items
    are added to the existing list rather than replacing it.  Plain
    fields use the default replace-on-write behaviour.
    """

    call_id: str
    conversation_history: Annotated[list[dict], operator.add]
    turn_count: int
    tool_results: Annotated[list[dict], operator.add]
    status: str  # "active" | "ending"


# ---- Nodes ----------------------------------------------------------------


def process_event(state: CallState) -> dict:
    """Increment the turn counter each time the graph is invoked.

    The new conversation event is already in conversation_history by the
    time this node runs (the append reducer merges it during input).
    """
    return {"turn_count": state.get("turn_count", 0) + 1}


def execute_tool(state: CallState) -> dict:
    """Run the requested tool and store the result.

    Reads the latest conversation_history entry (which should be a
    tool_call event), looks up the matching function in TOOL_REGISTRY,
    executes it, and appends the result to tool_results.
    """
    history = state.get("conversation_history", [])
    if not history:
        return {}

    last = history[-1]
    if last.get("type") != "tool_call":
        return {}

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

    return {
        "tool_results": [
            {
                "function_name": func_name,
                "arguments": func_args,
                "result": result,
            }
        ]
    }


def assess_call(state: CallState) -> dict:
    """Decide whether the call should continue or end.

    Scans the last conversation event for farewell language.  If Nova
    has said goodbye, the status is flipped to "ending" so the audio
    handler knows to disconnect.
    """
    history = state.get("conversation_history", [])
    if not history:
        return {"status": "active"}

    last = history[-1]

    if last.get("type") == "nova_turn" and last.get("text"):
        text = last["text"].lower()
        farewell_signals = [
            "goodbye", "bye", "have a great day",
            "take care", "have a good",
        ]
        if any(sig in text for sig in farewell_signals):
            return {"status": "ending"}

    return {"status": "active"}


# ---- Routing --------------------------------------------------------------


def route_event(state: CallState) -> Literal["execute_tool", "assess_call"]:
    """Conditional edge: route tool_call events to the tool executor,
    everything else straight to call assessment.
    """
    history = state.get("conversation_history", [])
    if history and history[-1].get("type") == "tool_call":
        return "execute_tool"
    return "assess_call"


# ---- Graph assembly -------------------------------------------------------


def build_call_graph():
    """Build and compile the LangGraph StateGraph.

    Returns a compiled graph backed by an in-memory checkpointer.
    Each phone call uses its call_id as the thread_id, giving every
    call fully isolated state.
    """
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
#
#  Call handler — bridges Telcoflow, Gemini Live, and LangGraph
#
#  Audio responsibility:
#    Telcoflow  →  streams caller PCM audio in and out
#    Gemini Live  →  speech-to-speech AI conversation
#
#  State responsibility:
#    LangGraph  →  records turns, executes tools, decides when to end
#
# ===========================================================================


async def handle_nova_call(call: ActiveCall, graph) -> None:
    """Handle one incoming phone call from start to finish.

    1. Answer the call and open the audio channel.
    2. Connect to Gemini Live with Nova's system prompt and tools.
    3. Run two concurrent tasks:
       - stream_to_gemini:   caller audio  →  Gemini
       - receive_from_gemini: Gemini events  →  caller audio + LangGraph
    4. When LangGraph marks status as "ending", disconnect the call.
    """
    call_id = call.call_id
    thread_cfg = {"configurable": {"thread_id": call_id}}
    clog = log.getChild(call_id[:8])

    # -- Step 1: answer the call --------------------------------------------
    await call.answer()
    clog.info("Call answered — starting Nova session")

    # -- Step 2: initialise LangGraph state for this call -------------------
    await graph.ainvoke(
        {
            "call_id": call_id,
            "conversation_history": [
                {"type": "call_started", "call_id": call_id}
            ],
            "turn_count": 0,
            "tool_results": [],
            "status": "active",
        },
        config=thread_cfg,
    )

    # -- Step 3: open a Gemini Live session ---------------------------------
    gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    live_config = types.LiveConnectConfig(
        system_instruction=NOVA_SYSTEM_PROMPT,
        response_modalities=["AUDIO"],
        tools=GEMINI_TOOLS,
    )

    async with gemini_client.aio.live.connect(
        model=GEMINI_MODEL, config=live_config
    ) as session:

        # -- Stream caller audio → Gemini ----------------------------------
        async def stream_to_gemini():
            """Read PCM chunks from the caller and forward to Gemini Live.

            call.audio_stream() is an async iterator of raw PCM bytes from
            the phone call.  Each chunk is wrapped in a types.Blob and sent
            to the Gemini session as real-time audio input.
            """
            async for audio_chunk in call.audio_stream():
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=audio_chunk,
                        mime_type=f"audio/pcm;rate={SAMPLE_RATE}",
                    )
                )

        # -- Receive Gemini events → caller audio + LangGraph ---------------
        async def receive_from_gemini():
            """Handle all Gemini Live events for this call.

            Three event types matter here:

            1. server_content — model audio/text and turn lifecycle
               • inline_data parts  →  forwarded to caller via send_audio()
               • text parts  →  buffered and fed to LangGraph on turn_complete
               • interrupted  →  clear the audio buffer immediately
               • turn_complete  →  invoke LangGraph to record the turn

            2. tool_call — Gemini wants information from a tool
               • routed through LangGraph (which executes the tool)
               • tool result sent back to Gemini so it can keep talking

            3. tool_call_cancellation — Gemini cancelled a pending tool call
               (logged but otherwise ignored in this example)
            """
            nova_text_buffer: list[str] = []

            async for response in session.receive():

                # ---- Audio / text content from Gemini --------------------
                if response.server_content:
                    content = response.server_content

                    # Caller spoke over Nova — stop pending audio right away
                    if content.interrupted:
                        await call.clear_send_audio_buffer()
                        nova_text_buffer.clear()
                        clog.info("Interrupted — audio buffer cleared")

                    # Process model turn parts (audio and optional text)
                    if content.model_turn:
                        for part in content.model_turn.parts:
                            if part.inline_data:
                                await call.send_audio(
                                    part.inline_data.data
                                )
                            if part.text:
                                nova_text_buffer.append(part.text)

                    # Nova finished this turn — feed transcript to LangGraph
                    if content.turn_complete:
                        transcript = "".join(nova_text_buffer).strip()
                        nova_text_buffer.clear()

                        result = await graph.ainvoke(
                            {
                                "conversation_history": [
                                    {
                                        "type": "nova_turn",
                                        "text": transcript or "(audio only)",
                                    }
                                ],
                            },
                            config=thread_cfg,
                        )

                        clog.info(
                            "Turn %d — status=%s",
                            result.get("turn_count", 0),
                            result.get("status", "active"),
                        )

                        # LangGraph says the call should end
                        if result.get("status") == "ending":
                            clog.info("LangGraph decided to end the call")
                            await asyncio.sleep(1.5)
                            await call.disconnect()
                            return

                # ---- Tool call from Gemini --------------------------------
                # Gemini pauses audio and asks for tool results.  We route
                # every tool call through LangGraph so the execution is
                # recorded in conversation state.  Then we send the result
                # back to Gemini and it resumes speaking.
                if response.tool_call:
                    function_responses = []

                    for fc in response.tool_call.function_calls:
                        clog.info(
                            "Tool requested: %s(%s)", fc.name, fc.args
                        )

                        # Invoke LangGraph with the tool_call event.
                        # The execute_tool node runs the function and
                        # stores the result in state.tool_results.
                        result = await graph.ainvoke(
                            {
                                "conversation_history": [
                                    {
                                        "type": "tool_call",
                                        "function_name": fc.name,
                                        "arguments": (
                                            dict(fc.args) if fc.args else {}
                                        ),
                                    }
                                ],
                            },
                            config=thread_cfg,
                        )

                        # Pull the latest tool result from LangGraph state
                        tool_results = result.get("tool_results", [])
                        tool_output = (
                            tool_results[-1]["result"]
                            if tool_results
                            else {"error": "no result"}
                        )

                        function_responses.append(
                            types.FunctionResponse(
                                name=fc.name,
                                response=tool_output,
                            )
                        )

                    # Send all tool results back to Gemini in one batch
                    await session.send_tool_response(
                        function_responses=function_responses
                    )

        # -- Run both directions concurrently -------------------------------
        await asyncio.gather(stream_to_gemini(), receive_from_gemini())

    clog.info("Nova session complete")


# ===========================================================================
#  Entry point
# ===========================================================================


async def main() -> None:
    """Start the Nova voice agent.

    1. Compile the LangGraph graph (shared across calls; state is
       isolated per call via thread_id).
    2. Connect to Telcoflow in sandbox mode.
    3. Register the incoming-call handler.
    4. Wait for calls forever.
    """
    graph = build_call_graph()
    log.info("LangGraph call-management graph compiled")

    tf_config = TelcoflowClientConfig.sandbox(
        api_key=os.environ["WSS_API_KEY"],
        connector_uuid=os.environ["WSS_CONNECTOR_UUID"],
        sample_rate=SAMPLE_RATE,
    )

    async with TelcoflowClient(tf_config) as client:

        @client.on(events.INCOMING_CALL)
        async def on_incoming_call(call: ActiveCall):
            """Each incoming call is handled in its own coroutine."""
            cid = call.call_id[:8]
            log.info("[%s] Incoming call", cid)
            try:
                await handle_nova_call(call, graph)
            except Exception:
                log.exception("[%s] Call handler failed", cid)

        log.info(
            "Nova (B3networks) is live — waiting for calls …"
        )
        await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
