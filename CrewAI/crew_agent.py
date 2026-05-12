"""
Crew — B3networks Call Triage System
=============================================

A call triage system using:
  - Telcoflow SDK  →  phone call audio I/O
  - Google Gemini Live (GenAI SDK)  →  audio layer (speech ↔ text)
  - CrewAI  →  multi-agent triage (Receptionist → Analyst → Resolver)

Architecture
------------
    Caller audio  →  Telcoflow  →  Gemini Live  →  Audio back to caller
                                        ↕
                                  CrewAI Crew
                      (3 agents collaborate after each caller turn)

Gemini Live acts as the audio bridge between the phone line and the
triage pipeline.  It receives caller speech, processes it, and speaks
the CrewAI Resolver's response back to the caller.

CrewAI runs a sequential three-agent pipeline after each caller turn:
  1. Receptionist  — extracts key information from the caller's message
  2. Analyst       — categorises the issue (billing, technical,etc.)
  3. Resolver      — crafts the spoken response for the caller

The call_id from Telcoflow isolates each call's crew run.

Run with:  python crew_agent.py
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from crewai import Agent, Task, Crew, Process, LLM

from google import genai
from google.genai import types

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
log = logging.getLogger("crew")

# ---------------------------------------------------------------------------
# Configuration — all values from environment variables
# ---------------------------------------------------------------------------

GEMINI_LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# System prompt for Gemini Live (audio layer only)
#
# Gemini Live's responsibility in this architecture:
#   1. Receive caller audio and produce a text understanding of it.
#   2. Receive text from the triage pipeline and speak it to the caller.
#
# It does NOT drive the conversation — CrewAI does.
# ---------------------------------------------------------------------------

CREW_SYSTEM_PROMPT = """\
Your name is Crew. You are a customer care agent from B3networks.
You are polite, warm, and professional at all times.
Listen carefully to the caller and transcribe their message accurately.

When you receive a text message, it contains the exact response you must
speak to the caller. Say those words naturally in your warm, professional
voice. Do not add commentary or change the message.
"""

# ---------------------------------------------------------------------------
# CrewAI LLM — native Gemini binding (Agent rejects raw ChatGoogleGenerativeAI)
#
# Uses GOOGLE_API_KEY / GEMINI_API_KEY from the environment. Model id matches
# CrewAI’s Google provider: gemini/gemini-2.0-flash
# ---------------------------------------------------------------------------

crew_llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

# ---------------------------------------------------------------------------
# CrewAI agent definitions
#
# Three specialised agents form the triage pipeline.  Each has a clearly
# defined role, goal, and backstory as CrewAI requires.
# ---------------------------------------------------------------------------

receptionist_agent = Agent(
    role="Call Receptionist",
    goal=(
        "Greet the caller warmly, understand why they are calling, "
        "and extract the key details from their message."
    ),
    backstory=(
        "You are the front-desk receptionist at B3networks with years of "
        "experience handling customer calls. You are friendly, attentive, "
        "and skilled at quickly understanding what a customer needs. You "
        "always make callers feel welcome and heard."
    ),
    llm=crew_llm,
    verbose=True,
    allow_delegation=False,
)

analyst_agent = Agent(
    role="Issue Analyst",
    goal=(
        "Take the receptionist's summary, deeply understand the core issue, "
        "and categorise it accurately into: billing, technical support, "
        "general enquiry, account management, or escalation."
    ),
    backstory=(
        "You are a senior support analyst at B3networks with deep knowledge "
        "of all products — Unified Communications, Contact Center, CPaaS, "
        "and SIP Trunking. You quickly identify patterns and classify "
        "customer problems with precision."
    ),
    llm=crew_llm,
    verbose=True,
    allow_delegation=False,
)

resolver_agent = Agent(
    role="Response Resolver",
    goal=(
        "Craft a clear, helpful, and warm spoken response for the caller "
        "based on the analyst's categorisation. The response will be spoken "
        "aloud by a voice agent, so it must be conversational and concise."
    ),
    backstory=(
        "You are B3networks' top customer care specialist. You excel at "
        "turning technical analysis into friendly, human spoken responses "
        "that make customers feel heard and helped. You keep answers short "
        "and actionable."
    ),
    llm=crew_llm,
    verbose=True,
    allow_delegation=False,
)


# ---------------------------------------------------------------------------
# Build a fresh triage crew for one caller turn
#
# A new Crew is built per turn so that tasks receive the latest context
# (current transcript and full conversation history).  The three tasks
# chain via the `context` parameter:
#   reception_task → analysis_task → resolver_task
# ---------------------------------------------------------------------------


def build_triage_crew(
    caller_transcript: str,
    conversation_history: str,
    call_id: str,
) -> Crew:
    """Assemble the Receptionist → Analyst → Resolver pipeline.

    Args:
        caller_transcript: What the caller said in this turn.
        conversation_history: Full conversation so far (all turns).
        call_id: Telcoflow call ID for logging and context.

    Returns:
        A Crew ready to kickoff().
    """
    # Task 1 — Receptionist extracts key information from the transcript
    reception_task = Task(
        description=(
            f"You are handling call {call_id}.\n\n"
            f"Conversation so far:\n{conversation_history}\n\n"
            f"Caller's latest message:\n\"{caller_transcript}\"\n\n"
            "Extract: who the caller is (if mentioned), which product or "
            "service they are asking about, and the reason for their call."
        ),
        expected_output=(
            "A concise summary: caller identity (if known), product/service "
            "mentioned, and reason for calling."
        ),
        agent=receptionist_agent,
    )

    # Task 2 — Analyst categorises the issue using Receptionist's output
    analysis_task = Task(
        description=(
            "Based on the receptionist's summary, categorise the customer "
            "issue into ONE of: billing, technical support, general enquiry, "
            "account management, or escalation. Include a brief rationale."
        ),
        expected_output=(
            "Category (billing / technical support / general enquiry / "
            "account management / escalation) and a one-line rationale."
        ),
        agent=analyst_agent,
        context=[reception_task],
    )

    # Task 3 — Resolver crafts the spoken response using Analyst's output
    resolver_task = Task(
        description=(
            "Using the analyst's categorisation, craft a warm, professional "
            "spoken response for the caller. Requirements:\n"
            "- Conversational and natural (will be spoken aloud)\n"
            "- No bullet points, markdown, or formatting\n"
            "- Address the caller's concern directly\n"
            "- Offer concrete next steps if you cannot fully resolve\n"
            "- Keep it concise: 2–4 sentences maximum"
        ),
        expected_output=(
            "A short spoken response (2–4 sentences) ready to be read aloud. "
            "No formatting. Conversational tone."
        ),
        agent=resolver_agent,
        context=[analysis_task],
    )

    return Crew(
        agents=[receptionist_agent, analyst_agent, resolver_agent],
        tasks=[reception_task, analysis_task, resolver_task],
        process=Process.sequential,
        verbose=True,
    )


# ===========================================================================
#
#  Call handler — bridges Telcoflow, Gemini Live, and CrewAI
#
#  Audio layer (Telcoflow + Gemini Live):
#    - Telcoflow streams caller PCM audio in and out of the phone line
#    - Gemini Live converts speech → text (listening) and text → speech
#      (speaking the CrewAI response)
#
#  Triage layer (CrewAI):
#    - After each caller turn, the transcript goes through the 3-agent
#      pipeline: Receptionist → Analyst → Resolver
#    - The Resolver's output is sent to Gemini Live for speaking
#
#  Two-phase turn cycle:
#
#    SPEAKING phase (speaking_phase = True):
#      Gemini generates audio from the CrewAI response (or greeting).
#      Audio parts ARE forwarded to the caller via call.send_audio().
#      When the turn completes, we switch to LISTENING.
#
#    LISTENING phase (speaking_phase = False):
#      The caller is speaking.  Gemini receives their audio and produces
#      text (its understanding of the caller's message).  Audio output
#      from Gemini is suppressed — we only collect the text.
#      When the turn completes, we run CrewAI, then switch to SPEAKING.
#
# ===========================================================================

# Per-call conversation history, keyed by call_id
call_histories: dict[str, list[str]] = {}


async def handle_crew_call(call: ActiveCall) -> None:
    """Handle one phone call through the CrewAI triage pipeline.

    Flow per turn:
      1. Caller speaks → audio streams to Gemini Live.
      2. Gemini produces text (transcript) → collected, audio suppressed.
      3. CrewAI triage runs: Receptionist → Analyst → Resolver.
      4. Resolver's output → sent to Gemini → Gemini speaks it to caller.
      5. Repeat from step 1.
    """
    call_id = call.call_id
    call_histories[call_id] = []
    clog = log.getChild(call_id[:8])

    # -- Step 1: Answer the call and open the audio channel -----------------
    await call.answer()
    clog.info("Call answered — starting Crew triage session")

    # -- Step 2: Connect to Gemini Live for the audio layer -----------------
    gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    live_config = types.LiveConnectConfig(
        system_instruction=CREW_SYSTEM_PROMPT,
        response_modalities=["AUDIO"],
    )

    async with gemini_client.aio.live.connect(
        model=GEMINI_LIVE_MODEL, config=live_config
    ) as session:

        # Start in SPEAKING phase — Crew speaks the greeting first
        speaking_phase = True

        greeting = "Hi, I am Crew from B3networks. How can I help you today?"
        call_histories[call_id].append(f"Crew: {greeting}")
        await session.send(text=greeting, end_of_turn=True)

        # -- Coroutine 1: Stream caller audio → Gemini (runs continuously) --
        async def stream_to_gemini():
            """Forward PCM audio chunks from the phone line to Gemini Live.

            call.audio_stream() yields raw PCM bytes at SAMPLE_RATE.
            Each chunk is sent to Gemini as real-time audio input.
            """
            async for audio_chunk in call.audio_stream():
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=audio_chunk,
                        mime_type=f"audio/pcm;rate={SAMPLE_RATE}",
                    )
                )

        # -- Coroutine 2: Handle Gemini events → audio + CrewAI triage ------
        async def receive_from_gemini():
            """Process all events from the Gemini Live session.

            SPEAKING phase:
              Audio parts → forwarded to caller via call.send_audio()
              turn_complete → switch to LISTENING

            LISTENING phase:
              Audio parts → suppressed (only collecting text)
              turn_complete → run CrewAI triage → switch to SPEAKING
            """
            nonlocal speaking_phase
            text_buffer: list[str] = []

            async for response in session.receive():
                if not response.server_content:
                    continue

                content = response.server_content

                # Caller interrupted while Crew was speaking — clear audio
                if content.interrupted:
                    await call.clear_send_audio_buffer()
                    text_buffer.clear()
                    clog.info("Interrupted — audio buffer cleared")

                # Process model turn parts (audio and text)
                if content.model_turn:
                    for part in content.model_turn.parts:
                        # Only forward audio to the caller in SPEAKING phase
                        if part.inline_data and speaking_phase:
                            await call.send_audio(part.inline_data.data)
                        if part.text:
                            text_buffer.append(part.text)

                # A turn has completed — act based on current phase
                if content.turn_complete:
                    transcript = "".join(text_buffer).strip()
                    text_buffer.clear()

                    if speaking_phase:
                        # Crew finished speaking — now listen for the caller
                        speaking_phase = False
                        clog.info("Crew finished speaking — listening")

                    else:
                        # Gemini processed caller audio — we have a transcript
                        caller_text = transcript or "(inaudible)"
                        call_histories[call_id].append(
                            f"Caller: {caller_text}"
                        )
                        clog.info("Caller: %s", caller_text)

                        # ---- Bridge: CrewAI triage pipeline ---------------
                        # crew.kickoff() is synchronous, so we run it in a
                        # thread to avoid blocking the async event loop.
                        history_text = "\n".join(call_histories[call_id])
                        crew = build_triage_crew(
                            caller_transcript=caller_text,
                            conversation_history=history_text,
                            call_id=call_id,
                        )

                        clog.info(
                            "Running triage: Receptionist → Analyst → Resolver"
                        )

                        try:
                            result = await asyncio.to_thread(crew.kickoff)
                            resolver_response = result.raw.strip()
                        except Exception as exc:
                            clog.error("CrewAI triage failed: %s", exc)
                            resolver_response = (
                                "I apologise, I'm having a little trouble "
                                "processing that. Could you please repeat "
                                "your question?"
                            )

                        call_histories[call_id].append(
                            f"Crew: {resolver_response}"
                        )
                        clog.info("Resolver: %s", resolver_response)

                        # ---- Bridge: Send response to Gemini for TTS -----
                        # Switch to SPEAKING phase and send the Resolver's
                        # output to Gemini Live.  Gemini will generate audio
                        # for this text, which gets forwarded to the caller
                        # via call.send_audio() in the SPEAKING handler above.
                        speaking_phase = True
                        await session.send(
                            text=resolver_response, end_of_turn=True
                        )

        # -- Run both coroutines concurrently for the call's lifetime -------
        await asyncio.gather(stream_to_gemini(), receive_from_gemini())

    # Clean up per-call state
    call_histories.pop(call_id, None)
    clog.info("Crew triage session complete")


# ===========================================================================
#  Entry point
# ===========================================================================


async def main() -> None:
    """Start the Crew triage voice agent.

    1. Connect to Telcoflow in sandbox mode.
    2. Register the incoming-call handler.
    3. Wait for calls forever.
    """
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
                await handle_crew_call(call)
            except Exception:
                log.exception("[%s] Call handler failed", cid)

        log.info(
            "Crew (B3networks) is live — waiting for calls …"
        )
        await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
