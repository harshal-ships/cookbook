"""
Nova — AI Customer Care Agent for B3Networks
=============================================
A fully autonomous customer care voice agent built with the **Google ADK**
(Agent Development Kit) and the **Telcoflow SDK**.

Nova answers every call, identifies the caller, and resolves their issue
using tools for account lookup, ticketing, billing, product info, and FAQ
search.  There is no human escalation — Nova handles it all.

Key difference from the other agents in this repo: Nova uses the Google ADK
`Agent → Runner → run_live()` pipeline instead of the raw GenAI Live API.
ADK manages the tool-execution loop automatically, so there is no manual
`tool_call` / `send_tool_response` dispatch — tools are plain Python
functions that ADK calls on behalf of the model.

State flow:  PENDING → ANSWERED → DISCONNECTED
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types
from telcoflow_sdk import TelcoflowClient, TelcoflowClientConfig, ActiveCall
import telcoflow_sdk.events as events

from config import NovaConfig
from database import NovaDB
from tools import CallMeta, create_nova_tools

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nova")

# ---------------------------------------------------------------------------
# System instruction
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """
You are **Nova**, the AI customer care assistant for **{company}**.
You handle all customer inquiries over the phone — you never transfer to a
human agent.

## YOUR PERSONALITY
- Warm, professional, and solution-oriented.
- Concise but thorough — answer the question, don't ramble.
- Patient with frustrated customers.
- Confident — you can resolve everything.

## ON EVERY CALL
1. Greet the caller: "Hi, this is Nova from {company}. How can I help you today?"
2. Immediately call `look_up_customer` with the caller's phone number.
3. If the customer is found, use their name naturally in the conversation.
4. If not found, still help them — create a ticket if needed.
5. Listen to the request and use the right tools.
6. When the caller is satisfied, summarise what you did, say goodbye, and call
   `end_call` exactly once.

## TOOLS
- `look_up_customer` — **Always call first** to identify the caller.
- `get_ticket_details` — Check an existing ticket by ID.
- `list_customer_tickets` — Show all tickets for the caller.
- `create_ticket` — Open a new ticket when you cannot fully resolve the issue
  right now.  This guarantees a human follow-up within 24 hours.
- `get_billing_summary` — Answer billing and invoice questions.
- `get_product_info` — Explain B3Networks products and services.
- `search_faq` — Find answers to technical and product questions.
- `update_customer_email` — Update the email on file.
- `end_call` — End the call.  Call exactly once, as the very last action.

## RULES
- NEVER say you need to transfer the caller to a human.
- If the FAQ doesn't answer a question, create a ticket and assure follow-up.
- If the caller mentions a ticket ID (e.g. TKT-1001), look it up immediately.
- Handle multiple topics one at a time — ask "anything else?" between them.
- Keep answers conversational and brief.  This is a phone call, not an email.
- Always end with a clear summary of what was accomplished.
""".strip()

# ---------------------------------------------------------------------------
# Per-call session
# ---------------------------------------------------------------------------


async def handle_nova_call(
    call: ActiveCall,
    db: NovaDB,
    cfg: NovaConfig,
) -> None:
    call_id = getattr(call, "call_id", "unknown")
    caller_phone = getattr(call, "caller_number", None) or "unknown"
    log = logger.getChild(call_id[:12])
    call_start = time.monotonic()

    @call.on(events.CALL_TERMINATED)
    def on_terminated():
        log.info("CALL_TERMINATED")

    await call.answer()
    log.info("Nova call from %s", caller_phone)

    # -- per-call tool closures --------------------------------------------
    should_end = asyncio.Event()
    meta = CallMeta(caller_phone=caller_phone)
    nova_tools = create_nova_tools(db, meta, should_end)

    # -- ADK agent (created per call so tools can close over call state) ----
    instruction = SYSTEM_INSTRUCTION.format(company=cfg.business_name)

    agent = Agent(
        name="nova",
        model=cfg.gemini_model,
        instruction=instruction,
        tools=nova_tools,
    )
    session_service = InMemorySessionService()
    runner = Runner(
        app_name="nova_customer_care",
        agent=agent,
        session_service=session_service,
    )

    live_queue = LiveRequestQueue()

    # -- Caller audio → ADK ------------------------------------------------
    async def stream_to_adk():
        try:
            async for chunk in call.audio_stream():
                live_queue.send_realtime(
                    types.Blob(
                        data=chunk,
                        mime_type=f"audio/pcm;rate={cfg.sample_rate}",
                    )
                )
        finally:
            live_queue.close()

    # -- ADK → Caller (audio events only; tool calls handled by Runner) ----
    async def receive_from_adk():
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
        )
        async for event in runner.run_live(
            user_id=call_id,
            session_id=call_id,
            live_request_queue=live_queue,
            run_config=run_config,
        ):
            if event.interrupted:
                await call.clear_send_audio_buffer()

            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.inline_data:
                        await call.send_audio(part.inline_data.data)

    # -- Lifecycle watcher (waits for end_call tool, then disconnects) -----
    async def lifecycle_watcher():
        await should_end.wait()
        await asyncio.sleep(2)
        log.info("Ending call")
        await call.disconnect()

    await asyncio.gather(
        stream_to_adk(),
        receive_from_adk(),
        lifecycle_watcher(),
    )

    # -- Post-call logging -------------------------------------------------
    duration = time.monotonic() - call_start
    await db.log_interaction(
        call_id=call_id,
        customer_id=meta.customer_id,
        summary=meta.summary,
        tools_used=",".join(meta.tools_used),
        duration_seconds=round(duration, 1),
    )
    log.info(
        "Call complete — tools=%s duration=%.1fs summary=%s",
        meta.tools_used,
        duration,
        meta.summary,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def main() -> None:
    cfg = NovaConfig.from_env()

    db = NovaDB(cfg.db_path)
    await db.connect()

    os.environ.setdefault("GOOGLE_API_KEY", cfg.gemini_api_key)

    tf_config = TelcoflowClientConfig.sandbox(
        api_key=cfg.wss_api_key,
        connector_uuid=cfg.wss_connector_uuid,
        sample_rate=cfg.sample_rate,
    )

    async with TelcoflowClient(tf_config) as tf_client:

        @tf_client.on(events.INCOMING_CALL)
        async def on_incoming(call: ActiveCall):
            cid = getattr(call, "call_id", "?")
            logger.info("[%s] Incoming call", cid)
            try:
                await handle_nova_call(call, db, cfg)
            except Exception:
                logger.exception("[%s] Nova session failed", cid)

        logger.info(
            "Nova [%s] is live — fully autonomous customer care — waiting for calls …",
            cfg.business_name,
        )

        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        run_task = asyncio.create_task(tf_client.run_forever())
        stop_task = asyncio.create_task(stop.wait())
        await asyncio.wait(
            [run_task, stop_task], return_when=asyncio.FIRST_COMPLETED
        )

        if stop.is_set():
            logger.info("Shutdown signal received")
            run_task.cancel()

    await db.close()
    logger.info("Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
