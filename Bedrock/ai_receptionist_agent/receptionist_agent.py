"""AI receptionist with CRM lookup and human handoff — Telcoflow + Nova 2 Sonic.

Based on: https://docs.agentao.com/use-cases/database-lookup

On each inbound call:
1. Look up caller_number in CRM (JSON demo or Pipedrive)
2. Known caller → personalized greeting + open tickets in system prompt
3. Unknown caller → create lead in leads.json
4. Nova Sonic handles the conversation (intent, FAQs, routing)
5. transfer_to_human tool → connect() + close() to ring the original callee

Run:
    python receptionist_agent.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from telcoflow_sdk import ActiveCall, TelcoflowClient, TelcoflowClientConfig
import telcoflow_sdk.events as events

from crm import CRMBackend, create_crm_backend
from nova_sonic_bridge import NovaCallResult, NovaSonicBridge, TranscriptLine

load_dotenv()

RECEPTION_CALLS_PATH = Path(os.getenv("RECEPTION_CALLS_PATH", "reception_calls.json")).resolve()
CRM_BACKEND = os.getenv("CRM_BACKEND", "json").strip().lower()
COMPANY_NAME = os.getenv("COMPANY_NAME", "Acme Corp")
DEFAULT_PHONE_COUNTRY_CODE = os.getenv("DEFAULT_PHONE_COUNTRY_CODE", "91").strip()
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
NOVA_MODEL_ID = os.getenv("NOVA_MODEL_ID", "amazon.nova-2-sonic-v1:0")
NOVA_VOICE_ID = os.getenv("NOVA_VOICE_ID", "matthew")
NOVA_ENABLE_TOOLS = os.getenv("NOVA_ENABLE_TOOLS", "true").lower() in {"1", "true", "yes", "on"}
HANDOFF_AUDIO_GRACE_SECONDS = float(os.getenv("HANDOFF_AUDIO_GRACE_SECONDS", "3"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


def build_system_prompt(
    *,
    customer: dict[str, Any] | None,
    open_tickets: list[dict[str, Any]],
    lead_created: bool,
    caller_number: str | None,
) -> str:
    lines = [
        f"You are the AI receptionist for {COMPANY_NAME}.",
        "You answer inbound phone calls professionally and concisely.",
        f"Caller ID on file: {caller_number or 'unknown'}.",
        "",
        "Your goals:",
        "1. Greet the caller (by name if they are a known customer).",
        "2. Understand their intent: billing, sales, support, or general inquiry.",
        "3. Use get_open_tickets when a known customer may have account issues.",
        "4. Use record_intent once you understand what they need.",
        "5. If they want a person, are frustrated, or need something you cannot do, "
        "say you will connect them and call transfer_to_human.",
        "6. For unknown callers, ask their name and use update_lead_name.",
        "",
        "You may answer simple questions yourself. Do not invent account data.",
        "Never claim you connected them without calling transfer_to_human.",
    ]

    if customer:
        lines.extend(
            [
                "",
                "KNOWN CUSTOMER (from CRM):",
                f"- Name: {customer.get('name')}",
                f"- Company: {customer.get('company', 'n/a')}",
                f"- Preferred department: {customer.get('preferred_department', 'general')}",
            ]
        )
        if open_tickets:
            ticket_summary = "; ".join(
                f"{t.get('id')}: {t.get('subject')}" for t in open_tickets
            )
            lines.append(f"- Open deals/tickets ({len(open_tickets)}): {ticket_summary}")
            lines.append(
                "Mention open tickets briefly in your greeting if relevant."
            )
    elif lead_created:
        lines.append("")
        lines.append(
            "UNKNOWN CALLER: a new lead record was created. "
            "Welcome them and ask how you can help."
        )

    return "\n".join(lines)


def build_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "toolSpec": {
                "name": "get_open_tickets",
                "description": "Fetch open support tickets for the known customer",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                        },
                        "required": ["customer_id"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "record_intent",
                "description": "Record classified caller intent for routing logs",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "intent": {
                                "type": "string",
                                "enum": ["billing", "sales", "support", "general"],
                            },
                            "notes": {"type": "string"},
                        },
                        "required": ["intent"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "update_lead_name",
                "description": "Save the caller name for a new lead",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                        },
                        "required": ["name"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "transfer_to_human",
                "description": (
                    "Connect caller to a human agent. Say 'Let me connect you now' "
                    "then invoke this tool."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "department": {
                                "type": "string",
                                "enum": ["billing", "sales", "support", "general"],
                            },
                        },
                        "required": ["reason"],
                    }
                },
            }
        },
    ]


def transcript_to_text(lines: list[TranscriptLine]) -> str:
    return "\n".join(f"{line.role}: {line.text}" for line in lines if line.text.strip())


class ReceptionCallLog:
    def __init__(self, path: Path = RECEPTION_CALLS_PATH):
        self.path = path

    def append(self, entry: dict[str, Any]) -> None:
        data = {"calls": []}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
        data.setdefault("calls", []).append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class AIReceptionist:
    def __init__(self, db: CRMBackend | None = None) -> None:
        self.db = db or create_crm_backend()
        self.call_log = ReceptionCallLog()
        self._bridge: NovaSonicBridge | None = None
        self._current_customer: dict[str, Any] | None = None
        self._current_caller: str | None = None
        self._classified_intent: str | None = None

    async def handle_get_open_tickets(self, _tool: str, tool_input: dict[str, Any]) -> str:
        customer_id = str(tool_input.get("customer_id", "")).strip()
        if not customer_id and self._current_customer:
            customer_id = str(self._current_customer.get("id", ""))
        tickets = await self.db.get_open_tickets(customer_id)
        return json.dumps({"customer_id": customer_id, "open_tickets": tickets})

    async def handle_record_intent(self, _tool: str, tool_input: dict[str, Any]) -> str:
        self._classified_intent = str(tool_input.get("intent", "general"))
        return json.dumps(
            {
                "recorded": True,
                "intent": self._classified_intent,
                "notes": tool_input.get("notes", ""),
            }
        )

    async def handle_update_lead_name(self, _tool: str, tool_input: dict[str, Any]) -> str:
        name = str(tool_input.get("name", "")).strip()
        updated = await self.db.update_lead_name(self._current_caller, name)
        return json.dumps({"updated": updated, "name": name})

    async def handle_transfer_to_human(self, _tool: str, tool_input: dict[str, Any]) -> str:
        reason = str(tool_input.get("reason", "Caller requested human agent")).strip()
        department = str(tool_input.get("department", "general"))
        if self._bridge:
            self._bridge.request_handoff(f"{department}: {reason}")
        return json.dumps(
            {
                "transfer": True,
                "reason": reason,
                "department": department,
                "message": "Connecting caller to human agent now.",
            }
        )

    def make_bridge(self, system_prompt: str) -> NovaSonicBridge:
        self._bridge = NovaSonicBridge(
            model_id=NOVA_MODEL_ID,
            region=AWS_REGION,
            voice_id=NOVA_VOICE_ID,
            system_prompt=system_prompt,
            tools=build_tool_specs() if NOVA_ENABLE_TOOLS else [],
            tool_handlers={
                "get_open_tickets": self.handle_get_open_tickets,
                "record_intent": self.handle_record_intent,
                "update_lead_name": self.handle_update_lead_name,
                "transfer_to_human": self.handle_transfer_to_human,
            },
            handoff_grace_seconds=HANDOFF_AUDIO_GRACE_SECONDS,
        )
        return self._bridge

    async def perform_human_handoff(self, call: ActiveCall, reason: str) -> None:
        """Pre-answer connect to original callee, then leave (Telcoflow docs pattern)."""
        logger.info("Human handoff for call %s: %s", call.call_id, reason)
        await call.connect()
        await call.close()
        logger.info("Call %s connected to callee; AI left the call", call.call_id)

    async def handle_call(self, call: ActiveCall) -> dict[str, Any]:
        self._current_caller = call.caller_number
        self._classified_intent = None

        customer = await self.db.get_customer_by_phone(call.caller_number)
        lead_created = False
        lead_record: dict[str, Any] | None = None
        if customer:
            logger.info("Known customer: %s", customer.get("name"))
            open_tickets = await self.db.get_open_tickets(str(customer.get("id", "")))
        else:
            lead_record = await self.db.create_lead(call.caller_number, call.call_id)
            lead_created = bool(lead_record.get("created", True))
            open_tickets = []
            logger.info("New Pipedrive/CRM lead for %s: %s", call.caller_number, lead_record.get("id"))

        self._current_customer = customer
        system_prompt = build_system_prompt(
            customer=customer,
            open_tickets=open_tickets,
            lead_created=lead_created,
            caller_number=call.caller_number,
        )

        bridge = self.make_bridge(system_prompt)
        result = await bridge.run_call(call)

        if result.handoff_requested:
            await self.perform_human_handoff(call, result.handoff_reason)
            outcome = "handoff_to_human"
        else:
            try:
                await call.disconnect()
            except Exception:
                logger.debug("Call %s already disconnected", call.call_id)
            outcome = "ai_handled"

        entry = {
            "id": str(uuid.uuid4()),
            "call_id": result.call_id,
            "caller_number": result.caller_number,
            "customer_id": customer.get("id") if customer else None,
            "customer_name": customer.get("name") if customer else None,
            "lead_created": lead_created,
            "crm_backend": CRM_BACKEND,
            "lead_record": lead_record,
            "intent": self._classified_intent,
            "outcome": outcome,
            "handoff_reason": result.handoff_reason or None,
            "ended_at": datetime.now().isoformat(),
            "transcript": transcript_to_text(result.transcript),
            "tool_calls": result.tool_calls,
        }
        self.call_log.append(entry)
        return entry


async def main() -> None:
    receptionist = AIReceptionist()
    config = make_telcoflow_config()

    async with TelcoflowClient(config) as client:
        @client.on(events.INCOMING_CALL)
        async def on_call(call: ActiveCall) -> None:
            try:
                entry = await receptionist.handle_call(call)
                print(json.dumps({"call_id": call.call_id, "log": entry}, indent=2))
            except Exception as exc:
                logger.exception("Call %s failed", call.call_id)
                print(f"Call {call.call_id} failed: {exc}", file=sys.stderr)
                try:
                    await call.disconnect()
                except Exception:
                    pass

        logger.info(
            "AI receptionist listening (%s, CRM=%s, Nova %s)",
            COMPANY_NAME,
            CRM_BACKEND,
            NOVA_MODEL_ID,
        )
        await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
