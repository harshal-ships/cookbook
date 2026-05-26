"""
Medical triage voice agent using Telcoflow, Amazon Nova Sonic 2, and LangGraph.

Run with:
    python triage_agent.py

Required environment variables:
    WSS_API_KEY
    WSS_CONNECTOR_UUID
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_SESSION_TOKEN
    AWS_REGION=us-east-1
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from operator import add
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict
from dotenv import load_dotenv

load_dotenv()


from aws_sdk_bedrock_runtime.client import (
    BedrockRuntimeClient,
    InvokeModelWithBidirectionalStreamOperationInput,
)
from aws_sdk_bedrock_runtime.config import Config
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
)
from langgraph.graph import END, START, StateGraph
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver
from telcoflow_sdk import ActiveCall, TelcoflowClient, TelcoflowClientConfig
from telcoflow_sdk.exceptions import BufferFullError, WSSCallCommandError
import telcoflow_sdk.events as events
from websockets.exceptions import ConnectionClosed

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # Older LangGraph releases used MemorySaver for the same in-memory checkpointer.
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver


# This section centralizes constants so the three systems keep clear responsibilities.
LOGGER = logging.getLogger("healthfirst_triage")
NOVA_MODEL_ID = "amazon.nova-2-sonic-v1:0"
SUPPORTED_AWS_REGION = "us-east-1"
TELCOFLOW_SAMPLE_RATE = 24000
TRIAGE_LOG_PATH = Path("triage_log.jsonl")
CLINIC_NAME = "HealthFirst Clinic"
DEFAULT_CLINIC_PHONE = "+6567504645"
CLINIC_PHONE = os.getenv("HEALTHFIRST_CLINIC_PHONE", DEFAULT_CLINIC_PHONE)


def format_phone_for_speech(phone: str) -> str:
    """Format E.164 numbers so Nova reads them clearly on a phone call."""
    digit_words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    digits = re.sub(r"\D", "", phone)
    if digits.startswith("65") and len(digits) >= 10:
        local = digits[2:10]
        spoken = " ".join(digit_words[digit] for digit in local)
        return f"plus six five, {spoken}"
    return phone

# Opening script Nova speaks once; LangGraph drives every later turn via [ROUTING] messages.
NOVA_OPENING_SCRIPT = (
    "I am a virtual assistant and not a doctor. If you are in immediate danger, please call 995 now. "
    "Hi, thank you for calling HealthFirst Clinic. I am Aria. May I have your name, please?"
)

# Nova is the voice layer only. LangGraph owns intake state and sends one [ROUTING] line per turn.
NOVA_SYSTEM_PROMPT = f"""You are Aria, a calm virtual triage assistant for HealthFirst Clinic. You are not a doctor and never diagnose.

You CANNOT book appointments. You CANNOT schedule visits. You CANNOT take appointment dates or times.
Never say "I can help you book", "I can book that for you", "let me schedule", or anything that implies you will book an appointment.
Your only job is triage: ask intake questions, then advise the caller what to do next.

At the very start of the call, say exactly once: "{NOVA_OPENING_SCRIPT}"

After the opening: never reintroduce yourself, never repeat the 995 disclaimer, and never ask your own questions.

When a message starts with [ROUTING], speak the quoted text exactly — nothing before it, nothing after it.

If the caller asks you to book an appointment, say you cannot book on this line and tell them to call {format_phone_for_speech(CLINIC_PHONE)} during business hours.
"""


def build_book_appointment_message() -> str:
    """Spoken LOW-urgency routing: caller must phone the clinic to book — Aria cannot."""
    spoken_phone = format_phone_for_speech(CLINIC_PHONE)
    return (
        "Thank you for sharing those details. I cannot book appointments on this line. "
        f"To schedule a visit at {CLINIC_NAME}, please call {spoken_phone} during business hours "
        "and our clinic team will help you book an appointment. "
        "If your symptoms worsen or you feel in immediate danger, please call 995."
    )

NEGATIVE_ASSOCIATED_PATTERNS = (
    r"\bno\b.{0,30}\b(other symptoms?|symptoms?)\b",
    r"\bnone\b",
    r"\bnothing else\b",
    r"\bno other\b",
    r"\bdon'?t have any other\b",
)
NEGATIVE_MEDICAL_PATTERNS = (
    r"\bno\b.{0,30}\b(medications?|medicines?|conditions?|medical)\b",
    r"\bnot on any\b",
    r"\bno known\b",
    r"\bnone\b.{0,20}\b(medications?|conditions?)\b",
)


# These dataclasses are the LangGraph-owned clinical state, not audio state.
UrgencyLevel = Literal["LOW", "MEDIUM", "HIGH"]
RoutingDecision = Literal[
    "continue_collection",
    "book_appointment",
    "transfer_to_nurse",
    "emergency_advisory",
]


@dataclass
class SymptomData:
    patient_name: str | None = None
    age: int | None = None
    main_symptom: str | None = None
    duration: str | None = None
    severity: int | None = None
    associated_symptoms: list[str] = field(default_factory=list)
    medical_context: str | None = None


class TriageState(TypedDict, total=False):
    latest_transcript: str
    transcript_history: Annotated[list[str], add]
    symptoms: dict[str, Any]
    urgency_level: UrgencyLevel
    urgency_reasons: list[str]
    routing_decision: RoutingDecision
    action_taken: str
    response_to_patient: str
    next_question: str | None
    aria_instruction: str


# This section extracts structured triage facts from transcripts with deterministic rules.
SYMPTOM_KEYWORDS = [
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "trouble breathing",
    "stroke",
    "weakness",
    "numbness",
    "severe headache",
    "headache",
    "abdominal pain",
    "stomach pain",
    "fever",
    "cough",
    "vomiting",
    "diarrhea",
    "dizziness",
    "fainting",
    "bleeding",
    "rash",
    "allergic reaction",
    "pain",
]

ASSOCIATED_SYMPTOM_KEYWORDS = [
    "fever",
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "trouble breathing",
    "sweating",
    "nausea",
    "vomiting",
    "dizziness",
    "fainting",
    "confusion",
    "weakness",
    "numbness",
    "slurred speech",
    "severe bleeding",
    "rash",
    "swelling",
]


def extract_symptoms(state: TriageState) -> TriageState:
    """Collect symptoms node: merge the newest patient transcript into structured state."""
    current = SymptomData(**state.get("symptoms", {}))
    transcript = state.get("latest_transcript", "")
    text = transcript.lower()

    if current.patient_name is None:
        name_match = re.search(r"\b(?:my name is|this is|i am|i'm)\s+([a-z][a-z .'-]{1,50})", transcript, re.I)
        if name_match and not re.search(r"\b(years?\s+old|calling about|having|experiencing)\b", name_match.group(1), re.I):
            current.patient_name = name_match.group(1).strip(" .")

    if current.age is None:
        age_match = re.search(r"\b(?:i am|i'm|age is|aged?)\s+(\d{1,3})\b|\b(\d{1,3})\s+years?\s+old\b", text)
        if age_match:
            age = int(age_match.group(1) or age_match.group(2))
            if 0 <= age <= 120:
                current.age = age

    if current.main_symptom is None:
        for keyword in SYMPTOM_KEYWORDS:
            if keyword in text:
                current.main_symptom = keyword
                break

    if current.duration is None:
        duration_match = re.search(
            r"\b(?:for|since|started|began)\s+((?:about\s+)?\d+\s+(?:minutes?|hours?|days?|weeks?)|"
            r"(?:a|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:minutes?|hours?|days?|weeks?)|"
            r"yesterday|today|last night|this morning)\b",
            text,
        )
        if duration_match:
            current.duration = duration_match.group(1).strip()

    if current.severity is None:
        severity_match = re.search(r"\b(10|[1-9])\s*(?:/|out of)\s*10\b", text)
        if severity_match is None:
            severity_match = re.search(r"\b(?:severity|pain|level)\D{0,20}(10|[1-9])\b", text)
        if severity_match:
            current.severity = int(severity_match.group(1))

    for keyword in ASSOCIATED_SYMPTOM_KEYWORDS:
        if keyword in text and keyword not in current.associated_symptoms:
            current.associated_symptoms.append(keyword)

    if not current.associated_symptoms and any(re.search(pattern, text) for pattern in NEGATIVE_ASSOCIATED_PATTERNS):
        current.associated_symptoms = ["none"]

    if current.medical_context is None and any(re.search(pattern, text) for pattern in NEGATIVE_MEDICAL_PATTERNS):
        current.medical_context = "none reported"
    elif current.medical_context is None and re.search(
        r"\b(medication|medicine|diabetes|heart|asthma|pregnant|condition)\b", text
    ):
        current.medical_context = transcript.strip()

    return {"symptoms": asdict(current)}


# This section applies deterministic clinical triage rules without diagnosing the patient.
def assess_urgency(state: TriageState) -> TriageState:
    """Assess urgency node: classify risk from red flags, severity, age, and duration."""
    symptoms = SymptomData(**state.get("symptoms", {}))
    associated = set(symptoms.associated_symptoms)
    main = (symptoms.main_symptom or "").lower()
    reasons: list[str] = []
    urgency: UrgencyLevel = "LOW"

    if main == "chest pain" and ({"shortness of breath", "difficulty breathing", "sweating", "nausea"} & associated):
        urgency = "HIGH"
        reasons.append("Chest pain with breathing difficulty, sweating, or nausea is an emergency red flag.")
    if {"difficulty breathing", "trouble breathing", "shortness of breath"} & ({main} | associated):
        if symptoms.severity is None or symptoms.severity >= 6:
            urgency = "HIGH"
            reasons.append("Moderate or severe breathing difficulty needs emergency care.")
    if {"slurred speech", "weakness", "numbness", "confusion", "fainting"} & associated:
        urgency = "HIGH"
        reasons.append("Neurologic symptoms or fainting can indicate an emergency.")
    if main in {"allergic reaction", "bleeding"} or "severe bleeding" in associated:
        urgency = "HIGH"
        reasons.append("Severe allergy symptoms or bleeding need immediate emergency assessment.")
    if symptoms.severity is not None and symptoms.severity >= 9:
        urgency = "HIGH"
        reasons.append("Pain severity of 9 or 10 out of 10 is treated as emergency-level risk.")

    if urgency != "HIGH":
        if symptoms.severity is not None and symptoms.severity >= 7:
            urgency = "MEDIUM"
            reasons.append("Severity of 7 or 8 out of 10 should be reviewed by a nurse today.")
        if main == "fever" and (symptoms.age is not None and (symptoms.age < 3 or symptoms.age >= 65)):
            urgency = "MEDIUM"
            reasons.append("Fever in very young children or older adults should be reviewed today.")
        if {"vomiting", "dizziness"} & ({main} | associated):
            urgency = "MEDIUM"
            reasons.append("Vomiting or dizziness can worsen quickly and should be nurse-triaged today.")
        if not reasons:
            reasons.append("No emergency red flags were detected from the available information.")

    return {"urgency_level": urgency, "urgency_reasons": reasons}


def next_missing_question(symptoms: SymptomData) -> str | None:
    """Return the single next intake question LangGraph still needs."""
    if symptoms.patient_name is None:
        return "May I have your name, please?"
    if symptoms.age is None:
        return "How old are you?"
    if symptoms.main_symptom is None:
        return "What is the main symptom you are experiencing today?"
    if symptoms.duration is None:
        return "How long have you had this symptom?"
    if symptoms.severity is None:
        return "On a scale of 1 to 10, with 10 being the worst, how severe is it?"
    if not associated_symptoms_collected(symptoms):
        return "Do you have any other symptoms, such as fever, chest pain, or difficulty breathing?"
    if symptoms.medical_context is None:
        return "Do you have any known medical conditions or take any medications?"
    return None


def associated_symptoms_collected(symptoms: SymptomData) -> bool:
    """True once the caller listed associated symptoms or explicitly said there are none."""
    return bool(symptoms.associated_symptoms)


def build_aria_instruction(
    symptoms: SymptomData,
    routing_decision: RoutingDecision,
    response: str,
    next_question: str | None,
) -> str:
    """Build the single spoken line LangGraph hands to Nova."""
    if routing_decision == "continue_collection" and next_question:
        if symptoms.patient_name:
            return f"Thank you, {symptoms.patient_name}. {next_question}"
        return next_question
    return response


def triage_state_fingerprint(state: TriageState) -> str:
    """Fingerprint LangGraph output so Nova only gets a new [ROUTING] line when something changed."""
    return json.dumps(
        {
            "symptoms": state.get("symptoms", {}),
            "urgency_level": state.get("urgency_level"),
            "routing_decision": state.get("routing_decision"),
            "aria_instruction": state.get("aria_instruction"),
        },
        sort_keys=True,
    )


def route_decision(state: TriageState) -> TriageState:
    """Route decision node: produce the exact routing action and spoken response."""
    symptoms = SymptomData(**state.get("symptoms", {}))
    urgency = state.get("urgency_level", "LOW")
    next_question = next_missing_question(symptoms)

    if urgency == "HIGH":
        decision: RoutingDecision = "emergency_advisory"
        action = "Patient advised to call 995 immediately"
        response = (
            "Based on what you told me, this may need emergency care. "
            "Please call 995 immediately and stay on the line with emergency services. "
            "I am not a doctor, but I do not want you to wait with these symptoms."
        )
    elif next_missing_question(symptoms) is not None:
        decision = "continue_collection"
        action = "Continue collecting triage details"
        response = next_missing_question(symptoms) or "Is there anything else about your symptoms I should note?"
    elif urgency == "MEDIUM":
        decision = "transfer_to_nurse"
        action = "Patient should speak to a nurse today"
        response = (
            "Thank you for sharing those details. I am going to transfer you to a nurse right away "
            "so they can review your symptoms today."
        )
    else:
        decision = "book_appointment"
        action = f"Patient advised to call {CLINIC_PHONE} to book at {CLINIC_NAME}"
        response = build_book_appointment_message()

    aria_instruction = build_aria_instruction(symptoms, decision, response, next_question)

    return {
        "routing_decision": decision,
        "action_taken": action,
        "response_to_patient": response,
        "next_question": next_question,
        "aria_instruction": aria_instruction,
    }


def build_triage_graph():
    """Build the thread-aware LangGraph state machine for per-call triage decisions."""
    graph = StateGraph(TriageState)
    graph.add_node("collect_symptoms", extract_symptoms)
    graph.add_node("assess_urgency", assess_urgency)
    graph.add_node("route_decision", route_decision)
    graph.add_edge(START, "collect_symptoms")
    graph.add_edge("collect_symptoms", "assess_urgency")
    graph.add_edge("assess_urgency", "route_decision")
    graph.add_edge("route_decision", END)
    return graph.compile(checkpointer=InMemorySaver())


# This section logs the latest triage result in the requested JSON object shape.
def write_triage_log(call_id: str, state: TriageState) -> None:
    symptoms = SymptomData(**state.get("symptoms", {}))
    record = {
        "call_id": call_id,
        "patient_name": symptoms.patient_name,
        "age": symptoms.age,
        "symptoms": {
            "main_symptom": symptoms.main_symptom,
            "duration": symptoms.duration,
            "severity": symptoms.severity,
            "associated_symptoms": symptoms.associated_symptoms,
        },
        "urgency_level": state.get("urgency_level"),
        "routing_decision": state.get("routing_decision"),
        "action_taken": state.get("action_taken"),
        "timestamp": datetime.now().replace(microsecond=0).isoformat(),
    }
    with TRIAGE_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")


# This class owns only Bedrock Nova Sonic 2 bidirectional audio and transcript events.
# Event sequencing follows nova_sonic_integration.py, which is the working Telcoflow reference.
class NovaSonicTriageSession:
    def __init__(
        self,
        call: ActiveCall,
        bedrock_client: BedrockRuntimeClient,
        triage_graph: Any,
    ) -> None:
        self.call = call
        self.call_id = call.call_id
        self.bedrock_client = bedrock_client
        self.triage_graph = triage_graph
        self.prompt_name = str(uuid.uuid4())
        self.system_content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.stream: Any | None = None
        self.is_active = False
        self.latest_triage_state: TriageState | None = None
        self.content_metadata: dict[str, dict[str, Any]] = {}
        self.text_buffers: dict[str, list[str]] = {}
        self._send_to_nova_task: asyncio.Task | None = None
        self._recv_from_nova_task: asyncio.Task | None = None
        self._last_injected_fingerprint: str | None = None
        self._transfer_attempted = False

    async def send_event(self, event_json: str) -> None:
        """Send one JSON event string into Nova Sonic's bidirectional stream."""
        if self.stream is None:
            raise RuntimeError("Nova Sonic stream has not been started.")
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode("utf-8"))
        )
        await self.stream.input_stream.send(event)

    async def start_session(self) -> None:
        """Start Nova Sonic using the same event order as nova_sonic_integration.py."""
        self.stream = await self.bedrock_client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=NOVA_MODEL_ID)
        )
        self.is_active = True

        session_start = """
        {
          "event": {
            "sessionStart": {
              "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
              }
            }
          }
        }
        """
        await self.send_event(session_start)

        prompt_start = f"""
        {{
          "event": {{
            "promptStart": {{
              "promptName": "{self.prompt_name}",
              "textOutputConfiguration": {{
                "mediaType": "text/plain"
              }},
              "audioOutputConfiguration": {{
                "mediaType": "audio/lpcm",
                "sampleRateHertz": {TELCOFLOW_SAMPLE_RATE},
                "sampleSizeBits": 16,
                "channelCount": 1,
                "voiceId": "tiffany",
                "encoding": "base64",
                "audioType": "SPEECH"
              }}
            }}
          }}
        }}
        """
        await self.send_event(prompt_start)

        text_content_start = f"""
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.system_content_name}",
                    "type": "TEXT",
                    "interactive": false,
                    "role": "SYSTEM",
                    "textInputConfiguration": {{
                        "mediaType": "text/plain"
                    }}
                }}
            }}
        }}
        """
        await self.send_event(text_content_start)

        text_input = json.dumps(
            {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.system_content_name,
                        "content": NOVA_SYSTEM_PROMPT,
                    }
                }
            }
        )
        await self.send_event(text_input)

        text_content_end = f"""
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.system_content_name}"
                }}
            }}
        }}
        """
        await self.send_event(text_content_end)

        audio_content_start = f"""
        {{
            "event": {{
                "contentStart": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{self.audio_content_name}",
                    "type": "AUDIO",
                    "interactive": true,
                    "role": "USER",
                    "audioInputConfiguration": {{
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": {TELCOFLOW_SAMPLE_RATE},
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }}
                }}
            }}
        }}
        """
        await self.send_event(audio_content_start)

    async def stream_telcoflow_audio_to_nova(self) -> None:
        """Telcoflow phone audio enters here and is passed directly to Nova Sonic at 24 kHz."""
        try:
            async for audio_chunk in self.call.audio_stream():
                if not self.is_active:
                    break
                if not audio_chunk:
                    continue
                blob = base64.b64encode(audio_chunk)
                audio_event = f"""
                {{
                    "event": {{
                        "audioInput": {{
                            "promptName": "{self.prompt_name}",
                            "contentName": "{self.audio_content_name}",
                            "content": "{blob.decode('utf-8')}"
                        }}
                    }}
                }}
                """
                await self.send_event(audio_event)
        except ConnectionClosed:
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            if self.is_active:
                LOGGER.exception("Error streaming Telcoflow audio to Nova for call %s", self.call_id)
            raise

    async def process_nova_events_to_telcoflow(self) -> None:
        """Nova audio returns here; user transcripts are routed through LangGraph."""
        try:
            while self.is_active:
                if not self.stream:
                    await asyncio.sleep(0.1)
                    continue

                output = await self.stream.await_output()
                result = await output[1].receive()
                if not result.value or not result.value.bytes_:
                    continue

                event_payload = json.loads(result.value.bytes_.decode("utf-8"))
                await self._handle_nova_event(event_payload)
        except ConnectionClosed:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as error:
            message = str(error)
            if "ExpiredTokenException" in message:
                LOGGER.error(
                    "AWS session token expired for call %s. Refresh AWS SSO credentials in .env and restart.",
                    self.call_id,
                )
            elif "ValidationException" in message:
                LOGGER.error(
                    "Nova Sonic rejected an input event for call %s: %s",
                    self.call_id,
                    message,
                )
            elif self.is_active:
                LOGGER.exception("Nova Sonic event processing failed for call %s", self.call_id)
        finally:
            self.is_active = False

    async def _handle_nova_event(self, payload: dict[str, Any]) -> None:
        """Dispatch Nova output events by type while keeping audio and triage separated."""
        event = payload.get("event", {})
        if "contentStart" in event:
            self._handle_content_start(event["contentStart"])
        elif "textOutput" in event:
            text_content = event["textOutput"].get("content", "")
            if '{ "interrupted" : true }' in text_content:
                await self.call.clear_send_audio_buffer()
            self._handle_text_output(event["textOutput"])
        elif "audioOutput" in event:
            audio = base64.b64decode(event["audioOutput"]["content"])
            try:
                await self.call.send_audio(audio)
            except BufferFullError:
                LOGGER.warning(
                    "Telcoflow send buffer full for call %s; clearing stale audio.",
                    self.call_id,
                )
                await self.call.clear_send_audio_buffer()
        elif "contentEnd" in event:
            await self._handle_content_end(event["contentEnd"])

    def _handle_content_start(self, content_start: dict[str, Any]) -> None:
        """Track content block metadata so text chunks can be classified by role."""
        content_id = content_start.get("contentId") or content_start.get("contentName")
        if content_id is None:
            return
        additional_fields = {}
        if content_start.get("additionalModelFields"):
            try:
                additional_fields = json.loads(content_start["additionalModelFields"])
            except json.JSONDecodeError:
                additional_fields = {}
        self.content_metadata[content_id] = {
            "role": content_start.get("role"),
            "type": content_start.get("type"),
            "generation_stage": additional_fields.get("generationStage"),
        }
        if content_start.get("type") == "TEXT":
            self.text_buffers[content_id] = []

    def _handle_text_output(self, text_output: dict[str, Any]) -> None:
        """Buffer final user ASR transcripts; assistant text is informational only."""
        content_id = text_output.get("contentId") or text_output.get("contentName")
        if content_id is None:
            return
        self.text_buffers.setdefault(content_id, []).append(text_output.get("content", ""))

    async def _handle_content_end(self, content_end: dict[str, Any]) -> None:
        """Close content blocks, clear interrupted audio, and run LangGraph on patient turns."""
        if content_end.get("stopReason") == "INTERRUPTED":
            await self.call.clear_send_audio_buffer()

        content_id = content_end.get("contentId") or content_end.get("contentName")
        metadata = self.content_metadata.get(content_id or "", {})
        if metadata.get("type") == "TEXT" and metadata.get("role") == "USER":
            transcript = "".join(self.text_buffers.get(content_id or "", [])).strip()
            if transcript and not transcript.startswith("[ROUTING]"):
                await self._run_langgraph_triage(transcript)

    async def _run_langgraph_triage(self, transcript: str) -> None:
        """Invoke LangGraph with call.call_id as the isolated thread ID."""
        config = {"configurable": {"thread_id": self.call_id}}
        state = await asyncio.to_thread(
            self.triage_graph.invoke,
            {"latest_transcript": transcript, "transcript_history": [transcript]},
            config,
        )
        self.latest_triage_state = state
        write_triage_log(self.call_id, state)

        fingerprint = triage_state_fingerprint(state)
        if fingerprint == self._last_injected_fingerprint:
            LOGGER.debug("Call %s triage unchanged; skipping Nova routing injection", self.call_id)
            return

        self._last_injected_fingerprint = fingerprint
        LOGGER.info(
            "Call %s triage=%s route=%s",
            self.call_id,
            state.get("urgency_level"),
            state.get("routing_decision"),
        )

        instruction = state.get("aria_instruction", "")
        if instruction:
            await self._inject_routing_instruction(instruction)
        await self._execute_routing_action(state)

    async def _inject_routing_instruction(self, instruction: str) -> None:
        """Hand LangGraph's single spoken line to Nova."""
        guidance_content_name = str(uuid.uuid4())
        routing_message = f"[ROUTING] Say exactly: {json.dumps(instruction)}"

        content_start = json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": guidance_content_name,
                        "type": "TEXT",
                        "interactive": True,
                        "role": "USER",
                        "textInputConfiguration": {"mediaType": "text/plain"},
                    }
                }
            }
        )
        text_input = json.dumps(
            {
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": guidance_content_name,
                        "content": routing_message,
                    }
                }
            }
        )
        content_end = f"""
        {{
            "event": {{
                "contentEnd": {{
                    "promptName": "{self.prompt_name}",
                    "contentName": "{guidance_content_name}"
                }}
            }}
        }}
        """
        await self.send_event(content_start)
        await self.send_event(text_input)
        await self.send_event(content_end)

    async def _execute_routing_action(self, state: TriageState) -> None:
        """Perform Telcoflow actions that match LangGraph routing decisions."""
        if state.get("routing_decision") != "transfer_to_nurse" or self._transfer_attempted:
            return

        self._transfer_attempted = True
        try:
            await self.call.connect(ring_time_seconds=30)
            LOGGER.info("Nurse transfer connected for call %s", self.call_id)
            await self.call.close()
        except WSSCallCommandError as error:
            if error.error_code == "CALL_UNANSWERED":
                LOGGER.warning("Nurse line did not answer for call %s", self.call_id)
            else:
                LOGGER.warning("Nurse transfer failed for call %s: %s", self.call_id, error)
        except Exception as error:
            LOGGER.warning("Nurse transfer failed for call %s: %s", self.call_id, error)

    async def end_session(self) -> None:
        """Close Nova Sonic in the same order as nova_sonic_integration.py."""
        if not self.stream:
            return

        try:
            audio_content_end = f"""
            {{
                "event": {{
                    "contentEnd": {{
                        "promptName": "{self.prompt_name}",
                        "contentName": "{self.audio_content_name}"
                    }}
                }}
            }}
            """
            await self.send_event(audio_content_end)

            prompt_end = f"""
            {{
                "event": {{
                    "promptEnd": {{
                        "promptName": "{self.prompt_name}"
                    }}
                }}
            }}
            """
            await self.send_event(prompt_end)

            session_end = """
            {
                "event": {
                    "sessionEnd": {}
                }
            }
            """
            await self.send_event(session_end)
        except Exception:
            pass
        finally:
            if self.stream:
                await self.stream.input_stream.close()
                self.stream = None

    async def _on_terminated(self) -> None:
        """Stop Nova tasks cleanly when Telcoflow ends the call."""
        if not self.is_active:
            return

        LOGGER.info("Call terminated: %s", self.call_id)
        self.is_active = False

        if self._send_to_nova_task:
            self._send_to_nova_task.cancel()
        if self._recv_from_nova_task:
            self._recv_from_nova_task.cancel()

        tasks = [task for task in [self._send_to_nova_task, self._recv_from_nova_task] if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        try:
            await self.end_session()
        except Exception:
            pass

    async def run(self) -> None:
        """Bridge Telcoflow audio to Nova Sonic using the reference integration lifecycle."""
        self.call.register_event_handler(events.CALL_TERMINATED, self._on_terminated)
        await self.start_session()

        self._send_to_nova_task = asyncio.create_task(self.stream_telcoflow_audio_to_nova())
        self._recv_from_nova_task = asyncio.create_task(self.process_nova_events_to_telcoflow())

        await self._send_to_nova_task
        await self._recv_from_nova_task


def create_bedrock_client() -> BedrockRuntimeClient:
    """Create a shared Bedrock client using the same config as nova_sonic_integration.py."""
    region = get_aws_region() or SUPPORTED_AWS_REGION
    aws_config = Config(
        endpoint_uri=f"https://bedrock-runtime.{region}.amazonaws.com",
        region=region,
        aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
    )
    return BedrockRuntimeClient(config=aws_config)


# This section owns only Telcoflow call lifecycle and connects it to the Nova session.
async def handle_call(
    call: ActiveCall,
    bedrock_client: BedrockRuntimeClient,
    triage_graph: Any,
) -> None:
    LOGGER.info("Incoming call %s from %s", call.call_id, call.caller_number)

    @call.on(events.CALL_ERROR)
    def on_call_error(data: dict[str, Any]) -> None:
        LOGGER.error("Call error for %s: %s", call.call_id, data)

    await call.answer()
    session = NovaSonicTriageSession(call, bedrock_client, triage_graph)
    try:
        await session.run()
    except Exception:
        LOGGER.exception("Error in triage call %s", call.call_id)
        raise
    finally:
        await session.end_session()
        await call.close()


# This section validates configuration before opening any network streams.
def get_aws_region() -> str | None:
    """Read AWS region from .env-compatible names and normalize it for the SDK."""
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or SUPPORTED_AWS_REGION
    os.environ["AWS_REGION"] = region
    return region


def require_environment() -> None:
    required = [
        "WSS_API_KEY",
        "WSS_CONNECTOR_UUID",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ]
    missing = [name for name in required if not os.getenv(name)]
    aws_region = get_aws_region()
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if aws_region != SUPPORTED_AWS_REGION:
        raise RuntimeError("AWS_REGION or AWS_DEFAULT_REGION must be us-east-1 for Amazon Nova Sonic 2.")


# This is the runnable entrypoint: Telcoflow handles calls, Nova handles audio, LangGraph handles triage.
async def main() -> None:
    require_environment()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    triage_graph = build_triage_graph()
    bedrock_client = create_bedrock_client()
    config = TelcoflowClientConfig.sandbox(
        api_key=os.environ["WSS_API_KEY"],
        connector_uuid=os.environ["WSS_CONNECTOR_UUID"],
        buffer_size=1024 * 1024,
        sample_rate=TELCOFLOW_SAMPLE_RATE,
    )

    async with TelcoflowClient(config) as client:
        LOGGER.info("Connected to Telcoflow. Waiting for HealthFirst Clinic calls.")

        @client.on(events.INCOMING_CALL)
        async def on_incoming_call(call: ActiveCall) -> None:
            await handle_call(call, bedrock_client, triage_graph)

        await client.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
