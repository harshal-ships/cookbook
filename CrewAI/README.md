# CrewAI + Telcoflow — call triage

## Overview

**Crew** is a B3networks **customer-care call triage** voice agent. After each **caller turn** is turned into text by the speech layer, a **CrewAI** crew decides what to say next: three specialised agents run **in sequence**, and only the **Resolver**'s output is spoken back to the caller. Use this pattern when you want a **multi-step reasoning / handoff** workflow (reception → categorise → reply) on every turn, instead of a single model driving the whole call.

- **Telcoflow** — answer, stream caller audio, play agent audio, clear buffer on interrupt  
- **Speech layer** — **Gemini Live** (`crew_gemini_agent.py`) or **Amazon Nova Sonic 2** (`crew_bedrock_agent.py`): listen / speak for the phone  
- **CrewAI** — after each **caller** turn: **Receptionist → Analyst → Resolver** (all use **`crewai.LLM` → `gemini/gemini-2.0-flash`** and `GOOGLE_API_KEY`)

The voice persona name is **Crew** (not Nova).

| Script | Speech layer | Description |
|--------|----------------|-------------|
| `crew_gemini_agent.py` | Gemini Live | Telcoflow + Gemini + sequential crew per turn |
| `crew_bedrock_agent.py` | Nova Sonic 2 (Bedrock) | Same crew; **24 kHz → 16 kHz** to Nova; **IAM** `AWS_*` for streaming |

## Run

Use **Python 3.12+** (Bedrock streaming client).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python crew_gemini_agent.py
# or
.venv/bin/python crew_bedrock_agent.py
```

## Environment

- **All scripts:** `WSS_API_KEY`, `WSS_CONNECTOR_UUID`, `GOOGLE_API_KEY` (CrewAI agents; Gemini Live also needs this for `crew_gemini_agent.py`).
- **`crew_bedrock_agent.py` only:** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` (Nova Sonic bidirectional stream — same idea as `LangGraph/nova_bedrock_agent.py`).
