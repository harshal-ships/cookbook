# CrewAI + Telcoflow + Gemini Live — call triage

B3networks **call triage** voice stack:

- **Telcoflow** — answer, stream caller audio, play agent audio, clear buffer on interrupt  
- **Gemini Live** — speech layer (listen / speak)  
- **CrewAI** — after each **caller** turn, runs three agents in order: **Receptionist → Analyst → Resolver**. The Resolver’s text is what gets spoken back via Gemini  

The voice persona name is **Crew** (not Nova). All CrewAI agents use **Gemini 2.0 Flash** via `langchain_google_genai` (no OpenAI key).

| Script | Description |
|--------|-------------|
| `crew_agent.py` | Full pipeline: Telcoflow + Gemini Live + sequential crew per turn |

## Run

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python crew_agent.py
```

## Environment

`WSS_API_KEY`, `WSS_CONNECTOR_UUID`, `GOOGLE_API_KEY` (Gemini Live + all CrewAI LLM calls).
