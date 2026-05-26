# HealthFirst Triage Voice Agent

A phone-based medical triage assistant named **Aria** for HealthFirst Clinic.

When someone calls in, Aria has a voice conversation, asks about their symptoms, decides how urgent the case is, and tells them what to do next. She is a triage assistant — not a doctor — and does not give diagnoses.

## What happens on a call

1. **Caller rings in** through Telcoflow (the phone layer).
2. **Aria talks** using Amazon Nova Sonic 2 on AWS Bedrock (native speech in and out — no separate speech-to-text or text-to-speech).
3. **After each patient turn**, the transcript goes to a **LangGraph** workflow that:
   - Collects symptom details (name, age, main symptom, duration, severity, and related info)
   - Assesses urgency: **LOW**, **MEDIUM**, or **HIGH**
   - Chooses a route

## What Aria does at each urgency level

| Urgency | What Aria tells the caller |
|--------|----------------------------|
| **LOW** | She **cannot book** on this line. Call **+65XXXXXX5** during business hours and the clinic team will schedule an appointment. |
| **MEDIUM** | She tries to **transfer the caller to a nurse** for same-day review. |
| **HIGH** | She tells them to **call 995 immediately** (Singapore emergency services). |

Every call starts with a safety line: *"I am a virtual assistant and not a doctor. If you are in immediate danger, please call 995 now."*

## How the pieces fit together

```
Phone call → Telcoflow → Nova Sonic 2 (voice) → audio back to caller
                              ↕
                   Transcript → LangGraph (triage logic only)
```

- **Telcoflow** — handles the phone call and audio stream
- **Nova Sonic 2** — conducts the conversation and speaks
- **LangGraph** — triage rules, urgency scoring, and routing decisions (no audio)

## Run it

```bash
python3.12 -m venv .venv
source .venv/bin/activate

pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ -r requirements.txt

python triage_agent.py
```

Create a `.env` file in this folder:

```env
WSS_API_KEY=your_telcoflow_key
WSS_CONNECTOR_UUID=your_connector_uuid

AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_REGION=us-east-1
HEALTHFIRST_CLINIC_PHONE=
```

Nova Sonic 2 only runs in **`us-east-1`**. Refresh AWS SSO credentials when the session token expires.

## Logs

Each triage result is appended to **`triage_log.jsonl`** (one JSON object per line) with call ID, symptoms, urgency, and routing decision.
