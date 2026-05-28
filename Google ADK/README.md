# HealthFirst phone booking agent

This is a voice agent for **HealthFirst Clinic**. When someone calls, **Maya** answers, collects appointment details, checks the calendar, and books the visit after the call ends.

Built with:

- **[Telcoflow](https://docs.telcoflow.com)** — phone calls in and out
- **[Google ADK](https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/adk)** — live voice conversation with Maya
- **Google Calendar** — real appointment slots (not just a promise in the prompt)
- **Gemini** — turns the call transcript into structured booking data after hang-up
- **Telegram** (optional) — confirmations and alerts for staff

---

## What happens on a call

Here is the flow in plain English:

1. **Someone calls** → Telcoflow connects to your running agent.
2. **Maya talks** → She greets the caller, listens for name / phone / date / time / appointment type. If the caller gives everything upfront (“Hi, I’m Harshal, general checkup today at 5 PM”), she does not ask again.
3. **Calendar check during the call** → When date and time are known, the app checks Google Calendar and sends Maya a `[Calendar system]` message: slot available, or busy with alternatives.
4. **Maya confirms** → She read back the details and waits for “yes” / “all right” / similar.
5. **Caller hangs up** → Post-call runs automatically.
6. **Booking** → Gemini extracts the final details from the transcript, creates the Google Calendar event, saves to `bookings.json`, and sends Telegram if configured.

If something is unclear, the booking goes to **human review** and staff get a Telegram alert. If the slot is already taken, staff get an **unavailable** alert with suggested times.

```text
Phone call (Telcoflow)
        │
        ▼
  Maya (ADK live voice)
  + calendar check injected during call
        │
        ▼
   Call transcript
        │
        ▼
  Gemini extraction (post-call)
        │
        ▼
  Google Calendar + bookings.json
        │
        ▼
  Telegram confirmation or alert
```

---

## Before you start

You will need:

1. **Telcoflow** sandbox API key and connector UUID  
2. **Google API key** with Gemini access  
3. **Google Calendar service account** JSON file  
4. Your **calendar shared** with the service account email — give it **“Make changes to events”**  
5. **Python 3.11+**


---

## Setup (about 5 minutes)

```bash
cd adk_healthfirst
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your real values:

| Variable | What it is |
|----------|------------|
| `WSS_API_KEY` | Telcoflow API key |
| `WSS_CONNECTOR_UUID` | Telcoflow connector for your phone line |
| `GOOGLE_API_KEY` | Gemini / Google AI key |
| `GOOGLE_CALENDAR_CREDENTIALS` | Path to service account JSON **or** use `GOOGLE_CALENDAR_CREDENTIALS_JSON` |
| `GOOGLE_CALENDAR_ID` | Calendar email or ID where events are created |
| `TELEGRAM_BOT_TOKEN` | Optional — bot token |
| `TELEGRAM_CONFIRM_CHAT_ID` | Optional — your chat ID for messages |

**Calendar tip:** Open Google Calendar → Settings → your calendar → Share with the service account email from the JSON (looks like `something@project-id.iam.gserviceaccount.com`).

**Telcoflow tip:** `telcoflow-sdk` installs from TestPyPI — that is already handled in `requirements.txt`. You do not need to do anything extra.

---

## Run it

Start the agent and leave it running:

```bash
python run.py
```

You should see:

```text
HealthFirst ADK booking agent connected. Waiting for calls...
```

Then place a test call through Telcoflow.

### Modes

Set `AGENT_MODE` in `.env` or on the command line:

| Mode | What it does |
|------|----------------|
| `booking` | Inbound booking calls only (default) |
| `reminder` | Hourly Telegram reminders for upcoming appointments |
| `both` | Booking + reminders on one connection |

```bash
AGENT_MODE=booking python run.py
AGENT_MODE=reminder python run.py
AGENT_MODE=both python run.py
```

---

## Clinic hours and slot options

By default the clinic is open **9 AM – 5 PM** (Singapore timezone unless you change it). Appointments are **30 minutes**.

You can tune this in `.env`:

```env
CLINIC_TIMEZONE=Asia/Singapore
CLINIC_OPEN_HOUR=9
CLINIC_CLOSE_HOUR=17
APPOINTMENT_DURATION_MINUTES=30
```
---

## Project layout

| File | Role |
|------|------|
| `run.py` | Start here — picks mode from `AGENT_MODE` |
| `booking_agent.py` | Handles inbound calls |
| `adk_voice.py` | Bridges Telcoflow audio ↔ ADK (Maya) |
| `availability_inject.py` | Checks calendar during the call, tells Maya if slot is free |
| `post_call.py` | After hang-up: extract details, book calendar, notify |
| `google_calendar.py` | Google Calendar read/write |
| `bookings.json` | Local record of confirmed bookings |
| `notify.py` | Telegram confirmations and alerts |
| `prompts.py` | Maya’s voice instructions |
| `reminder_agent.py` | Reminder worker (optional) |
| `DEPLOYMENT.md` | Production notes (Cloud Run, Docker, etc.) |

---

## Docker

```bash
docker build -t healthfirst-adk .
docker run --env-file .env healthfirst-adk
```

For production deployment, read [`DEPLOYMENT.md`](DEPLOYMENT.md).

---

## Troubleshooting

| Problem | Likely cause | What to try |
|---------|----------------|-------------|
| `telcoflow-sdk` not found | PyPI vs TestPyPI | `pip install -r requirements.txt` (extra index is in the file) |
| No calendar events | Calendar not shared with service account | Share calendar with service account email |
| `needs_human_review` | Noisy transcript / conflicting phone or time | Speak clearly; confirm when Maya read back |
| `unavailable` | Slot already booked | Delete test events or choose another time |
| Call drops mid-conversation | Gemini Live hiccup | Post-call still runs if there is a partial transcript; try again |

---
